'use strict';
/**
 * llama.cpp adapter — main-process side, Claude-owned. Registers the
 * manifest with registry.js and owns the one llama worker process via
 * host.js. main.js's chat:load / chat:send / chat:stop / chat:unload IPC
 * handlers delegate here now instead of loading node-llama-cpp in-process —
 * same public function names/shapes main.js already had, so nothing else
 * about the app needed to change.
 */
const path = require('path');
const host = require('../../engine/host');
const scheduler = require('../../engine/scheduler');

const WORKER_ID = 'llama';
const WORKER_MODULE = path.join(__dirname, 'worker.js');

const manifest = {
  id: 'llama',
  version: '1',
  jobs: ['chat', 'embed', 'rerank'],
  accepts: [{ format: 'gguf' }],
  devices: ['gpu', 'cpu'],
  requirements: { minRamGb: 1 },
  caps: { tensorSplit: true, streaming: true },
};

function jobIdFor(modelId) {
  return `chat:${modelId}`;
}

/**
 * Unlike a native-binary adapter (sd.cpp/whisper.cpp in later phases), llama
 * has nothing to lazy-fetch: node-llama-cpp ships prebuilt binaries for every
 * backend it supports as part of the npm package already installed, and
 * npm's own integrity check already covers that download — a second
 * sha256-pin here would just be re-checking a file npm already verified.
 * What actually needs checking is whether the REQUESTED backend has a real,
 * working binary + drivers on this machine right now.
 */
async function ensureInstalled(backend) {
  if (!backend || backend === 'cpu') return { ok: true, backend: 'cpu' };
  try {
    const { getLlamaGpuTypes } = await import('node-llama-cpp');
    const supported = new Set(await getLlamaGpuTypes('supported'));
    if (!supported.has(backend)) {
      return {
        ok: false,
        reason: `No working ${backend} backend on this machine (missing drivers, or no prebuilt binary for this platform/arch).`,
      };
    }
    return { ok: true, backend };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  }
}

function ensureWorker() {
  if (host.health(WORKER_ID).running) return Promise.resolve();
  return new Promise((resolve, reject) => {
    host.spawn(WORKER_ID, WORKER_MODULE);
    const onMsg = (msg) => {
      if (msg.type === 'hello') {
        host.off(WORKER_ID, onMsg);
        resolve();
      } else if (msg.type === 'exit') {
        host.off(WORKER_ID, onMsg);
        reject(new Error('llama worker exited before starting up'));
      }
    };
    host.on(WORKER_ID, onMsg);
  });
}

// Resolves/rejects once a 'result'/'error' for the given op+modelId arrives,
// or the worker exits mid-flight — never leaves a dangling listener behind.
function waitFor(op, modelId) {
  return new Promise((resolve, reject) => {
    const onMsg = (msg) => {
      if (msg.type === 'exit') {
        host.off(WORKER_ID, onMsg);
        return reject(new Error(`llama worker crashed (code ${msg.code ?? 'null'}, signal ${msg.signal ?? 'null'})`));
      }
      if (msg.op !== op || (modelId && msg.modelId && msg.modelId !== modelId)) return;
      if (msg.type === 'error') {
        host.off(WORKER_ID, onMsg);
        return reject(new Error(msg.message || 'adapter error'));
      }
      if (msg.type === 'result') {
        host.off(WORKER_ID, onMsg);
        return resolve(msg);
      }
    };
    host.on(WORKER_ID, onMsg);
  });
}

/** hwEstimate: best-effort { ramGb, vramGb, device } — Phase 1 keeps this loose. */
async function load(modelId, file, opts, hwEstimate = {}) {
  await ensureWorker();
  const device = hwEstimate.device === 'gpu' ? 'gpu' : 'cpu';
  const reserved = scheduler.reserve(jobIdFor(modelId), {
    adapterId: manifest.id,
    ramGb: hwEstimate.ramGb || 0,
    vramGb: hwEstimate.vramGb || 0,
    device,
    tensorSplit: manifest.caps.tensorSplit,
  });
  if (!reserved.ok) throw new Error(reserved.reason);
  const pending = waitFor('load', modelId);
  host.send(WORKER_ID, { op: 'load', modelId, file, opts });
  try {
    return await pending;
  } catch (e) {
    scheduler.release(jobIdFor(modelId));
    throw e;
  }
}

// Tracks which modelId currently has generation in flight, so stop() can
// answer synchronously (matching the old in-process function's signature —
// every caller, including the selftest, calls it without awaiting) instead
// of needing an IPC round-trip just to know whether there was anything to
// stop.
let generatingModelId = null;

function send(modelId, message, opts, onChunk) {
  return new Promise((resolve, reject) => {
    generatingModelId = modelId;
    const settle = (fn, arg) => {
      if (generatingModelId === modelId) generatingModelId = null;
      fn(arg);
    };
    const onMsg = (msg) => {
      if (msg.type === 'exit') {
        host.off(WORKER_ID, onMsg);
        return settle(reject, new Error(`llama worker crashed (code ${msg.code ?? 'null'}, signal ${msg.signal ?? 'null'})`));
      }
      if (msg.type === 'chunk' && msg.modelId === modelId) {
        onChunk?.(msg.text);
        return;
      }
      if (msg.op !== 'send' || (msg.modelId && msg.modelId !== modelId)) return;
      if (msg.type === 'error') {
        host.off(WORKER_ID, onMsg);
        return settle(reject, new Error(msg.message || 'generation failed'));
      }
      if (msg.type === 'result') {
        host.off(WORKER_ID, onMsg);
        return settle(resolve, msg);
      }
    };
    host.on(WORKER_ID, onMsg);
    host.send(WORKER_ID, { op: 'send', modelId, message, opts });
  });
}

function stop(modelId) {
  if (generatingModelId !== modelId || !host.health(WORKER_ID).running) return { stopped: false };
  host.send(WORKER_ID, { op: 'stop', modelId }); // fire-and-forget; send()'s promise settles when the worker acks
  return { stopped: true };
}

async function unload(modelId) {
  if (host.health(WORKER_ID).running) {
    const pending = waitFor('unload', null).catch(() => {}); // best-effort ack
    host.send(WORKER_ID, { op: 'unload' });
    await Promise.race([pending, new Promise((r) => setTimeout(r, 2000))]);
  }
  if (modelId) scheduler.release(jobIdFor(modelId));
}

function embedJobIdFor(modelId) {
  return `embed:${modelId}`;
}

/**
 * Phase 5: embeddings via node-llama-cpp's own createEmbeddingContext — the
 * same engine already loaded for chat, no new binary/runner. Each call
 * reserves/releases the scheduler lock around itself (embedding models are
 * small and fast; the worker's own embedState cache avoids reloading the
 * same model on repeated calls, per worker.js).
 */
async function embed(modelId, file, text) {
  await ensureWorker();
  const jobId = embedJobIdFor(modelId);
  const reserved = scheduler.reserve(jobId, { adapterId: manifest.id, device: 'cpu', tensorSplit: false });
  if (!reserved.ok) throw new Error(reserved.reason);
  const pending = waitFor('embed', modelId);
  host.send(WORKER_ID, { op: 'embed', modelId, file, text });
  try {
    const result = await pending;
    return { vector: result.vector };
  } finally {
    scheduler.release(jobId);
  }
}

function rerankJobIdFor(modelId) {
  return `rerank:${modelId}`;
}

/** Same pattern as embed — node-llama-cpp's createRankingContext, no new binary. */
async function rerank(modelId, file, query, documents) {
  await ensureWorker();
  const jobId = rerankJobIdFor(modelId);
  const reserved = scheduler.reserve(jobId, { adapterId: manifest.id, device: 'cpu', tensorSplit: false });
  if (!reserved.ok) throw new Error(reserved.reason);
  const pending = waitFor('rerank', modelId);
  host.send(WORKER_ID, { op: 'rerank', modelId, file, query, documents });
  try {
    const result = await pending;
    return { ranked: result.ranked };
  } finally {
    scheduler.release(jobId);
  }
}

module.exports = { manifest, load, send, stop, unload, embed, rerank, ensureWorker, ensureInstalled, jobIdFor };
