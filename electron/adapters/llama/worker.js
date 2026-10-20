'use strict';
/**
 * llama.cpp adapter worker — Claude-owned. Runs node-llama-cpp in its own
 * OS process (spawned via host.js's fork()) so a crash here can never take
 * the Electron window down with it. This is a line-for-line move of what
 * used to be main.js's in-process loadChatModel/sendChatMessage/
 * stopGeneration/unloadChat — same node-llama-cpp calls, same error
 * messages, just relocated and talking back over process.send() instead of
 * calling win.webContents.send() directly.
 */

let llamaInstance = null;
let chat = null; // { modelId, model, context, session, settings, abortController }
let embedState = null; // { modelId, model, context } — separate from chat; an embedding model may differ from the loaded chat model
let rankState = null; // { modelId, model, context } — separate again; a reranker model differs from both chat and embedding models

function send(msg) {
  process.send?.(msg);
}

function settingsChanged(a, b) {
  return (
    (a?.contextSize || null) !== (b?.contextSize || null) ||
    (a?.systemPrompt || '') !== (b?.systemPrompt || '') ||
    (a?.gpuLayers ?? 'auto') !== (b?.gpuLayers ?? 'auto')
  );
}

async function unload() {
  if (!chat) return;
  const { session, context, model } = chat;
  chat = null;
  try {
    session?.dispose?.();
  } catch {
    /* ignore */
  }
  await context?.dispose?.().catch(() => {});
  await model?.dispose?.().catch(() => {});
}

async function load({ modelId, file, opts = {} }) {
  const settings = {
    contextSize: opts.contextSize || null,
    systemPrompt: opts.systemPrompt || '',
    gpuLayers: opts.gpuLayers ?? 'auto',
  };
  if (chat?.modelId === modelId && !settingsChanged(chat.settings, settings)) {
    send({ type: 'result', op: 'load', modelId, ready: true });
    return;
  }
  await unload();
  const { getLlama, LlamaChatSession } = await import('node-llama-cpp');
  if (!llamaInstance) llamaInstance = await getLlama();
  let model;
  try {
    model = await llamaInstance.loadModel({ modelPath: file, gpuLayers: settings.gpuLayers });
  } catch (e) {
    send({
      type: 'error',
      op: 'load',
      modelId,
      message:
        `This GGUF file failed to load (${e.message || e}). It may use a model architecture ` +
        "llama.cpp doesn't support yet, or the download may be corrupted — try removing and re-downloading it.",
    });
    return;
  }
  let context;
  try {
    context = await model.createContext(settings.contextSize ? { contextSize: settings.contextSize } : {});
  } catch (e) {
    await model.dispose().catch(() => {}); // model loaded fine but is now unusable — don't leak it
    send({
      type: 'error',
      op: 'load',
      modelId,
      message:
        `This model loaded, but failed to set up for generation (${e.message || e}). It may use an architecture ` +
        'feature (like multi-token-prediction) that llama.cpp only partially supports — check the model card for notes.',
    });
    return;
  }
  const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    ...(settings.systemPrompt ? { systemPrompt: settings.systemPrompt } : {}),
  });
  chat = { modelId, model, context, session, settings };
  send({ type: 'result', op: 'load', modelId, ready: true });
}

function stop({ modelId } = {}) {
  if (chat?.modelId !== modelId || !chat.abortController) {
    send({ type: 'result', op: 'stop', modelId, stopped: false });
    return;
  }
  chat.abortController.abort();
  send({ type: 'result', op: 'stop', modelId, stopped: true });
}

async function generate({ modelId, message, opts = {} }) {
  if (chat?.modelId !== modelId) {
    send({ type: 'error', op: 'send', modelId, message: 'Model not loaded. Load it first.' });
    return;
  }
  let tokenCount = 0;
  const startedAt = Date.now();
  const abortController = new AbortController();
  chat.abortController = abortController;
  let answer;
  try {
    // JSON-schema grammar mode: real node-llama-cpp feature
    // (llama.createGrammarForJsonSchema), not a new model/binary — forces
    // the model's own output to conform to opts.jsonSchema token-by-token
    // during sampling, not a post-hoc parse-and-hope.
    let grammar;
    if (opts.jsonSchema) {
      try {
        grammar = await llamaInstance.createGrammarForJsonSchema(opts.jsonSchema);
      } catch (e) {
        send({ type: 'error', op: 'send', modelId, message: `Invalid JSON schema for grammar mode: ${e.message || e}` });
        return;
      }
    }
    answer = await chat.session.prompt(message, {
      signal: abortController.signal,
      stopOnAbortSignal: true, // stop cleanly and return partial text, don't throw
      ...(typeof opts.temperature === 'number' ? { temperature: opts.temperature } : {}),
      ...(grammar ? { grammar } : {}),
      onTextChunk: (text) => send({ type: 'chunk', modelId, text }),
      onToken: (tokens) => {
        tokenCount += tokens.length;
      },
    });
  } catch (e) {
    send({ type: 'error', op: 'send', modelId, message: String(e.message || e) });
    return;
  } finally {
    if (chat?.abortController === abortController) chat.abortController = null;
  }
  const elapsedSec = (Date.now() - startedAt) / 1000;
  const tokensPerSecond = tokenCount > 0 && elapsedSec > 0 ? Math.round((tokenCount / elapsedSec) * 10) / 10 : null;
  send({ type: 'result', op: 'send', modelId, answer, tokensPerSecond, tokenCount });
}

// Phase 5: embeddings via node-llama-cpp's own createEmbeddingContext — no
// new binary/runner needed, the engine already loaded for chat supports
// this natively. Kept as separate state from `chat` since the embedding
// model may not be the same file as whatever's loaded for chat.
async function unloadEmbed() {
  if (!embedState) return;
  const { context, model } = embedState;
  embedState = null;
  await context?.dispose?.().catch(() => {});
  await model?.dispose?.().catch(() => {});
}

async function embed({ modelId, file, text }) {
  if (embedState?.modelId !== modelId) {
    await unloadEmbed();
    const { getLlama } = await import('node-llama-cpp');
    if (!llamaInstance) llamaInstance = await getLlama();
    let model;
    try {
      model = await llamaInstance.loadModel({ modelPath: file });
    } catch (e) {
      send({ type: 'error', op: 'embed', modelId, message: `This model failed to load for embeddings (${e.message || e}).` });
      return;
    }
    let context;
    try {
      context = await model.createEmbeddingContext();
    } catch (e) {
      await model.dispose().catch(() => {});
      send({ type: 'error', op: 'embed', modelId, message: `This model loaded, but isn't usable for embeddings (${e.message || e}). It may not be an embedding-capable architecture.` });
      return;
    }
    embedState = { modelId, model, context };
  }
  try {
    const embedding = await embedState.context.getEmbeddingFor(text);
    send({ type: 'result', op: 'embed', modelId, vector: embedding.vector });
  } catch (e) {
    send({ type: 'error', op: 'embed', modelId, message: String(e.message || e) });
  }
}

async function unloadRank() {
  if (!rankState) return;
  const { context, model } = rankState;
  rankState = null;
  await context?.dispose?.().catch(() => {});
  await model?.dispose?.().catch(() => {});
}

// Rerank: node-llama-cpp's own createRankingContext — same "no new binary"
// story as embed. Separate state again since a reranker (cross-encoder)
// model is architecturally different from both chat and embedding models.
async function rerank({ modelId, file, query, documents }) {
  if (rankState?.modelId !== modelId) {
    await unloadRank();
    const { getLlama } = await import('node-llama-cpp');
    if (!llamaInstance) llamaInstance = await getLlama();
    let model;
    try {
      model = await llamaInstance.loadModel({ modelPath: file });
    } catch (e) {
      send({ type: 'error', op: 'rerank', modelId, message: `This model failed to load for reranking (${e.message || e}).` });
      return;
    }
    let context;
    try {
      context = await model.createRankingContext();
    } catch (e) {
      await model.dispose().catch(() => {});
      send({ type: 'error', op: 'rerank', modelId, message: `This model loaded, but isn't usable for reranking (${e.message || e}). It may not be a reranker/cross-encoder architecture.` });
      return;
    }
    rankState = { modelId, model, context };
  }
  try {
    const ranked = await rankState.context.rankAndSort(query, documents);
    send({ type: 'result', op: 'rerank', modelId, ranked });
  } catch (e) {
    send({ type: 'error', op: 'rerank', modelId, message: String(e.message || e) });
  }
}

process.on('message', (msg) => {
  if (!msg || typeof msg !== 'object') return;
  const { op } = msg;
  if (op === 'load') return void load(msg).catch((e) => send({ type: 'error', op: 'load', modelId: msg.modelId, message: String(e.message || e) }));
  if (op === 'send') return void generate(msg).catch((e) => send({ type: 'error', op: 'send', modelId: msg.modelId, message: String(e.message || e) }));
  if (op === 'stop') return stop(msg);
  if (op === 'unload') return void unload().then(() => send({ type: 'result', op: 'unload', ok: true }));
  if (op === 'embed') return void embed(msg).catch((e) => send({ type: 'error', op: 'embed', modelId: msg.modelId, message: String(e.message || e) }));
  if (op === 'unloadEmbed') return void unloadEmbed().then(() => send({ type: 'result', op: 'unloadEmbed', ok: true }));
  if (op === 'rerank') return void rerank(msg).catch((e) => send({ type: 'error', op: 'rerank', modelId: msg.modelId, message: String(e.message || e) }));
  if (op === 'unloadRank') return void unloadRank().then(() => send({ type: 'result', op: 'unloadRank', ok: true }));
});

// The whole point of isolation is that a fatal error HERE is visible to the
// host as a reported failure, not a silent vanish it has to guess about.
process.on('uncaughtException', (e) => {
  send({ type: 'error', op: 'fatal', message: String(e?.message || e) });
  process.exit(1);
});

send({ type: 'hello', adapter: 'llama', pid: process.pid });
