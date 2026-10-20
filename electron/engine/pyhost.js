'use strict';
/**
 * Python runtime host — the second primitive alongside host.js (fork-based
 * Node workers) and the native one-shot spawn pattern (sd/whisper/piper).
 * Neither of those fits a Python model: there's no persistent Node worker to
 * fork(), and it isn't a single static binary — it's an interpreter plus a
 * pile of packages (torch, diffusers, ...) that vary per model family and
 * are far too big to ship in the app bundle.
 *
 * The zero-setup promise this exists for: the user never installs Python,
 * never opens a terminal, never runs pip. This module downloads a portable,
 * self-contained `uv` binary (astral-sh/uv — a single static executable,
 * sha256-pinned per platform same as the native adapters), then uses uv for
 * everything else: `uv venv --python X.Y` transparently downloads that exact
 * Python build too if it isn't present anywhere on the system — there is no
 * "install Python first" step for the user to do, uv does it. Each model
 * family gets its own isolated venv ("stack") under userData/runtimes/py/
 * so a video-gen stack's torch/diffusers pins can never conflict with an
 * image-gen stack's — see electron/adapters/py/ for the stacks themselves.
 */
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const { spawn } = require('child_process');
const { app } = require('electron');
const plat = require('../adapters/_shared/platform');

const UV_VERSION = '0.13.0';
const UV_RELEASE_BASE = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}`;

// Verified by downloading each of these exact assets and computing their
// sha256 myself against the real GitHub release — same bar as the native
// adapters' pinned binaries. `innerDir` is the wrapper folder the posix
// tarballs extract into (confirmed by listing them); the Windows zip is flat.
const UV_ASSETS = {
  'linux-x64': { file: 'uv-x86_64-unknown-linux-gnu.tar.gz', innerDir: 'uv-x86_64-unknown-linux-gnu', sha256: '1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154' },
  'linux-arm64': { file: 'uv-aarch64-unknown-linux-gnu.tar.gz', innerDir: 'uv-aarch64-unknown-linux-gnu', sha256: '3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3' },
  'mac-x64': { file: 'uv-x86_64-apple-darwin.tar.gz', innerDir: 'uv-x86_64-apple-darwin', sha256: '5f44dcbde809b632f47c36fadb241cb4d6f9af71d0c8f172b5d2026d3dde742c' },
  'mac-arm64': { file: 'uv-aarch64-apple-darwin.tar.gz', innerDir: 'uv-aarch64-apple-darwin', sha256: 'a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6' },
  'win-x64': { file: 'uv-x86_64-pc-windows-msvc.zip', innerDir: null, sha256: '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c' },
};

function uvRuntimeDir() {
  return path.join(app.getPath('userData'), 'runtimes', 'uv', plat.platformArchKey());
}

function uvExePath() {
  const asset = UV_ASSETS[plat.platformArchKey()];
  const dir = uvRuntimeDir();
  const base = asset?.innerDir ? path.join(dir, asset.innerDir) : dir;
  return path.join(base, `uv${plat.exeSuffix()}`);
}

/** Downloads + sha256-verifies + extracts the uv binary if not already installed. Idempotent, same pattern as the native adapters. */
async function ensureUv() {
  const asset = UV_ASSETS[plat.platformArchKey()];
  if (!asset) return { ok: false, reason: `No verified uv build for ${plat.platformArchKey()}.` };
  const dir = uvRuntimeDir();
  const markerFile = path.join(dir, '.installed-sha256');
  try {
    const marker = await fsp.readFile(markerFile, 'utf8');
    if (marker.trim() === asset.sha256 && fs.existsSync(uvExePath())) {
      return { ok: true, path: uvExePath(), cached: true };
    }
  } catch {
    /* not installed yet */
  }

  const tmpArchive = path.join(app.getPath('temp'), `uv-${Date.now()}${path.extname(asset.file)}`);
  try {
    await plat.downloadFile(`${UV_RELEASE_BASE}/${asset.file}`, tmpArchive);
    const actualHash = await plat.sha256File(tmpArchive);
    if (actualHash.toLowerCase() !== asset.sha256) {
      throw new Error(`sha256 mismatch for ${asset.file}: expected ${asset.sha256}, got ${actualHash}.`);
    }
    await plat.extractArchive(tmpArchive, dir);
    if (!fs.existsSync(uvExePath())) throw new Error('uv binary not found after extraction — archive layout may have changed upstream.');
    await plat.makeExecutable(uvExePath());
    await fsp.writeFile(markerFile, asset.sha256, 'utf8');
    return { ok: true, path: uvExePath(), cached: false };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  } finally {
    await fsp.unlink(tmpArchive).catch(() => {});
  }
}

function stackDir(stackId) {
  return path.join(app.getPath('userData'), 'runtimes', 'py', stackId);
}

function venvPythonPath(stackId) {
  const dir = path.join(stackDir(stackId), 'venv');
  return plat.osKey() === 'win' ? path.join(dir, 'Scripts', 'python.exe') : path.join(dir, 'bin', 'python');
}

function run(cmd, args, { onLog } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args);
    let stderrTail = '';
    proc.stdout.on('data', (d) => onLog?.(d.toString()));
    proc.stderr.on('data', (d) => {
      const t = d.toString();
      stderrTail = (stderrTail + t).slice(-4000);
      onLog?.(t);
    });
    proc.on('error', reject);
    proc.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} ${args.join(' ')} exited ${code}: ${stderrTail.slice(-800)}`))));
  });
}

/**
 * Creates (or reuses) an isolated venv for `stackId` with `packages`
 * installed, using `pythonVersion` (uv downloads that exact Python build
 * itself if nothing on the system already provides it — no user action).
 * `extraIndexUrl` lets a stack pick a hardware-specific wheel index (e.g.
 * PyTorch's own CPU vs CUDA index) without this module needing to know
 * anything about torch specifically.
 *
 * Idempotent via a marker file hashing the exact install spec: changing
 * pythonVersion/packages/extraIndexUrl triggers a real reinstall, anything
 * else is a fast no-op re-use of the existing venv.
 */
async function ensureVenv(stackId, { pythonVersion = '3.12', packages = [], extraIndexUrl = null } = {}, onLog) {
  const uv = await ensureUv();
  if (!uv.ok) return { ok: false, reason: uv.reason };

  const dir = stackDir(stackId);
  const venvDir = path.join(dir, 'venv');
  const specHash = crypto.createHash('sha256').update(JSON.stringify({ pythonVersion, packages, extraIndexUrl })).digest('hex');
  const markerFile = path.join(dir, '.installed-spec-sha256');

  try {
    const marker = await fsp.readFile(markerFile, 'utf8');
    if (marker.trim() === specHash && fs.existsSync(venvPythonPath(stackId))) {
      return { ok: true, pythonPath: venvPythonPath(stackId), cached: true };
    }
  } catch {
    /* not installed yet */
  }

  try {
    fs.mkdirSync(dir, { recursive: true });
    onLog?.(`Setting up Python ${pythonVersion} environment for "${stackId}" (first run only)...\n`);
    await run(uv.path, ['venv', '--python', pythonVersion, venvDir], { onLog });
    if (packages.length) {
      const args = ['pip', 'install', '--python', venvPythonPath(stackId), ...packages];
      if (extraIndexUrl) args.push('--extra-index-url', extraIndexUrl);
      await run(uv.path, args, { onLog });
    }
    await fsp.writeFile(markerFile, specHash, 'utf8');
    return { ok: true, pythonPath: venvPythonPath(stackId), cached: false };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  }
}

/**
 * Spawns `pythonPath scriptPath` and parses stdout as NDJSON (one JSON
 * object per line — same event shapes as contract.js's PIPE_EVENTS:
 * hello/progress/token/chunk/result/error/exit), calling onEvent per line.
 * The job payload goes over stdin as a single JSON line so argv never has to
 * deal with escaping prompts/paths — the worker script reads its one input
 * line, then streams NDJSON back on stdout for the rest of its life.
 * stderr is forwarded as raw log lines via onEvent({type:'log', text}) —
 * Python/torch/diffusers print plenty of their own diagnostic noise there
 * that isn't a protocol event but is still useful for debugging.
 */
/**
 * Python workers that call `from_pretrained(repo_id)` directly (pydiffusion,
 * pyvideo, pymusic — unlike sd/whisper/piper, which receive an
 * already-downloaded local path) do their own HF Hub download internally.
 * Left alone, huggingface_hub caches those under the user's home directory
 * (~/.cache/huggingface), outside this app's own storage entirely — breaking
 * the "everything lives under userData, works offline next day" contract
 * every other adapter follows (see README's Data section: models/ is the
 * one place weights live). Pointing HF_HOME here keeps Python's own
 * downloads inside that same managed, visible, cleanable location.
 */
function hfCacheDir() {
  return path.join(app.getPath('userData'), 'models', 'hf-cache');
}

function runWorker(pythonPath, scriptPath, payload, onEvent) {
  const proc = spawn(pythonPath, [scriptPath], {
    env: { ...process.env, PYTHONUNBUFFERED: '1', HF_HOME: hfCacheDir() },
  });
  let buf = '';
  proc.stdout.on('data', (d) => {
    buf += d.toString();
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed));
      } catch {
        onEvent({ type: 'log', text: trimmed });
      }
    }
  });
  proc.stderr.on('data', (d) => onEvent({ type: 'log', text: d.toString() }));
  proc.on('error', (e) => onEvent({ type: 'error', error: String(e.message || e) }));
  proc.on('exit', (code) => onEvent({ type: 'exit', code }));
  proc.stdin.write(`${JSON.stringify(payload)}\n`);
  return proc;
}

module.exports = { ensureUv, ensureVenv, venvPythonPath, runWorker, stackDir, hfCacheDir };
