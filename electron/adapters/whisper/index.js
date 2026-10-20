'use strict';
/**
 * whisper.cpp adapter — Claude-owned. Same shape as adapters/sd: a native
 * binary (whisper-cli[.exe] from github.com/ggml-org/whisper.cpp), not an
 * npm-wrapped Node addon, so it manages its own one-shot child_process.spawn
 * per job rather than going through host.js's fork-based worker model.
 *
 * CLI flags below are copied from running `whisper-cli -h` against the real
 * binary (release v1.9.2) — verified 2026-08-13, not from docs. Cross-platform
 * asset table added 2026-10-12: Linux x64/arm64 builds exist upstream and
 * were verified the same way (downloaded + hashed + actually executed on
 * Linux — including that LD_LIBRARY_PATH is required, see platform.js).
 * macOS has no plain CLI binary release upstream (only an xcframework meant
 * for embedding in Apple apps), so it's deliberately absent here rather than
 * guessed — ensureInstalled returns a clear reason instead of a wrong path.
 */
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron');
const scheduler = require('../../engine/scheduler');
const plat = require('../_shared/platform');

const RELEASE_TAG = 'v1.9.2';
const RELEASE_BASE = `https://github.com/ggml-org/whisper.cpp/releases/download/${RELEASE_TAG}`;

// Verified by downloading each of these exact assets and computing their
// sha256 myself against the real GitHub release (not copied from a third
// party). cublas (Windows CUDA) variants exist upstream but aren't verified
// the same way, so they're deliberately absent — ensureInstalled refuses an
// unpinned backend instead of guessing a hash.
const ASSETS = {
  'win-x64': {
    cpu: { file: 'whisper-bin-x64.zip', sha256: '49DCC16DE826F20BD53D44F947A1AE49DFA81F86CAD67A64D80820CB192D674A'.toLowerCase() },
  },
  'linux-x64': {
    cpu: { file: 'whisper-bin-ubuntu-x64.tar.gz', sha256: '46811a3ecf584307480a220b9ef5ff81b7b22dc41577cbc274ce3afc61f753b1' },
  },
  'linux-arm64': {
    cpu: { file: 'whisper-bin-ubuntu-arm64.tar.gz', sha256: '7e26fa6a36d9174d5c0bf033ccbc026c3b5e569e2ee787058241346ef5392719' },
  },
};

const manifest = {
  id: 'whisper',
  version: '1',
  jobs: ['asr'],
  accepts: [{ format: 'bin' }, { format: 'gguf', arch: 'whisper' }],
  devices: ['cpu'],
  requirements: { minRamGb: 1 },
  caps: { tensorSplit: false, streaming: false },
};

function runtimeDir(backend) {
  return path.join(app.getPath('userData'), 'runtimes', 'whisper', plat.platformArchKey(), backend);
}

// Windows build nests the binary under Release/; Linux builds extract the
// binary directly inside a whisper-bin-ubuntu-{arch}/ wrapper folder —
// confirmed by actually extracting and listing both archives, not assumed.
function exeName(backend) {
  const dir = runtimeDir(backend);
  if (plat.osKey() === 'win') return path.join(dir, 'Release', `whisper-cli${plat.exeSuffix()}`);
  return path.join(dir, `whisper-bin-ubuntu-${plat.archKey()}`, 'whisper-cli');
}

async function ensureInstalled(backend) {
  const b = backend && ASSETS[plat.platformArchKey()]?.[backend] ? backend : 'cpu';
  const asset = ASSETS[plat.platformArchKey()]?.[b];
  if (!asset) {
    return {
      ok: false,
      reason: `No verified whisper-cli build for ${plat.platformArchKey()} — ` +
        (plat.osKey() === 'mac'
          ? 'whisper.cpp only ships an xcframework for macOS (for embedding in Apple apps), not a standalone CLI binary.'
          : 'only cpu is pinned right now.'),
    };
  }
  const dir = runtimeDir(b);
  const markerFile = path.join(dir, '.installed-sha256');
  try {
    const marker = await fsp.readFile(markerFile, 'utf8');
    if (marker.trim() === asset.sha256 && fs.existsSync(exeName(b))) {
      return { ok: true, backend: b, path: exeName(b), cached: true };
    }
  } catch {
    /* not installed yet */
  }

  const tmpArchive = path.join(app.getPath('temp'), `whisper-${b}-${Date.now()}${path.extname(asset.file)}`);
  try {
    await plat.downloadFile(`${RELEASE_BASE}/${asset.file}`, tmpArchive);
    const actualHash = await plat.sha256File(tmpArchive);
    if (actualHash.toLowerCase() !== asset.sha256) {
      throw new Error(`sha256 mismatch for ${asset.file}: expected ${asset.sha256}, got ${actualHash}.`);
    }
    await plat.extractArchive(tmpArchive, dir);
    if (!fs.existsSync(exeName(b))) throw new Error('whisper-cli not found after extraction — archive layout may have changed upstream.');
    await plat.makeExecutable(exeName(b));
    await fsp.writeFile(markerFile, asset.sha256, 'utf8');
    return { ok: true, backend: b, path: exeName(b), cached: false };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  } finally {
    await fsp.unlink(tmpArchive).catch(() => {});
  }
}

function canRun(ir, _hardware) {
  if (!ir || ir.family !== 'asr' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'bin' && ir.format !== 'gguf') return 'unsupported_format';
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  if (weightFiles.length !== 1) return 'unexpected_file_count';
  return true;
}

/** Per mixedplan.md §5: weights + audio buffer, scaled by realtime factor — small and RAM-bound. */
function estimateResources(ir, hardware, _job) {
  const weightFile = (ir.files || []).find((f) => f.role === 'weights');
  const weightGb = weightFile ? weightFile.size / 1024 ** 3 : 0.5;
  const audioBufferGb = 0.2; // decoded PCM buffer for a typical clip, generous headroom
  return {
    ram: weightGb + audioBufferGb,
    vram: 0, // CPU-only backend pinned for now
    disk: weightGb,
    device: 'cpu',
    preset: 'cpu',
  };
}

const runningJobs = new Map();

/**
 * Spawns whisper-cli for exactly one transcription. Emits progress via
 * onProgress(rawLogLine) — same honesty note as adapters/sd: no numeric
 * percentage without `-pp` producing parseable output, so this forwards raw
 * stderr/stdout lines rather than fabricating a progress bar.
 */
async function transcribe(jobId, { modelPath, audioPath, language, outputBase, hwEstimate = {} }, onProgress) {
  const install = await ensureInstalled('cpu');
  if (!install.ok) throw new Error(install.reason);

  const reserved = scheduler.reserve(jobId, {
    adapterId: manifest.id,
    ramGb: hwEstimate.ramGb || 0,
    vramGb: 0,
    device: 'cpu',
    tensorSplit: false,
  });
  if (!reserved.ok) throw new Error(reserved.reason);

  const args = [
    '-m', modelPath,
    '-f', audioPath,
    '-l', language || 'auto',
    '-otxt',
    '-of', outputBase,
    '-np', // no prints other than results — cleaner to parse, still get stderr for real errors
  ];

  try {
    await new Promise((resolve, reject) => {
      const binDir = path.dirname(install.path);
      const proc = spawn(install.path, args, {
        cwd: binDir,
        env: { ...process.env, ...plat.libraryPathEnv(binDir) },
      });
      runningJobs.set(jobId, proc);
      let stderrTail = '';
      proc.stdout.on('data', (d) => onProgress?.(d.toString()));
      proc.stderr.on('data', (d) => {
        const text = d.toString();
        stderrTail = (stderrTail + text).slice(-4000);
        onProgress?.(text);
      });
      proc.on('error', reject);
      proc.on('exit', (code) => {
        runningJobs.delete(jobId);
        if (code !== 0) return reject(new Error(`whisper-cli exited ${code}: ${stderrTail.slice(-500)}`));
        resolve();
      });
    });
    const txtPath = `${outputBase}.txt`;
    if (!fs.existsSync(txtPath)) throw new Error('whisper-cli exited 0 but the transcript file was not written');
    const text = (await fsp.readFile(txtPath, 'utf8')).trim();
    return { text, txtPath };
  } finally {
    scheduler.release(jobId);
  }
}

function cancel(jobId) {
  const proc = runningJobs.get(jobId);
  if (!proc) return { stopped: false };
  proc.kill();
  return { stopped: true };
}

module.exports = { manifest, ensureInstalled, canRun, estimateResources, transcribe, cancel };
