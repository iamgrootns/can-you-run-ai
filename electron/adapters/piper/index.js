'use strict';
/**
 * piper TTS adapter — Claude-owned. Same one-shot-process shape as
 * adapters/sd and adapters/whisper. Uses the last standalone-binary release
 * of the ORIGINAL rhasspy/piper project (MIT, archived Oct 2025, frozen at
 * this version) rather than the actively-developed successor
 * OHF-Voice/piper1-gpl, because that successor now ships only a Python
 * wheel — no standalone binary — which conflicts with this app's own
 * "no Python in v1, native binaries only" decision (mixedplan.md, now
 * superseded for whole new model families — see electron/adapters/py/ — but
 * still the right call for TTS specifically since this binary covers it).
 * Worth knowing: this means piper support here is frozen at whatever the
 * 2023.11.14-2 release supports, not tracking upstream improvements, until
 * someone builds a real Python-sidecar story or a maintained binary fork
 * shows up.
 *
 * CLI flags below are copied from running `piper --help` against the real
 * binary — verified 2026-08-13, not from docs. Cross-platform asset table
 * added 2026-10-12: upstream actually ships Linux (x64, arm64) and macOS
 * (x64, arm64) builds already — this was the one adapter of the three that
 * didn't need upstream to add anything, just wiring. Verified by downloading
 * + hashing + actually executing `piper --help` on Linux.
 */
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron');
const scheduler = require('../../engine/scheduler');
const plat = require('../_shared/platform');

const RELEASE_TAG = '2023.11.14-2';
const RELEASE_BASE = `https://github.com/rhasspy/piper/releases/download/${RELEASE_TAG}`;

const ASSETS = {
  'win-x64': {
    cpu: { file: 'piper_windows_amd64.zip', sha256: 'F3C58906402B24F3A96D92145F58ACBA6D86C9B5DB896D207F78DC80811EFCEA'.toLowerCase() },
  },
  'linux-x64': {
    cpu: { file: 'piper_linux_x86_64.tar.gz', sha256: 'a50cb45f355b7af1f6d758c1b360717877ba0a398cc8cbe6d2a7a3a26e225992' },
  },
  'linux-arm64': {
    cpu: { file: 'piper_linux_aarch64.tar.gz', sha256: 'fea0fd2d87c54dbc7078d0f878289f404bd4d6eea6e7444a77835d1537ab88eb' },
  },
  'mac-x64': {
    cpu: { file: 'piper_macos_x64.tar.gz', sha256: 'ced85c0a3df13945b1e623b878a48fdc2854d5c485b4b67f62857cf551deaf8b' },
  },
  'mac-arm64': {
    cpu: { file: 'piper_macos_aarch64.tar.gz', sha256: '6b1eb03b3735946cb35216e063e7eebcc33a6bbf5dd96ec0217959bf1cdcb0cc' },
  },
};

const manifest = {
  id: 'piper',
  version: '1',
  jobs: ['tts'],
  accepts: [{ format: 'onnx' }],
  devices: ['cpu'],
  requirements: { minRamGb: 1 },
  caps: { tensorSplit: false, streaming: false },
};

function runtimeDir(backend) {
  return path.join(app.getPath('userData'), 'runtimes', 'piper', plat.platformArchKey(), backend);
}

// Confirmed identical nested layout (piper/piper[.exe]) across all five
// platform archives by actually extracting and listing them.
function exeName(backend) {
  return path.join(runtimeDir(backend), 'piper', `piper${plat.exeSuffix()}`);
}

function espeakDataDir(backend) {
  return path.join(runtimeDir(backend), 'piper', 'espeak-ng-data');
}

async function ensureInstalled(backend) {
  const platAssets = ASSETS[plat.platformArchKey()] || {};
  const b = backend && platAssets[backend] ? backend : 'cpu';
  const asset = platAssets[b];
  if (!asset) {
    return { ok: false, reason: `No verified piper build for ${plat.platformArchKey()} — only cpu is pinned right now.` };
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

  const tmpArchive = path.join(app.getPath('temp'), `piper-${b}-${Date.now()}${path.extname(asset.file)}`);
  try {
    await plat.downloadFile(`${RELEASE_BASE}/${asset.file}`, tmpArchive);
    const actualHash = await plat.sha256File(tmpArchive);
    if (actualHash.toLowerCase() !== asset.sha256) {
      throw new Error(`sha256 mismatch for ${asset.file}: expected ${asset.sha256}, got ${actualHash}.`);
    }
    await plat.extractArchive(tmpArchive, dir);
    if (!fs.existsSync(exeName(b))) throw new Error('piper binary not found after extraction — archive layout may have changed upstream.');
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
  if (!ir || ir.family !== 'tts' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'onnx') return 'unsupported_format';
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  if (weightFiles.length !== 1) return 'unexpected_file_count';
  return true;
}

/** Per mixedplan.md §5: small weights, latency matters more than VRAM. */
function estimateResources(ir, _hardware, _job) {
  const weightFile = (ir.files || []).find((f) => f.role === 'weights');
  const weightGb = weightFile ? weightFile.size / 1024 ** 3 : 0.1;
  return { ram: weightGb + 0.1, vram: 0, disk: weightGb, device: 'cpu', preset: 'cpu' };
}

/**
 * A piper voice is two files (model.onnx + model.onnx.json), but only the
 * .onnx counts as "the weight" in this app's IR (see main.js's getWeightMeta
 * — deliberate, so canRun's single-file check isn't confused by a normal
 * two-file voice). That means the .json config isn't fetched by the app's
 * regular per-model download flow. If it's missing next to the model on
 * disk, fetch it directly from the same HF repo path (same filename +
 * ".json") — a small, self-contained fix scoped to this adapter rather
 * than changing the shared download path for every model type.
 */
async function ensureVoiceConfig(modelPath, modelId, hfRelativePath) {
  const configPath = `${modelPath}.json`;
  if (fs.existsSync(configPath)) return configPath;
  if (!modelId || !hfRelativePath) throw new Error('Voice config (.onnx.json) is missing and no source repo info was given to fetch it.');
  const url = `https://huggingface.co/${modelId.split('/').map(encodeURIComponent).join('/')}/resolve/main/${hfRelativePath}.json`;
  await downloadFile(url, configPath);
  return configPath;
}

const runningJobs = new Map();

/**
 * Spawns piper for exactly one utterance. Text goes over stdin (piper's own
 * interface, not a CLI flag). No numeric progress — piper generates a whole
 * utterance in one pass, there's no meaningful per-step progress to report,
 * only start/done.
 */
async function speak(jobId, { modelPath, modelId, hfRelativePath, text, outputPath, hwEstimate = {} }, onProgress) {
  const install = await ensureInstalled('cpu');
  if (!install.ok) throw new Error(install.reason);
  const configPath = await ensureVoiceConfig(modelPath, modelId, hfRelativePath);

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
    '-c', configPath,
    '-f', outputPath,
    '--espeak_data', espeakDataDir('cpu'),
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
      proc.stderr.on('data', (d) => {
        const t = d.toString();
        stderrTail = (stderrTail + t).slice(-4000);
        onProgress?.(t);
      });
      proc.on('error', reject);
      proc.on('exit', (code) => {
        runningJobs.delete(jobId);
        if (code !== 0) return reject(new Error(`piper exited ${code}: ${stderrTail.slice(-500)}`));
        if (!fs.existsSync(outputPath)) return reject(new Error('piper exited 0 but the output WAV was not written'));
        resolve();
      });
      proc.stdin.write(text);
      proc.stdin.end();
    });
    return { outputPath };
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

module.exports = { manifest, ensureInstalled, canRun, estimateResources, speak, cancel };
