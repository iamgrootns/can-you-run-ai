'use strict';
/**
 * stable-diffusion.cpp adapter — Claude-owned. Unlike llama, sd-cli is a
 * native binary (github.com/leejet/stable-diffusion.cpp), not an npm-wrapped
 * Node addon — there's no persistent Node worker to fork() and message.
 * Every generation spawns its own one-shot sd-cli process (child_process
 * .spawn), so this module manages its own process lifecycle directly rather
 * than forcing host.js's fork/persistent-worker shape onto something that
 * isn't one. It still goes through the same scheduler.reserve/release GPU
 * lock as the llama adapter — that's the actual shared contract that
 * matters for proving Phase 3 (mixedplan.md §6): image gen can't start
 * while chat holds the GPU.
 *
 * CLI flags below are copied from running `sd-cli -h` directly against
 * the real binary (release master-820-de298c2, commit de298c2), not from
 * documentation — verified 2026-08-13. Cross-platform asset table added
 * 2026-10-12: Linux (x64, CPU + Vulkan) and macOS (arm64) builds exist
 * upstream and were verified the same way (downloaded + hashed + actually
 * executed `sd-cli -h` on Linux). Linux ROCm and the Windows CUDA/ROCm
 * builds are deliberately absent — unverified, same reasoning as before.
 */
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');
const { app } = require('electron');
const scheduler = require('../../engine/scheduler');
const plat = require('../_shared/platform');

const RELEASE_TAG = 'master-820-de298c2';
const RELEASE_BASE = `https://github.com/leejet/stable-diffusion.cpp/releases/download/${RELEASE_TAG}`;

// Verified by downloading these exact assets and computing their sha256
// myself against the real GitHub release — not copied from an unverified
// third party. cuda/rocm are deliberately absent: I have not verified those
// assets the same way, so ensureInstalled refuses them with a clear reason
// instead of guessing a hash (a wrong pin here would silently defeat the
// point of pinning at all).
const ASSETS = {
  'win-x64': {
    cpu: {
      file: 'sd-master-de298c2-bin-win-cpu-x64.zip',
      sha256: '54BC213932846C6DBE2119DC6FF71F06112D79700423224B1280729590A68187'.toLowerCase(),
    },
    vulkan: {
      file: 'sd-master-de298c2-bin-win-vulkan-x64.zip',
      sha256: 'E9D3072E090EAA0DC91970034EBCCEE6FAFD502C3226480B46BD12AC54F96889'.toLowerCase(),
      // Known issue found during Phase 3 research: this build fails to start
      // at all on this dev machine (STATUS_DLL_NOT_FOUND, 0xC0000135) even
      // though vulkan-1.dll is present system-wide — root cause not yet
      // isolated (possibly a ggml-vulkan.dll transitive dependency, possibly
      // a driver/SDK version mismatch). ensureInstalled still installs it
      // (the zip itself is verified-good), but callers should expect this
      // backend may fail at spawn time until that's debugged further.
    },
  },
  'linux-x64': {
    cpu: {
      file: 'sd-master-de298c2-bin-Linux-Ubuntu-24.04-x86_64.zip',
      sha256: '7ecc39e2485ec871d127d004d2bac4c2640eea7d0235cb6b131a8006322e0ce3',
    },
    vulkan: {
      file: 'sd-master-de298c2-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip',
      sha256: '945b697995d8c7dbde6323d04177d7c12d29091ca84796b423f9ab6c660090ad',
    },
  },
  'mac-arm64': {
    // Upstream ships exactly one macOS build (Apple Silicon only, no Intel
    // Mac build exists) — labeled 'cpu' here to match the ASSETS key shape
    // other platforms use, but not verified to confirm whether it's built
    // with Metal acceleration baked in (no Apple hardware available to test
    // against). Runs via the normal cpu device path either way.
    cpu: {
      file: 'sd-master-de298c2-bin-Darwin-macOS-26.5.2-arm64.zip',
      sha256: '22ddba91714447fa885ef97cb4040dcb043ffc57531a441cb87737f69d72b7f5',
    },
  },
};

const manifest = {
  id: 'sd',
  version: '1',
  jobs: ['t2i'],
  accepts: [{ format: 'safetensors' }, { format: 'gguf' }],
  devices: ['gpu', 'cpu'],
  requirements: { minRamGb: 2 },
  caps: { tensorSplit: false, streaming: false },
};

function runtimeDir(backend) {
  return path.join(app.getPath('userData'), 'runtimes', 'sd', plat.platformArchKey(), backend);
}

// Confirmed by extracting the real Linux/macOS archives: sd-cli sits flat at
// the archive root, same as the Windows zip — no wrapper folder on any OS.
function exeName(backend) {
  return path.join(runtimeDir(backend), `sd-cli${plat.exeSuffix()}`);
}

/**
 * Downloads + sha256-verifies + extracts the sd-cli binary for `backend` if
 * not already installed. Idempotent: a marker file records which verified
 * hash is currently extracted, so re-calling this after a successful
 * install is a fast no-op, not a re-download.
 */
async function ensureInstalled(backend) {
  const platAssets = ASSETS[plat.platformArchKey()] || {};
  const b = backend && platAssets[backend] ? backend : null;
  if (!b) {
    const available = Object.keys(platAssets);
    return {
      ok: false,
      reason: backend
        ? `No verified sd-cli build for backend "${backend}" on ${plat.platformArchKey()} — ` +
          (available.length ? `only ${available.join(', ')} ${available.length > 1 ? 'are' : 'is'} pinned right now.` : 'no build is pinned for this platform yet.')
        : 'No backend specified.',
    };
  }
  const asset = platAssets[b];
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

  const tmpArchive = path.join(app.getPath('temp'), `sd-${b}-${Date.now()}${path.extname(asset.file)}`);
  try {
    await plat.downloadFile(`${RELEASE_BASE}/${asset.file}`, tmpArchive);
    const actualHash = await plat.sha256File(tmpArchive);
    if (actualHash.toLowerCase() !== asset.sha256) {
      throw new Error(
        `sha256 mismatch for ${asset.file}: expected ${asset.sha256}, got ${actualHash}. ` +
          'Refusing to install a binary that does not match the pinned hash.'
      );
    }
    await plat.extractArchive(tmpArchive, dir);
    if (!fs.existsSync(exeName(b))) throw new Error('sd-cli not found after extraction — archive layout may have changed upstream.');
    await plat.makeExecutable(exeName(b));
    await fsp.writeFile(markerFile, asset.sha256, 'utf8');
    return { ok: true, backend: b, path: exeName(b), cached: false };
  } catch (e) {
    return { ok: false, reason: String(e.message || e) };
  } finally {
    await fsp.unlink(tmpArchive).catch(() => {});
  }
}

/**
 * Real yes/no for "can sd.cpp load this specific Hub repo" — not just
 * "is this a diffusion model." Most HF text-to-image repos are diffusers
 * pipelines (separate unet/vae/text_encoder folders + model_index.json),
 * which sd-cli's single -m/--model flag cannot load. sd-cli wants ONE merged
 * checkpoint file. The Model IR's file roles (ir.js, Grok-owned) already
 * distinguish weights/vae/tokenizer/etc per file — more than one file with
 * role "weights" means this is a split pipeline, not a single checkpoint,
 * so it's correctly "no" here rather than downloading multiple GB that
 * sd-cli then can't actually load.
 */
function canRun(ir, _hardware) {
  if (!ir || ir.family !== 'diffusion' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'safetensors' && ir.format !== 'gguf') return 'unsupported_format';
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  if (weightFiles.length !== 1) return 'multi_file_pipeline_not_single_checkpoint';
  return true;
}

/**
 * Per mixedplan.md §5's diffusion fit row: weights + VAE + resolution² +
 * batch. VAE is usually embedded in a single-checkpoint file (the case this
 * adapter actually accepts, per canRun above), so it's folded into the
 * weight size rather than estimated separately.
 */
function estimateResources(ir, hardware, job) {
  const weightFile = (ir.files || []).find((f) => f.role === 'weights');
  const weightGb = weightFile ? weightFile.size / 1024 ** 3 : 2;
  const width = job?.width || 512;
  const height = job?.height || 512;
  const batch = job?.batchCount || 1;
  // Empirical-ish overhead: latent + activation memory scales with pixel
  // count; 512x512 batch-1 on a typical SD1.5-class checkpoint costs
  // roughly another ~1.5-2GB beyond the weights themselves. Scaled by
  // resolution² relative to the 512 baseline per the plan's fit rule.
  const resolutionFactor = (width * height) / (512 * 512);
  const activationGb = 1.5 * resolutionFactor * batch;
  const totalGb = weightGb + activationGb;
  const device = hardware?.budgets?.canUseGpu ? 'gpu' : 'cpu';
  return {
    ram: device === 'cpu' ? totalGb : Math.min(totalGb, 2),
    vram: device === 'gpu' ? totalGb : 0,
    disk: weightGb,
    device,
    preset: width <= 512 && height <= 512 ? '512px' : `${width}x${height}`,
  };
}

const generatingJobs = new Map(); // jobId -> ChildProcess

/**
 * Spawns sd-cli for exactly one image, reserving/releasing the GPU lock
 * around the process lifetime (not just the load, since there's no separate
 * "loaded" state here — each job owns the GPU for its own duration). Emits
 * progress via onProgress(rawLogLine) — sd-cli doesn't print a numeric
 * per-step progress bar to stdout (confirmed: no printf on the sampling
 * path in its source, only a preview-image callback), so this is raw
 * stderr log lines, not a percentage. Honest about that limit rather than
 * fabricating a fake progress bar.
 */
async function generate(jobId, { modelPath, prompt, negativePrompt, width, height, steps, cfgScale, seed, outputPath, backend, hwEstimate = {}, initImagePath, maskImagePath, strength }, onProgress) {
  const install = await ensureInstalled(backend || 'cpu');
  if (!install.ok) throw new Error(install.reason);

  const device = hwEstimate.device === 'gpu' ? 'gpu' : 'cpu';
  const reserved = scheduler.reserve(jobId, {
    adapterId: manifest.id,
    ramGb: hwEstimate.ramGb || 0,
    vramGb: hwEstimate.vramGb || 0,
    device,
    tensorSplit: false,
  });
  if (!reserved.ok) throw new Error(reserved.reason);

  const args = [
    '-m', modelPath,
    '-p', prompt,
    '-o', outputPath,
    '-W', String(width || 512),
    '-H', String(height || 512),
    '--steps', String(steps || 20),
    '--cfg-scale', String(cfgScale ?? 7.0),
    '-s', String(seed ?? -1),
    '--vae-tiling', // reduces VRAM at minor speed cost — safe default for the "don't OOM the user's GPU" ethos
  ];
  if (negativePrompt) args.push('-n', negativePrompt);
  // img2img: -i alone. inpaint: -i + --mask together. Both real sd-cli
  // flags in the same default img_gen mode — verified from `sd-cli -h`,
  // not guessed. --mask without --init-img makes no sense, so it's only
  // sent when an init image is present.
  if (initImagePath) {
    args.push('-i', initImagePath);
    args.push('--strength', String(strength ?? 0.75));
    if (maskImagePath) args.push('--mask', maskImagePath);
  }

  try {
    await new Promise((resolve, reject) => {
      const binDir = path.dirname(install.path);
      const proc = spawn(install.path, args, {
        cwd: binDir,
        env: { ...process.env, ...plat.libraryPathEnv(binDir) },
      });
      generatingJobs.set(jobId, proc);
      let stderrTail = '';
      proc.stderr.on('data', (d) => {
        const text = d.toString();
        stderrTail = (stderrTail + text).slice(-4000); // keep the tail for error messages, not the whole log
        onProgress?.(text);
      });
      proc.on('error', reject);
      proc.on('exit', (code) => {
        generatingJobs.delete(jobId);
        if (code !== 0) return reject(new Error(`sd-cli exited ${code}: ${stderrTail.slice(-500)}`));
        if (!fs.existsSync(outputPath)) return reject(new Error('sd-cli exited 0 but the output image was not written'));
        resolve();
      });
    });
    return { outputPath };
  } finally {
    scheduler.release(jobId);
  }
}

function cancel(jobId) {
  const proc = generatingJobs.get(jobId);
  if (!proc) return { stopped: false };
  proc.kill();
  return { stopped: true };
}

module.exports = { manifest, ensureInstalled, canRun, estimateResources, generate, cancel };
