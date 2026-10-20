'use strict';
/**
 * Python/diffusers text-to-image adapter — catches what sd.cpp (adapters/sd)
 * deliberately declines. sd-cli only loads a single merged checkpoint file;
 * most text-to-image repos on the Hub are "diffusers" pipelines — separate
 * unet/vae/text_encoder folders + a model_index.json — which sd-cli's
 * -m/--model flag cannot load at all. Previously that meant "no" for the
 * majority of HF image models (export_only). This adapter is the real "yes"
 * for that majority, via electron/engine/pyhost.js's venv-per-stack manager
 * (uv auto-installs its own Python, the user never sees pip/venv/python).
 *
 * Registered in main.js AFTER sd, so sd still wins for the single-checkpoint
 * case it actually handles well (faster, far lighter than a Python+torch
 * stack) — registry.js's candidate fallback (see its 2026-10-12 change)
 * tries sd first, falls through to this adapter only when sd's canRun
 * declines with 'multi_file_pipeline_not_single_checkpoint'.
 */
const fs = require('fs');
const path = require('path');
const { app } = require('electron');
const scheduler = require('../../engine/scheduler');
const pyhost = require('../../engine/pyhost');
const plat = require('../_shared/platform');

const STACK_ID = 'diffusion';
// unpackedPath: worker.py is spawned as an external process, which can't
// read a path still inside app.asar — see platform.js's own comment.
const WORKER_SCRIPT = plat.unpackedPath(path.join(__dirname, '..', 'py', 'stacks', 'diffusion', 'worker.py'));
const PACKAGES = ['diffusers', 'transformers', 'accelerate', 'safetensors', 'pillow', 'torch'];
const TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu';

const manifest = {
  id: 'pydiffusers',
  version: '1',
  jobs: ['t2i'],
  accepts: [{ format: 'safetensors' }],
  devices: ['gpu', 'cpu'],
  // Higher floor than sd.cpp's 2GB: a Python + torch + diffusers process has
  // real baseline overhead (interpreter, CUDA/MKL init, model class graph)
  // before a single weight is loaded, which sd.cpp's native binary doesn't pay.
  requirements: { minRamGb: 4 },
  caps: { tensorSplit: false, streaming: true },
};

/**
 * The inverse of sd/index.js's canRun: accepts exactly the multi-file
 * diffusers pipelines sd.cpp refuses. Same file-role logic (ir.js tags vae/
 * tokenizer/lora files with their own role, so "weights" role count reflects
 * real separate model components — unet, text_encoder(s) — not every file
 * in the repo).
 */
function canRun(ir, _hardware) {
  if (!ir || ir.family !== 'diffusion' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'safetensors') return 'unsupported_format';
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  if (weightFiles.length <= 1) return 'single_checkpoint_sd_cpp_handles_this';
  return true;
}

// Rough baseline: a cold Python process with torch+diffusers imported and a
// pipeline object constructed, before generation-time activations. Measured
// loosely against this adapter's own dev testing, not a precise figure —
// same "labeled approximation" honesty as main.js's KV-cache estimate.
const PY_RUNTIME_OVERHEAD_GB = 1.5;

function estimateResources(ir, hardware, job) {
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights' || f.role === 'vae');
  const weightGb = weightFiles.reduce((s, f) => s + (f.size || 0), 0) / 1024 ** 3 || 2;
  const width = job?.width || 512;
  const height = job?.height || 512;
  const batch = job?.batchCount || 1;
  const resolutionFactor = (width * height) / (512 * 512);
  const activationGb = 1.5 * resolutionFactor * batch;
  const totalGb = weightGb + activationGb + PY_RUNTIME_OVERHEAD_GB;
  const device = hardware?.budgets?.canUseGpu ? 'gpu' : 'cpu';
  return {
    ram: device === 'cpu' ? totalGb : PY_RUNTIME_OVERHEAD_GB + 0.5,
    vram: device === 'gpu' ? totalGb - PY_RUNTIME_OVERHEAD_GB : 0,
    disk: weightGb,
    device,
    preset: width <= 512 && height <= 512 ? '512px' : `${width}x${height}`,
  };
}

function torchDeviceFor(hardware) {
  if (!hardware?.budgets?.canUseGpu) return 'cpu';
  if (plat.osKey() === 'mac') return 'mps';
  // Vulkan/ROCm aren't torch device backends the way CUDA/MPS are — torch's
  // own device model only really has cpu/cuda/mps. A GPU this app detected
  // via Vulkan (e.g. most non-NVIDIA cards) still falls back to cpu here;
  // that's conservative/correct, not a bug — torch cannot target Vulkan.
  return hardware?.backends?.cuda ? 'cuda' : 'cpu';
}

const activeJobs = new Map(); // jobId -> ChildProcess

/**
 * Generates one image. First call for a fresh install downloads Python +
 * torch + diffusers (can be a few hundred MB to a few GB depending on
 * device) — onProgress receives that setup output too, so the UI can show
 * real "setting up, this is one-time" text instead of looking hung.
 */
async function generate(jobId, { modelPath, prompt, negativePrompt, width, height, steps, cfgScale, seed, outputPath, hwEstimate = {}, hardware }, onProgress) {
  const device = torchDeviceFor(hardware || {});
  const extraIndexUrl = device === 'cpu' && plat.osKey() !== 'mac' ? TORCH_CPU_INDEX : null;

  const venv = await pyhost.ensureVenv(STACK_ID, { pythonVersion: '3.12', packages: PACKAGES, extraIndexUrl }, onProgress);
  if (!venv.ok) throw new Error(venv.reason);

  const reserved = scheduler.reserve(jobId, {
    adapterId: manifest.id,
    ramGb: hwEstimate.ramGb || 0,
    vramGb: hwEstimate.vramGb || 0,
    device: device === 'cpu' ? 'cpu' : 'gpu',
    tensorSplit: false,
  });
  if (!reserved.ok) throw new Error(reserved.reason);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  try {
    return await new Promise((resolve, reject) => {
      const proc = pyhost.runWorker(
        venv.pythonPath,
        WORKER_SCRIPT,
        { modelPath, prompt, negativePrompt, width, height, steps, cfgScale, seed, outputPath, device },
        (ev) => {
          if (ev.type === 'progress' || ev.type === 'log') onProgress?.(ev.text || '');
          else if (ev.type === 'result') resolve({ outputPath: ev.outputPath });
          else if (ev.type === 'error') reject(new Error(ev.error));
          else if (ev.type === 'exit' && ev.code !== 0) reject(new Error(`diffusion worker exited ${ev.code}`));
        }
      );
      activeJobs.set(jobId, proc);
    });
  } finally {
    activeJobs.delete(jobId);
    scheduler.release(jobId);
  }
}

function cancel(jobId) {
  const proc = activeJobs.get(jobId);
  if (!proc) return { stopped: false };
  proc.kill();
  return { stopped: true };
}

module.exports = { manifest, canRun, estimateResources, generate, cancel };
