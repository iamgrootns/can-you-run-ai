'use strict';
/**
 * Python/transformers text-to-music adapter (MusicGen/AudioGen-family) —
 * same shape as adapters/pydiffusion, different stack: no diffusers U-Net
 * here, just AutoProcessor + a ConditionalGeneration causal LM. There is no
 * native/ggml runtime for this family to fall back from — unlike t2i (where
 * sd.cpp handles single-checkpoint models natively), every music-generation
 * checkpoint on the Hub needs the Python path, so this is the only adapter
 * registered for the 'music' family/'t2a' job, not a fallback.
 */
const fs = require('fs');
const path = require('path');
const scheduler = require('../../engine/scheduler');
const pyhost = require('../../engine/pyhost');
const plat = require('../_shared/platform');

const STACK_ID = 'music';
// unpackedPath: worker.py is spawned as an external process, which can't
// read a path still inside app.asar — see platform.js's own comment.
const WORKER_SCRIPT = plat.unpackedPath(path.join(__dirname, '..', 'py', 'stacks', 'music', 'worker.py'));
const PACKAGES = ['transformers', 'accelerate', 'scipy', 'torch'];
const TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu';

const manifest = {
  id: 'pymusic',
  version: '1',
  jobs: ['t2a'],
  // MusicGen-style repos ship as ordinary transformers checkpoints —
  // safetensors on anything recent, .bin on older/community uploads.
  accepts: [{ format: 'safetensors' }, { format: 'bin' }],
  devices: ['gpu', 'cpu'],
  requirements: { minRamGb: 4 },
  caps: { tensorSplit: false, streaming: true },
};

function canRun(ir, _hardware) {
  if (!ir || ir.family !== 'music' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'safetensors' && ir.format !== 'bin') return 'unsupported_format';
  return true;
}

const PY_RUNTIME_OVERHEAD_GB = 1.5;

function estimateResources(ir, hardware, job) {
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  const weightGb = weightFiles.reduce((s, f) => s + (f.size || 0), 0) / 1024 ** 3 || 2;
  const durationSeconds = job?.durationSeconds || 8;
  // Generation is autoregressive over audio tokens — activation memory
  // scales with sequence length (duration), not pixel count like diffusion.
  // Rough figure, not measured against every model size.
  const activationGb = 0.05 * durationSeconds;
  const totalGb = weightGb + activationGb + PY_RUNTIME_OVERHEAD_GB;
  const device = hardware?.budgets?.canUseGpu ? 'gpu' : 'cpu';
  return {
    ram: device === 'cpu' ? totalGb : PY_RUNTIME_OVERHEAD_GB + 0.5,
    vram: device === 'gpu' ? totalGb - PY_RUNTIME_OVERHEAD_GB : 0,
    disk: weightGb,
    device,
    preset: `${durationSeconds}s`,
  };
}

function torchDeviceFor(hardware) {
  if (!hardware?.budgets?.canUseGpu) return 'cpu';
  if (plat.osKey() === 'mac') return 'mps';
  return hardware?.backends?.cuda ? 'cuda' : 'cpu';
}

const activeJobs = new Map();

async function generate(jobId, { modelPath, prompt, durationSeconds, outputPath, hwEstimate = {}, hardware }, onProgress) {
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
        { modelPath, prompt, durationSeconds, outputPath, device },
        (ev) => {
          if (ev.type === 'progress' || ev.type === 'log') onProgress?.(ev.text || '');
          else if (ev.type === 'result') resolve({ outputPath: ev.outputPath });
          else if (ev.type === 'error') reject(new Error(ev.error));
          else if (ev.type === 'exit' && ev.code !== 0) reject(new Error(`music worker exited ${ev.code}`));
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
