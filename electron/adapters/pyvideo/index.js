'use strict';
/**
 * Python/diffusers text-to-video adapter — same shape as adapters/pydiffusion
 * (and shares most of its reasoning), for the 'video'/'t2v' family. There is
 * no native/ggml video-generation runtime at all right now, so unlike t2i
 * (sd.cpp handles single-checkpoint models, this adapter only catches the
 * rest) this is the only registered adapter for 't2v' — every accepted
 * video model goes through here.
 *
 * Honesty note (matching this codebase's existing standard for unverified
 * paths, e.g. sd/index.js's Windows Vulkan caveat): the worker script and
 * Node orchestration follow the exact pattern proven end-to-end by
 * pydiffusion (real image generated, real NDJSON events, real file written —
 * see that adapter's dev notes), but a full text-to-video generation has NOT
 * been run live during development — real text-to-video checkpoints are
 * multi-GB and no small CI-style test fixture could be found on the Hub at
 * the time this was written. The one actual difference from the proven image
 * path (frames -> export_to_video instead of a single image.save) is small
 * and mechanical, but flagging it rather than implying the same live-tested
 * confidence as pydiffusion would be dishonest.
 */
const fs = require('fs');
const path = require('path');
const scheduler = require('../../engine/scheduler');
const pyhost = require('../../engine/pyhost');
const plat = require('../_shared/platform');

const STACK_ID = 'video';
// unpackedPath: worker.py is spawned as an external process, which can't
// read a path still inside app.asar — see platform.js's own comment.
const WORKER_SCRIPT = plat.unpackedPath(path.join(__dirname, '..', 'py', 'stacks', 'video', 'worker.py'));
const PACKAGES = ['diffusers', 'transformers', 'accelerate', 'safetensors', 'imageio', 'imageio-ffmpeg', 'torch'];
const TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu';

const manifest = {
  id: 'pyvideo',
  version: '1',
  jobs: ['t2v'],
  accepts: [{ format: 'safetensors' }],
  devices: ['gpu', 'cpu'],
  // Video is far heavier than image gen (num_frames activations at once) —
  // a CPU-only box can technically run this per the adapter contract, but
  // "technically can" and "in acceptable time" are different claims; see
  // estimateResources below, which the planner's fit math is honest about.
  requirements: { minRamGb: 6 },
  caps: { tensorSplit: false, streaming: true },
};

function canRun(ir, _hardware) {
  if (!ir || ir.family !== 'video' || !ir.standalone) return 'wrong_family';
  if (ir.format !== 'safetensors') return 'unsupported_format';
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights');
  if (weightFiles.length < 1) return 'unexpected_file_count';
  return true;
}

const PY_RUNTIME_OVERHEAD_GB = 1.5;

function estimateResources(ir, hardware, job) {
  const weightFiles = (ir.files || []).filter((f) => f.role === 'weights' || f.role === 'vae');
  const weightGb = weightFiles.reduce((s, f) => s + (f.size || 0), 0) / 1024 ** 3 || 4;
  const width = job?.width || 512;
  const height = job?.height || 512;
  const numFrames = job?.numFrames || 16;
  // Every frame holds its own latent/activation tensors live at once (unlike
  // autoregressive audio) — this is the single biggest reason video gen is
  // so much heavier than a single image at the same resolution.
  const resolutionFactor = (width * height) / (512 * 512);
  const activationGb = 1.5 * resolutionFactor * numFrames;
  const totalGb = weightGb + activationGb + PY_RUNTIME_OVERHEAD_GB;
  const device = hardware?.budgets?.canUseGpu ? 'gpu' : 'cpu';
  return {
    ram: device === 'cpu' ? totalGb : PY_RUNTIME_OVERHEAD_GB + 0.5,
    vram: device === 'gpu' ? totalGb - PY_RUNTIME_OVERHEAD_GB : 0,
    disk: weightGb,
    device,
    preset: `${numFrames}f@${width}x${height}`,
  };
}

function torchDeviceFor(hardware) {
  if (!hardware?.budgets?.canUseGpu) return 'cpu';
  if (plat.osKey() === 'mac') return 'mps';
  return hardware?.backends?.cuda ? 'cuda' : 'cpu';
}

const activeJobs = new Map();

async function generate(jobId, { modelPath, prompt, negativePrompt, numFrames, fps, steps, seed, outputPath, hwEstimate = {}, hardware }, onProgress) {
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
        { modelPath, prompt, negativePrompt, numFrames, fps, steps, seed, outputPath, device },
        (ev) => {
          if (ev.type === 'progress' || ev.type === 'log') onProgress?.(ev.text || '');
          else if (ev.type === 'result') resolve({ outputPath: ev.outputPath });
          else if (ev.type === 'error') reject(new Error(ev.error));
          else if (ev.type === 'exit' && ev.code !== 0) reject(new Error(`video worker exited ${ev.code}`));
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
