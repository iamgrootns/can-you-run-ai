'use strict';
/**
 * Resource scheduler — Claude-owned (see Desktop/mixedplan.md §11).
 * Single in-process queue tracking estimated RAM/VRAM of loaded/in-flight
 * adapter jobs, plus a GPU exclusivity lock: one writer by default, more
 * only if the holder's own reservation declared tensor-split support.
 *
 * Queue depth of 1 (no actual waiter queue yet) is intentional for Phase 1 —
 * there's no second GPU-hungry adapter to contend with until sd.cpp lands in
 * Phase 3. reserve() just says yes/no for now; real queuing comes with it.
 */

let gpuHolder = null; // { jobId, adapterId, tensorSplit }
const loaded = new Map(); // jobId -> { adapterId, ramGb, vramGb, device }

function gpuFree() {
  return !gpuHolder;
}

/**
 * Reserve resources for a job before an adapter loads a model. Returns
 * { ok: true } to proceed, or { ok: false, reason } — reason is a sentence
 * a human can act on, not just a status code.
 */
function reserve(jobId, { adapterId, ramGb = 0, vramGb = 0, device = 'cpu', tensorSplit = false } = {}) {
  if (loaded.has(jobId)) return { ok: true }; // already reserved, idempotent
  if (device === 'gpu' && !gpuFree()) {
    if (!(gpuHolder.tensorSplit && tensorSplit)) {
      return {
        ok: false,
        reason: `GPU is in use by ${gpuHolder.adapterId}. Unload it first, or wait.`,
      };
    }
  }
  if (device === 'gpu' && gpuFree()) gpuHolder = { jobId, adapterId, tensorSplit };
  loaded.set(jobId, { adapterId, ramGb, vramGb, device });
  return { ok: true };
}

function release(jobId) {
  const entry = loaded.get(jobId) || null;
  loaded.delete(jobId);
  if (gpuHolder?.jobId === jobId) gpuHolder = null;
  return entry;
}

function status() {
  return {
    gpuHolder,
    loaded: [...loaded.entries()].map(([jobId, v]) => ({ jobId, ...v })),
  };
}

module.exports = { reserve, release, status, gpuFree };

// ponytail: tiny check, no test framework (same pattern as planner.js) —
// this is the "dummy hold VRAM job" unit test for the GPU lock, run before
// sd.cpp exists to actually contend for it.
if (require.main === module) {
  const a = reserve('chat:jobA', { adapterId: 'llama', vramGb: 3, device: 'gpu' });
  if (!a.ok) throw new Error('first GPU reserve should succeed');
  const b = reserve('t2i:jobB', { adapterId: 'sd', vramGb: 3, device: 'gpu' });
  if (b.ok) throw new Error('second GPU reserve should be blocked while first holds the lock');
  release('chat:jobA');
  const c = reserve('t2i:jobB', { adapterId: 'sd', vramGb: 3, device: 'gpu' });
  if (!c.ok) throw new Error('GPU reserve should succeed after release');
  const d = reserve('chat:jobD', { adapterId: 'llama-2', vramGb: 1, device: 'gpu', tensorSplit: true });
  if (d.ok) throw new Error('tensor-split should not help unless the CURRENT holder also declared it');
  release('t2i:jobB');
  console.log('scheduler ok');
}
