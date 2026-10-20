'use strict';
/**
 * Adapter registry — Claude-owned (see Desktop/mixedplan.md §11).
 *
 * Adapters register a manifest (jobs/accepts/devices/requirements) plus
 * whatever functions they implement. This module answers a purely
 * mechanical question: "is there a real, working adapter for this IR right
 * now, and does this hardware meet its floor?" It does NOT decide what "no"
 * means product-wise (download_runner vs export_only) — that's a family/
 * policy call planner.js (Grok's file) makes using this as one input, per
 * the handshake documented in mixedplan.md: planner.plan() asks
 * registry.canRun/estimateResources, then produces the RunPlan.
 */
const { FAMILY_JOB } = require('./contract');

const adapters = new Map(); // id -> adapter module (manifest + functions)

function register(adapterModule) {
  const m = adapterModule?.manifest;
  if (!m?.id) throw new Error('adapter manifest.id required');
  if (!Array.isArray(m.jobs) || !m.jobs.length) throw new Error(`adapter ${m.id}: manifest.jobs required`);
  adapters.set(m.id, adapterModule);
}

function list() {
  return [...adapters.values()].map((a) => a.manifest);
}

function get(id) {
  return adapters.get(id) || null;
}

function acceptsIr(manifest, ir) {
  if (!manifest.accepts?.length) return false;
  return manifest.accepts.some((a) => {
    if (a.format && a.format !== ir.format) return false;
    if (a.arch && ir.arch && a.arch !== ir.arch) return false;
    return true;
  });
}

/**
 * Every registered adapter that could plausibly accept this IR for this job,
 * in registration order. Most jobs still only ever have one candidate (chat,
 * asr, tts each have exactly one adapter) — this only matters once two
 * adapters compete for the same job, e.g. sd.cpp (single-checkpoint only)
 * and the Python diffusers adapter (everything sd.cpp declines) both
 * registering for 't2i'.
 */
function candidatesFor(ir, job) {
  const wantJob = job || ir?.job || FAMILY_JOB[ir?.family] || 'chat';
  const out = [];
  if (!ir?.standalone) return out;
  for (const mod of adapters.values()) {
    if (!mod.manifest.jobs.includes(wantJob)) continue;
    if (!acceptsIr(mod.manifest, ir)) continue;
    out.push(mod);
  }
  return out;
}

/** First candidate, or null. Kept for callers that only ever expect one adapter per job. */
function findFor(ir, job) {
  return candidatesFor(ir, job)[0] || null;
}

/**
 * Real yes/no — not the 3-way RunPlan verdict. { ok:true, adapterId } or
 * { ok:false, reason }, where reason is one of:
 *   'no_adapter'            no registered adapter accepts this IR/job at all
 *   'below_min_requirements' an adapter matched but hardware falls short
 *   'adapter_declined:<msg>' the adapter's own canRun() said no
 * When multiple adapters accept the same IR/job, each is tried in
 * registration order and the first real "yes" wins — a decline from one
 * (e.g. sd.cpp refusing a multi-file diffusers pipeline) falls through to
 * the next instead of being the final answer.
 */
function canRun(ir, hardware) {
  const candidates = candidatesFor(ir, ir?.job);
  if (!candidates.length) return { ok: false, reason: 'no_adapter' };
  let lastFail = null;
  for (const mod of candidates) {
    const req = mod.manifest.requirements || {};
    if (req.minRamGb && (hardware?.budgets?.maxRamForModelGb || 0) < req.minRamGb) {
      lastFail = { ok: false, reason: 'below_min_requirements', adapterId: mod.manifest.id };
      continue;
    }
    if (typeof mod.canRun === 'function') {
      const verdict = mod.canRun(ir, hardware);
      if (verdict !== true) {
        lastFail = { ok: false, reason: `adapter_declined:${verdict || 'unknown'}`, adapterId: mod.manifest.id };
        continue;
      }
    }
    return { ok: true, adapterId: mod.manifest.id };
  }
  return lastFail || { ok: false, reason: 'no_adapter' };
}

/**
 * Delegates to whichever adapter canRun() would actually pick, not just the
 * first format-accepting one — so the fit math matches the adapter that will
 * really run the job. Note `job` here is generation params (width/height/
 * batchCount, as sd's and pydiffusers' own estimateResources read it), not a
 * job-type string — the job type comes from `ir.job`/`ir.family` same as
 * everywhere else in this file. Don't conflate the two.
 */
function estimateResources(ir, hardware, job) {
  const verdict = canRun(ir, hardware);
  const mod = verdict.ok ? get(verdict.adapterId) : findFor(ir);
  if (!mod || typeof mod.estimateResources !== 'function') return null;
  return mod.estimateResources(ir, hardware, job);
}

module.exports = { register, list, get, findFor, candidatesFor, canRun, estimateResources };

// ponytail: tiny check, no test framework (same pattern as planner.js)
if (require.main === module) {
  register({
    manifest: {
      id: 'llama',
      jobs: ['chat'],
      accepts: [{ format: 'gguf' }],
      requirements: { minRamGb: 1 },
    },
  });
  const irLlm = { family: 'llm', format: 'gguf', standalone: true, job: 'chat' };
  const yes = canRun(irLlm, { budgets: { maxRamForModelGb: 8 } });
  if (!yes.ok) throw new Error('gguf llm with enough RAM should find the llama adapter');
  const no = canRun(irLlm, { budgets: { maxRamForModelGb: 0.1 } });
  if (no.ok) throw new Error('below min RAM should fail');
  const none = canRun({ family: 'diffusion', format: 'gguf', standalone: true, job: 't2i' }, {});
  if (none.ok || none.reason !== 'no_adapter') throw new Error('no diffusion adapter registered yet — should be no_adapter');
  console.log('registry ok');
}
