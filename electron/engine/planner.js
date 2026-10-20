/**
 * Planner asks Claude's registry.canRun, then maps to RunPlan.
 * sd/whisper/piper register themselves — guessCanRun is only the no_adapter fallback.
 */
const { FAMILY_JOB } = require('./contract');
const registry = require('./registry');

// Families we will ship a runner for — no adapter yet ⇒ download_runner, not a lie.
const UPCOMING = new Set(['asr', 'diffusion', 'tts']);

function guessCanRun(ir) {
  if (!ir.standalone || !ir.format) return 'export_only';
  if (ir.family === 'llm' && ir.format === 'gguf') return 'yes';
  if (UPCOMING.has(ir.family) && (ir.format === 'gguf' || ir.family === 'tts')) return 'download_runner';
  return 'export_only';
}

function canRunHereFromRegistry(ir, hw, warn) {
  const verdict = registry.canRun(ir, hw);
  if (verdict.ok) return { canRunHere: warn ? 'export_only' : 'yes', runner: verdict.adapterId };
  if (verdict.reason === 'no_adapter') {
    const g = guessCanRun(ir);
    return { canRunHere: g === 'yes' ? 'download_runner' : g, runner: g === 'download_runner' ? ir.family : 'none' };
  }
  return { canRunHere: 'export_only', runner: verdict.adapterId || 'none' };
}

// Claude's scan (mixedplan.md §4.6): hw.backends = { cuda, vulkan, metal, rocm, cpu }.
// Prefer the fastest working binary; cpu is always present.
const BACKEND_PREF = ['cuda', 'metal', 'vulkan', 'rocm', 'cpu'];
// sd-cli is only pinned for vulkan + cpu (cuda zip is not hashed). whisper/piper are CPU.
const SD_BACKEND_PREF = ['vulkan', 'cpu'];

function pickBackend(hw, family) {
  const b = hw && hw.backends;
  if (!b) return null;
  if (family === 'asr' || family === 'tts') return 'cpu';
  const pref = family === 'diffusion' || family === 't2i' ? SD_BACKEND_PREF : BACKEND_PREF;
  for (let i = 0; i < pref.length; i++) {
    if (b[pref[i]]) return pref[i];
  }
  return 'cpu';
}

function fitLine(ir, match) {
  const fit = match?.overall || 'unknown';
  const tag = (ir.pipelineTag || ir.family || 'model').replace(/-/g, ' ');
  if (ir.family === 'diffusion') {
    return fit === 'unlikely'
      ? `This ${tag} model looks too heavy for this GPU at 512px.`
      : `This ${tag} model should fit at ~512px on this machine (${fit}).`;
  }
  if (ir.family === 'asr' || ir.family === 'tts') {
    return `This ${tag} model is usually RAM/CPU-bound — fit ${fit}.`;
  }
  if (ir.family === 'llm') {
    return match?.summary || `Chat model — ${fit} on this hardware.`;
  }
  return `This is a ${tag} model — ${fit} by file size, not a chat GGUF.`;
}

function plan(ir, hw, match, extra) {
  const warn = extra?.experimentalWarning || null;
  const job = extra?.job || ir.job || FAMILY_JOB[ir.family] || 'chat';
  const { canRunHere, runner } = canRunHereFromRegistry({ ...ir, job }, hw, warn);
  const backend = pickBackend(hw, ir.family);
  return {
    canFit: match?.overall === 'unlikely' ? 'no' : match?.overall || 'unknown',
    canRunHere,
    runner,
    device: backend && backend !== 'cpu' ? 'gpu' : 'cpu',
    backend,
    job,
    family: ir.family,
    fitLine: fitLine(ir, match),
    warnings: warn ? [warn] : [],
  };
}

module.exports = { plan, guessCanRun, fitLine, pickBackend };

// ponytail: tiny check, no test framework
if (require.main === module) {
  registry.register({
    manifest: { id: 'llama', jobs: ['chat'], accepts: [{ format: 'gguf' }], requirements: { minRamGb: 1 } },
  });
  const ir = {
    family: 'llm', format: 'gguf', standalone: true, arch: 'qwen3',
    pipelineTag: 'text-generation', job: 'chat',
  };
  const hw = {
    budgets: { canUseGpu: true, maxRamForModelGb: 8 },
    backends: { cuda: true, vulkan: true, metal: false, rocm: false, cpu: true },
  };
  const p = plan(ir, hw, { overall: 'good', summary: 'ok' }, {});
  if (p.canRunHere !== 'yes' || p.runner !== 'llama') throw new Error('llm gguf should hit llama registry');
  if (p.backend !== 'cuda' || p.device !== 'gpu') throw new Error('should pick cuda over vulkan');
  const cpuOnly = plan(ir, { budgets: { canUseGpu: false }, backends: { cuda: false, vulkan: false, metal: false, rocm: false, cpu: true } }, { overall: 'good', summary: 'ok' }, {});
  if (cpuOnly.backend !== 'cpu' || cpuOnly.device !== 'cpu') throw new Error('cpu-only box should plan cpu');
  const img = plan(
    { family: 'diffusion', format: 'gguf', standalone: true, pipelineTag: 'text-to-image', job: 't2i' },
    hw,
    { overall: 'good' },
    {}
  );
  if (img.canRunHere !== 'download_runner') throw new Error('diffusion should wait for sd.cpp');
  if (img.backend !== 'vulkan') throw new Error('sd-cli has no cuda pin — vulkan over cuda, got ' + img.backend);
  const asr = plan(
    { family: 'asr', format: 'bin', standalone: true, pipelineTag: 'automatic-speech-recognition', job: 'asr' },
    hw,
    { overall: 'good' },
    {}
  );
  if (asr.backend !== 'cpu') throw new Error('whisper/piper are CPU — asr must not advertise cuda');
  console.log('planner ok');
}
