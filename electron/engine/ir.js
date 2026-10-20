const { FAMILY_JOB } = require('./contract');

const PIPELINE_FAMILY = {
  'text-generation': 'llm',
  'text2text-generation': 'llm',
  'image-text-to-text': 'llm',
  'any-to-any': 'llm',
  'text-to-image': 'diffusion',
  'image-to-image': 'diffusion',
  'unconditional-image-generation': 'diffusion',
  'automatic-speech-recognition': 'asr',
  'audio-to-audio': 'asr',
  'text-to-speech': 'tts',
  // 'text-to-audio' is NOT handled here directly — both piper voices and
  // MusicGen-style music generators carry this same HF pipeline tag, but
  // need completely different adapters. See familyOf()'s format-based split
  // below rather than a flat tag->family entry.
  'text-to-video': 'video',
  'image-to-video': 'video',
  'feature-extraction': 'embed',
  'sentence-similarity': 'embed',
};

const JUNK_RE = /imatrix|control[-_]?vector/i;
const MMPROJ_RE = /mmproj/i;
const LORA_RE = /(^|[/\-_])lora([-_.]|$)/i;
const VAE_RE = /vae/i;
const TOK_RE = /tokenizer/i;

function roleOf(p) {
  const n = String(p || '');
  if (JUNK_RE.test(n)) return 'junk';
  if (MMPROJ_RE.test(n)) return 'mmproj';
  if (LORA_RE.test(n)) return 'lora';
  if (VAE_RE.test(n)) return 'vae';
  if (TOK_RE.test(n)) return 'tokenizer';
  return 'weights';
}

// 'text-to-audio' is genuinely ambiguous on the Hub: piper-style voices and
// MusicGen/AudioGen/Stable-Audio-style music generators both carry it.
// Format is the real signal — piper voices are always a single .onnx file
// (adapters/piper's own canRun requires exactly that); anything else tagged
// text-to-audio is a transformers/diffusers checkpoint, i.e. a music model.
function familyOf(pipelineTag, arch, format) {
  if (arch === 'whisper') return 'asr';
  if (arch === 'clip') return 'embed';
  if (pipelineTag === 'text-to-audio') return format === 'onnx' ? 'tts' : 'music';
  return PIPELINE_FAMILY[pipelineTag] || 'other';
}

/** Build Model IR from existing weightMeta (+ optional GGUF realMeta). */
function fromWeightMeta(wm, realMeta) {
  const files = (wm.weightFiles || []).map((f) => ({
    path: f.path,
    size: f.sizeBytes || 0,
    role: roleOf(f.path),
  }));
  const hasWeights = files.some((f) => f.role === 'weights');
  const onlyCompanion = files.length > 0 && files.every((f) => f.role !== 'weights');
  const arch = realMeta && realMeta.ok ? realMeta.architecture || null : null;
  const family = familyOf(wm.pipelineTag, arch, wm.format);
  return {
    id: wm.modelId,
    pipelineTag: wm.pipelineTag || null,
    family,
    format: wm.format || null,
    files,
    arch,
    standalone: hasWeights && !onlyCompanion,
    paramsBillions: wm.parameters?.exactBillions ?? wm.parameters?.billions ?? null,
    job: FAMILY_JOB[family] || 'chat',
  };
}

module.exports = { fromWeightMeta, roleOf, familyOf };
