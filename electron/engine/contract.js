/** Shared shapes for Grok (IR/planner) + Claude (host/adapters). Do not fork. */

// 'video' and 'music' added 2026-10-12 alongside electron/adapters/py/ — the
// Python-venv adapter path (pyhost.js) that covers model families no native
// ggml-style runtime exists for. 'music' is deliberately separate from
// 'tts': both can carry HF's "text-to-audio" pipeline tag, but a MusicGen-
// style checkpoint and a piper voice need completely different adapters —
// see ir.js's familyOf for the actual split (by weight format, not just tag).
const FAMILIES = ['llm', 'diffusion', 'asr', 'tts', 'embed', 'video', 'music', 'other'];
// rerank has no dedicated family — reranker/cross-encoder HF repos are
// tagged pipeline_tag: "text-ranking" but sit in the same "compare/score
// text" bucket as embed (no ir.js family change needed). main.js overrides
// `job` to 'rerank' explicitly for text-ranking-tagged repos rather than
// widening FAMILY_JOB, since family stays 'embed' either way.
const JOBS = ['chat', 't2i', 't2v', 't2a', 'asr', 'tts', 'embed', 'rerank'];
const CAN_RUN = ['yes', 'download_runner', 'export_only'];
const PIPE_EVENTS = ['hello', 'progress', 'token', 'chunk', 'result', 'error', 'exit'];

// embed was 'chat' — a placeholder from before real embedding support
// existed (Phase 5: llama adapter's createEmbeddingContext). Fixed to route
// correctly; nothing else needed to change since planner.js reads this
// constant rather than hardcoding the mapping.
const FAMILY_JOB = { llm: 'chat', diffusion: 't2i', asr: 'asr', tts: 'tts', embed: 'embed', video: 't2v', music: 't2a', other: 'chat' };

module.exports = { FAMILIES, JOBS, CAN_RUN, PIPE_EVENTS, FAMILY_JOB };
