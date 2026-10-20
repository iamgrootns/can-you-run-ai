const { app, BrowserWindow, ipcMain, shell, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const fsp = require('fs/promises');
const https = require('https');
const http = require('http');
const si = require('systeminformation');
const { gguf } = require('@huggingface/gguf');
const { fromWeightMeta } = require('./engine/ir');
const { plan, pickBackend } = require('./engine/planner');
const registry = require('./engine/registry');
const llamaAdapter = require('./adapters/llama');
const sdAdapter = require('./adapters/sd');
const whisperAdapter = require('./adapters/whisper');
const piperAdapter = require('./adapters/piper');
const pydiffusionAdapter = require('./adapters/pydiffusion');
const pyvideoAdapter = require('./adapters/pyvideo');
const pymusicAdapter = require('./adapters/pymusic');
registry.register(llamaAdapter);
registry.register(sdAdapter);
registry.register(whisperAdapter);
registry.register(piperAdapter);
// Registered after sd: registry.canRun tries candidates in registration
// order, so sd.cpp (fast, native, single-checkpoint only) is tried first and
// this Python/diffusers adapter only catches what sd.cpp declines — the
// multi-file pipelines that are most of the Hub's text-to-image repos.
registry.register(pydiffusionAdapter);
// video ('t2v') and music ('t2a') have no native runtime to prefer — these
// Python adapters are the only registered candidate for their job, not a
// fallback the way pydiffusion is for t2i.
registry.register(pyvideoAdapter);
registry.register(pymusicAdapter);

const HF = 'https://huggingface.co';
const HF_API = `${HF}/api`;
const OVERHEAD = 1.2;
const MEM = { fp32: 4, fp16: 2, int8: 1, int4: 0.5 };

let win = null;
let paths = {};

// ─── paths / disk ────────────────────────────────────────────────────────────

function initPaths() {
  const root = app.getPath('userData');
  // Named "store", not "cache" — a folder literally called "cache" sitting inside
  // Electron's userData risks being swept by Chromium's own internal cache-eviction
  // conventions for that directory name. Our data isn't disposable the way an HTTP
  // cache is, so it needs a name Chromium has no opinion about.
  paths = {
    root,
    cache: path.join(root, 'store'),
    meta: path.join(root, 'store', 'meta'),
    lists: path.join(root, 'store', 'lists'),
    models: path.join(root, 'models'),
    generated: path.join(root, 'generated'),
    tags: path.join(root, 'store', 'tags.json'),
  };
  for (const p of [paths.cache, paths.meta, paths.lists, paths.models, paths.generated]) {
    fs.mkdirSync(p, { recursive: true });
  }
}

function safeName(id) {
  return String(id).replace(/[^\w.-]+/g, '__').slice(0, 180);
}

async function readJson(file) {
  try {
    return JSON.parse(await fsp.readFile(file, 'utf8'));
  } catch {
    return null;
  }
}

async function writeJson(file, data) {
  await fsp.mkdir(path.dirname(file), { recursive: true });
  await fsp.writeFile(file, JSON.stringify(data), 'utf8');
}

function encodeRepo(id) {
  return String(id).split('/').map(encodeURIComponent).join('/');
}

function round(n, d = 2) {
  if (n == null || Number.isNaN(+n)) return n;
  const f = 10 ** d;
  return Math.round(+n * f) / f;
}

// ─── network ─────────────────────────────────────────────────────────────────

function online() {
  return new Promise((resolve) => {
    const req = https.get(`${HF_API}/models?limit=1`, { timeout: 4000 }, (res) => {
      res.resume();
      resolve(res.statusCode >= 200 && res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

// HF occasionally blips (429 under load, momentary 5xx) — a single retry a
// couple seconds later resolves most of these instead of dead-ending the
// user with an error dropdown for something that fixes itself in moments.
const TRANSIENT_RETRY_DELAYS_MS = [800, 2500];
function isTransientHttpError(e) {
  const s = String(e?.message || e);
  return /HTTP 429|HTTP 5\d\d|ENOTFOUND|ECONNRESET|ETIMEDOUT|fetch failed/i.test(s);
}

async function fetchJson(url) {
  let lastErr;
  for (let attempt = 0; attempt <= TRANSIENT_RETRY_DELAYS_MS.length; attempt++) {
    try {
      const res = await fetch(url, {
        headers: {
          Accept: 'application/json',
          'User-Agent': 'CanYouRunAI/1.0',
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      lastErr = e;
      if (attempt >= TRANSIENT_RETRY_DELAYS_MS.length || !isTransientHttpError(e)) break;
      await new Promise((r) => setTimeout(r, TRANSIENT_RETRY_DELAYS_MS[attempt]));
    }
  }
  throw lastErr;
}

async function fetchText(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'CanYouRunAI/1.0' } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

// ─── hardware ────────────────────────────────────────────────────────────────

// Real backend availability, not a guess from GPU vendor strings — asks
// node-llama-cpp which GPU types have working libraries/drivers AND a
// prebuilt binary for this exact platform+arch right now. This is a fast
// capability probe, not model loading, so it's fine to run in the main
// process — it never touches actual inference state (that stays in the
// out-of-process llama worker, per mixedplan.md's isolation rule).
async function detectBackends() {
  try {
    const { getLlamaGpuTypes } = await import('node-llama-cpp');
    const supported = new Set(await getLlamaGpuTypes('supported'));
    return {
      cuda: supported.has('cuda'),
      vulkan: supported.has('vulkan'),
      metal: supported.has('metal'),
      // node-llama-cpp has no distinct ROCm build — AMD GPUs get routed
      // through its Vulkan backend instead, so this is always false rather
      // than a real probe. Listed anyway so the shape matches what the
      // planner/UI side expects (mixedplan.md §4.6).
      rocm: false,
      cpu: true,
    };
  } catch {
    // Backend probing failing (e.g. no native binary for this arch at all)
    // shouldn't take hardware detection down with it — CPU always works.
    return { cuda: false, vulkan: false, metal: false, rocm: false, cpu: true };
  }
}

async function detectHardware() {
  const [cpu, mem, graphics, osInfo, disk, system, backends] = await Promise.all([
    si.cpu(),
    si.mem(),
    si.graphics(),
    si.osInfo(),
    si.fsSize(),
    si.system(),
    detectBackends(),
  ]);

  const controllers = (graphics.controllers || []).map((g) => ({
    model: g.model || g.name || 'Unknown GPU',
    vendor: g.vendor || '',
    vramMb: typeof g.vram === 'number' && g.vram > 0 ? g.vram : null,
    vramDynamic: !!g.vramDynamic,
  }));

  let primary = null;
  for (const g of controllers) {
    if (!primary || (g.vramMb || 0) > (primary.vramMb || 0)) primary = g;
  }

  // llama.cpp can tensor-split a single model across multiple discrete GPUs, so a
  // 2x-GPU box can run something neither card fits alone. Dynamic-VRAM controllers
  // (integrated graphics sharing system RAM) don't count as separate pooled memory —
  // that's the unified-memory path below, not additive with a discrete card.
  const discreteGpus = controllers.filter((g) => g.vramMb != null && !g.vramDynamic);
  const gpuCount = discreteGpus.length;
  const totalRamGb = mem.total / 1024 ** 3;
  const freeRamGb = mem.available / 1024 ** 3;
  const totalVramGb = discreteGpus.length
    ? discreteGpus.reduce((s, g) => s + g.vramMb, 0) / 1024
    : primary?.vramMb != null
      ? primary.vramMb / 1024
      : null;
  const unified =
    /apple/i.test(cpu.manufacturer || '') ||
    (process.platform === 'darwin' && /arm|apple/i.test(cpu.brand || '')) ||
    (!!primary?.vramDynamic && (totalVramGb == null || totalVramGb < 1));

  const disks = (disk || []).map((d) => ({
    mount: d.mount,
    availableGb: d.available / 1024 ** 3,
    sizeGb: d.size / 1024 ** 3,
  }));
  const primaryDisk =
    disks.find((d) => /^[C]:\\?$/.test(d.mount) || d.mount === '/') ||
    disks.sort((a, b) => b.availableGb - a.availableGb)[0] ||
    null;

  const reserve = Math.min(4, Math.max(1.5, totalRamGb * 0.15));
  const maxRam = Math.max(0, totalRamGb - reserve);
  let maxVram = 0;
  if (unified) maxVram = maxRam;
  else if (totalVramGb > 0) maxVram = totalVramGb * 0.9;

  const hw = {
    detectedAt: new Date().toISOString(),
    platform: process.platform,
    cpu: {
      brand: cpu.brand,
      cores: cpu.cores,
      physicalCores: cpu.physicalCores,
      speedGHz: cpu.speed,
    },
    memory: {
      totalGb: round(totalRamGb, 2),
      freeGb: round(freeRamGb, 2),
    },
    gpu: {
      controllers,
      primary,
      gpuCount,
      totalVramGb: totalVramGb != null ? round(totalVramGb, 2) : null,
      unifiedMemory: unified,
    },
    os: {
      platform: osInfo.platform,
      distro: osInfo.distro,
      release: osInfo.release,
      arch: osInfo.arch,
    },
    disk: {
      primary: primaryDisk
        ? {
            mount: primaryDisk.mount,
            availableGb: round(primaryDisk.availableGb, 2),
            sizeGb: round(primaryDisk.sizeGb, 2),
          }
        : null,
    },
    system: { manufacturer: system.manufacturer, model: system.model },
    backends,
    budgets: {
      maxRamForModelGb: round(maxRam, 2),
      maxVramForModelGb: round(maxVram, 2),
      freeDiskGb: round(primaryDisk?.availableGb ?? 0, 2),
      canUseGpu: maxVram > 0.5 || unified,
      unifiedMemory: unified,
      // Headline number: what a typical Q4 GGUF quant tops out at on this machine.
      approxMaxParamsB: round(Math.max(maxRam, maxVram) / (MEM.int4 * OVERHEAD), 1),
    },
  };

  await writeJson(path.join(paths.cache, 'hardware.json'), hw);
  return hw;
}

// ─── HF + offline cache ──────────────────────────────────────────────────────

async function getTags() {
  const cached = await readJson(paths.tags);
  if (await online()) {
    try {
      const data = await fetchJson(`${HF_API}/models-tags-by-type`);
      const tags = (data.pipeline_tag || []).map((t) => ({
        id: t.id,
        label: t.label || t.id,
        subType: t.subType || 'other',
      }));
      const bySubType = {};
      for (const t of tags) {
        (bySubType[t.subType] ||= []).push(t);
      }
      const libraries = (data.library || []).map((l) => ({
        id: l.id,
        label: l.label || l.id,
      }));
      const out = {
        fetchedAt: new Date().toISOString(),
        offline: false,
        tags,
        bySubType,
        libraries,
      };
      await writeJson(paths.tags, out);
      return out;
    } catch (e) {
      if (cached) return { ...cached, offline: true, stale: true, error: String(e.message) };
      throw e;
    }
  }
  if (cached) return { ...cached, offline: true };
  throw new Error('Offline and no cached categories. Connect once to sync.');
}

async function listModels(opts = {}) {
  const {
    pipelineTag = null,
    search = '',
    sort = 'downloads',
    limit = 20,
    page = 0,
    library = null,
  } = opts;

  const key = safeName(
    JSON.stringify({ pipelineTag, search, sort, limit, page, library })
  );
  const file = path.join(paths.lists, `${key}.json`);
  const cached = await readJson(file);

  if (await online()) {
    try {
      const q = new URLSearchParams({
        limit: String(Math.min(50, limit)),
        skip: String(page * limit),
        sort,
        direction: '-1',
        full: 'true',
      });
      if (pipelineTag) q.set('pipeline_tag', pipelineTag);
      if (search) q.set('search', search);
      if (library) q.set('library', library);

      const raw = await fetchJson(`${HF_API}/models?${q}`);
      const models = (Array.isArray(raw) ? raw : []).map((m) => ({
        id: m.id || m.modelId,
        pipelineTag: m.pipeline_tag || null,
        libraryName: m.library_name || null,
        downloads: m.downloads ?? 0,
        likes: m.likes ?? 0,
        tags: m.tags || [],
        gated: !!m.gated,
      }));
      const out = {
        fetchedAt: new Date().toISOString(),
        offline: false,
        query: opts,
        models,
      };
      await writeJson(file, out);
      return out;
    } catch (e) {
      if (cached) return { ...cached, offline: true, stale: true };
      throw e;
    }
  }
  if (cached) return { ...cached, offline: true };

  // Offline fallback: any previously cached list so the app still works
  try {
    const files = await fsp.readdir(paths.lists);
    for (const f of files) {
      if (!f.endsWith('.json')) continue;
      const alt = await readJson(path.join(paths.lists, f));
      if (alt?.models?.length) {
        return {
          ...alt,
          offline: true,
          stale: true,
          note: 'Showing last cached list (query mismatch while offline).',
        };
      }
    }
  } catch {
    /* ignore */
  }
  throw new Error('Offline and no cached results. Connect once to sync the Hub.');
}

// llama.cpp splits large GGUF models across files like "name-00001-of-00003.gguf".
// Those shards are ONE model, not independent variants — grouping them wrong would
// let a huge sharded model masquerade as several small ones (whichever shard is
// smallest looks like a tiny standalone model, and only a fraction of it would
// ever get downloaded).
function groupGgufShards(weightFiles) {
  const SHARD_RE = /^(.*)-(\d{5})-of-\d{5}\.gguf$/i;
  const groups = new Map();
  for (const f of weightFiles) {
    const dir = path.dirname(f.path);
    const base = path.basename(f.path);
    const m = base.match(SHARD_RE);
    const key = m ? `${dir}/${m[1]}` : f.path;
    const idx = m ? parseInt(m[2], 10) : 0;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({ path: f.path, size: f.size, idx });
  }
  return Array.from(groups.values()).map((parts) => {
    parts.sort((a, b) => a.idx - b.idx);
    return {
      paths: parts.map((p) => p.path),
      name: path.basename(parts[0].path),
      sizeBytes: parts.reduce((s, p) => s + (p.size || 0), 0),
      shardCount: parts.length,
    };
  });
}

const README_SNIPPET_CHARS = 1500;

/**
 * Best-effort README fetch for "why won't this load" moments — the model card
 * often has the actual setup notes for anything that needs non-standard tooling.
 */
async function getReadme(modelId) {
  const file = path.join(paths.meta, `${safeName(modelId)}.readme.json`);
  const cached = await readJson(file);
  if (!(await online())) {
    if (cached) return { ...cached, offline: true };
    return { snippet: null, cardUrl: `${HF}/${modelId}`, offline: true, unavailable: true };
  }
  try {
    const repo = encodeRepo(modelId);
    let text = await fetchText(`${HF}/${repo}/raw/main/README.md`);
    // Strip the YAML frontmatter (license/tags block) HF prepends — not useful here.
    text = text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, '').trim();
    // Many model cards use raw HTML for fancy badges/headers instead of plain
    // markdown — without stripping it, the "snippet" is unreadable tag soup.
    text = text
      .replace(/<[^>]+>/g, ' ')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#0?39;/g, "'")
      .replace(/[ \t]{2,}/g, ' ')
      .split('\n')
      .map((line) => line.trim())
      .join('\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    const snippet = text.slice(0, README_SNIPPET_CHARS);
    const result = { snippet, truncated: text.length > README_SNIPPET_CHARS, cardUrl: `${HF}/${modelId}`, offline: false };
    await writeJson(file, result);
    return result;
  } catch (e) {
    if (cached) return { ...cached, offline: true, stale: true };
    return { snippet: null, cardUrl: `${HF}/${modelId}`, offline: false, unavailable: true, error: String(e.message || e) };
  }
}

async function getWeightMeta(modelId) {
  const file = path.join(paths.meta, `${safeName(modelId)}.json`);
  const cached = await readJson(file);
  const isOn = await online();

  if (!isOn) {
    if (cached?.weightMeta) return { ...cached.weightMeta, offline: true };
    throw new Error('Offline and this model is not cached. Open it once online.');
  }

  try {
    const repo = encodeRepo(modelId);
    let tree = [];
    try {
      tree = await fetchJson(`${HF_API}/models/${repo}/tree/main?recursive=true`);
    } catch {
      tree = await fetchJson(`${HF_API}/models/${repo}/tree/main`).catch(() => []);
    }
    if (!Array.isArray(tree)) tree = [];

    const files = tree
      .filter((f) => f?.type === 'file')
      .map((f) => ({
        path: f.path,
        size: f.lfs?.size ?? f.size ?? 0,
      }));

    const st = files.filter((f) => /\.safetensors$/i.test(f.path));
    // Several .gguf-suffixed file types aren't standalone loadable chat models —
    // they're companion/auxiliary files meant to be paired with (or used to build)
    // a real model, and picking one as "the model" fails the same way the imatrix
    // bug did: small, complete, and not actually loadable on its own.
    //   imatrix          — calibration data used to produce quantized GGUFs
    //   mmproj           — the vision projector half of a multimodal model, needs
    //                      a matching text model loaded alongside it
    //   lora / control vector — delta weights applied on top of a base model
    const NON_MODEL_GGUF = /imatrix|mmproj|control[-_]?vector|(^|[/\-_])lora([-_.]|$)/i;
    const gg = files.filter((f) => /\.gguf$/i.test(f.path) && !NON_MODEL_GGUF.test(f.path));
    const bin = files.filter(
      (f) =>
        /(pytorch_model|model|tf_model).*\.(bin|h5|msgpack)$/i.test(f.path) ||
        /model-\d+-of-\d+\.bin$/i.test(f.path) ||
        // whisper.cpp's classic ggml binary format (pre-dates GGUF, a different
        // container the @huggingface/gguf parser can't read) — near-universal
        // naming convention across HF-hosted whisper.cpp models (ggml-base.bin,
        // ggml-large-v3.bin, ...). Narrow prefix match, same spirit as the
        // "model" substring requirement above: specific enough not to catch
        // unrelated .bin blobs.
        /^(.*\/)?ggml[-_].*\.bin$/i.test(f.path)
    );
    // ONNX: piper TTS voices (model.onnx + model.onnx.json config). The json
    // companion deliberately isn't matched here — only the .onnx file itself
    // counts as "the weight" for fit/canRun purposes; the piper adapter
    // fetches the config companion itself when needed.
    const onnx = files.filter((f) => /\.onnx$/i.test(f.path));

    let format = null;
    let weightFiles = [];
    if (st.length) {
      format = 'safetensors';
      weightFiles = st;
    } else if (gg.length) {
      format = 'gguf';
      weightFiles = gg;
    } else if (bin.length) {
      format = 'bin';
      weightFiles = bin;
    } else if (onnx.length) {
      format = 'onnx';
      weightFiles = onnx;
    }

    const totalWeightBytes = weightFiles.reduce((s, f) => s + (f.size || 0), 0);
    let modelInfo = null;
    try {
      modelInfo = await fetchJson(`${HF_API}/models/${repo}`);
    } catch {
      /* ignore */
    }

    const tags = modelInfo?.tags || [];
    const sources = [modelId, ...tags, ...weightFiles.map((f) => f.path)].join(' ');
    const nameMatch = sources.match(/(?:^|[\s\-_/])(\d+(?:\.\d+)?)[Bb](?:[-\s_]|$)/);
    let params = nameMatch ? parseFloat(nameMatch[1]) * 1e9 : null;
    if (params == null && totalWeightBytes > 0) {
      params =
        format === 'gguf'
          ? (totalWeightBytes * 8) / (detectQuant(sources)?.bits || 4.5)
          : totalWeightBytes / 2;
    }

    const variants =
      format === 'gguf'
        ? groupGgufShards(weightFiles)
            .map((g) => ({
              name: g.name,
              path: g.paths[0],
              paths: g.paths,
              shardCount: g.shardCount,
              sizeBytes: g.sizeBytes,
              sizeGb: round(g.sizeBytes / 1024 ** 3, 3),
              quant: detectQuant(g.name)?.label || null,
            }))
            .sort((a, b) => a.sizeBytes - b.sizeBytes)
        : null;

    const weightMeta = {
      modelId,
      format,
      weightFiles: weightFiles.map((f) => ({
        path: f.path,
        sizeBytes: f.size,
        sizeGb: round(f.size / 1024 ** 3, 3),
      })),
      totalWeightBytes,
      totalWeightGb: round(totalWeightBytes / 1024 ** 3, 3),
      variants,
      parameters: {
        count: params,
        billions: params != null ? round(params / 1e9, 3) : null,
        source: nameMatch ? 'name_or_tag' : params != null ? 'weight_size' : null,
      },
      pipelineTag: modelInfo?.pipeline_tag || null,
      tags,
      gated: !!modelInfo?.gated,
      cardUrl: `${HF}/${modelId}`,
      fetchedAt: new Date().toISOString(),
      offline: false,
    };

    await writeJson(file, { weightMeta, savedAt: new Date().toISOString() });
    return weightMeta;
  } catch (e) {
    if (cached?.weightMeta) return { ...cached.weightMeta, offline: true, stale: true };
    throw e;
  }
}

function detectQuant(name) {
  if (!name) return null;
  const n = name.toUpperCase();
  const rows = [
    [/Q8_0|Q8/, 8.5, 'Q8'],
    [/Q6_K|Q6/, 6.5, 'Q6'],
    [/Q5_K|Q5/, 5.5, 'Q5'],
    [/Q4_K|Q4_0|Q4/, 4.5, 'Q4'],
    [/Q3_K|Q3/, 3.5, 'Q3'],
    [/Q2_K|Q2/, 2.5, 'Q2'],
    [/FP16|F16|BF16/, 16, 'FP16'],
    [/FP32|F32/, 32, 'FP32'],
    [/INT8|8-BIT|8BIT/, 8, 'INT8'],
    [/INT4|4-BIT|4BIT|GPTQ|AWQ/, 4, 'INT4'],
  ];
  for (const [re, bits, label] of rows) {
    if (re.test(n)) return { bits, label };
  }
  return null;
}

function pickGgufVariant(wm, hw) {
  const budget = Math.max(hw?.budgets?.maxVramForModelGb || 0, hw?.budgets?.maxRamForModelGb || 0);
  return wm.variants.find((v) => v.sizeGb * OVERHEAD <= budget) || wm.variants[0];
}

// Some Hub repos are research artifacts for a project's own patched llama.cpp fork
// (custom quantization schemes, modified architectures) — mainline llama.cpp, which
// is what node-llama-cpp bundles, can't load these even though they're valid,
// complete GGUF files. There's no perfect signal for this, but tags naming a novel
// scheme, or a quant name that doesn't match any known llama.cpp format, are strong
// tells — worth warning about before someone downloads a multi-GB file for nothing.
const EXPERIMENTAL_TAG_RE = /\bternary\b|hybrid-attention|bitnet|1\.58-bit|\b1-bit\b|research-preview|\bmtp\b|multi-token-prediction/i;
// GGUF is a container format several unrelated *.cpp projects use for their own
// purposes — a .gguf file being present doesn't mean it's a llama.cpp text model.
// whisper.cpp/transcribe.cpp (speech-to-text) is the common one that shows up
// mixed into "gguf" searches; node-llama-cpp can't load these at all.
const NON_TEXT_GEN_PIPELINE_TAGS = new Set([
  'automatic-speech-recognition', 'text-to-speech', 'text-to-audio', 'audio-classification',
  'audio-to-audio', 'text-to-image', 'image-to-image', 'image-classification', 'image-segmentation',
  'object-detection', 'video-classification', 'text-to-video', 'feature-extraction', 'sentence-similarity',
]);

function detectExperimentalRisk(modelId, tags, pickedVariant, pipelineTag) {
  if (pipelineTag && NON_TEXT_GEN_PIPELINE_TAGS.has(pipelineTag)) {
    return `This is a ${pipelineTag.replace(/-/g, ' ')} model, not text generation — GGUF here likely means a different, unrelated *.cpp project's format, not something this app's engine can load.`;
  }
  const haystack = [modelId, ...(tags || [])].join(' ');
  if (EXPERIMENTAL_TAG_RE.test(haystack)) {
    return "Tagged as an experimental/research format — it may need a custom llama.cpp fork to load, not the standard build this app uses.";
  }
  if (pickedVariant?.name && !pickedVariant.quant) {
    return `"${pickedVariant.name}" doesn't match a standard llama.cpp quantization format — it may need special tooling to load.`;
  }
  return null;
}

// ─── real GGUF header inspection ──────────────────────────────────────────────
// Everything above this point infers compatibility from filenames and Hub tags —
// proxies for the truth. A GGUF file's own header is the truth: it's a small
// binary block at the START of the file (readable via HTTP Range requests, no
// full download needed) containing the model's real architecture name, exact
// parameter count from real tensor shapes, and the dimensions needed to compute
// exact KV-cache memory. @huggingface/gguf reads it the same way Hugging Face's
// own site does — confirmed by instrumenting it: ~4MB of Range requests to fully
// parse a 105MB model file, not the whole thing.

// Architectures llama.cpp's mainline build recognizes for TEXT GENERATION, as
// verified against real repos this session (llama, qwen35 confirmed directly).
// New architectures land in llama.cpp regularly, so an architecture NOT in this
// list isn't proven unsupported — just unconfirmed. Needs periodic refreshing
// against llama.cpp's own architecture registry.
const KNOWN_TEXT_GEN_ARCHITECTURES = new Set([
  'llama', 'llama4', 'deci', 'falcon', 'baichuan', 'grok', 'gpt2', 'gptj', 'gptneox',
  'mpt', 'starcoder', 'starcoder2', 'refact', 'bloom', 'stablelm',
  'qwen', 'qwen2', 'qwen2moe', 'qwen2vl', 'qwen3', 'qwen3moe', 'qwen35', 'qwen35moe',
  'phi2', 'phi3', 'phimoe', 'plamo', 'plamo2', 'codeshell', 'orion', 'internlm2',
  'minicpm', 'minicpm3', 'gemma', 'gemma2', 'gemma3', 'gemma3n', 'mamba', 'mamba2',
  'xverse', 'command-r', 'cohere2', 'dbrx', 'olmo', 'olmo2', 'olmoe', 'openelm',
  'arctic', 'deepseek', 'deepseek2', 'chatglm', 'glm4', 'bitnet', 't5', 't5encoder',
  'jais', 'nemotron', 'exaone', 'rwkv6', 'rwkv6qwen2', 'rwkv7', 'granite', 'granitemoe',
  'chameleon', 'plm', 'bailingmoe', 'seed-oss', 'ernie4_5', 'hunyuan-moe', 'lfm2',
]);
// Architectures that ARE valid GGUF, just definitively not text generation — a
// hard "can't run here", not a soft warning, since these are known for certain.
const KNOWN_NON_TEXT_GEN_ARCHITECTURES = new Set([
  'clip', 'whisper', 'wavtokenizer-dec', 'bert', 'nomic-bert', 'jina-bert-v2',
]);

async function fetchGgufHeader(url) {
  try {
    const { metadata, parameterCount } = await gguf(url, { computeParametersCount: true });
    return { ok: true, metadata, parameterCount };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}

function classifyGgufHeader(metadata) {
  const arch = metadata['general.architecture'];
  if (!arch) {
    return {
      runnable: false,
      reason: "This file's header has no model architecture — it's very likely auxiliary data (like an imatrix calibration file), not a standalone model.",
    };
  }
  if (KNOWN_NON_TEXT_GEN_ARCHITECTURES.has(arch)) {
    return {
      runnable: false,
      reason: `This is a real "${arch}" model, just not text generation — this app only runs text-generation GGUF models.`,
    };
  }
  if (!KNOWN_TEXT_GEN_ARCHITECTURES.has(arch)) {
    return {
      runnable: 'unknown',
      reason: `"${arch}" isn't in this app's known-supported architecture list yet — it might still work, but hasn't been confirmed.`,
    };
  }
  return { runnable: true, reason: null };
}

// Real KV-cache formula (not an approximation): for each of n_layer transformer
// layers, llama.cpp keeps a K and V tensor sized [n_ctx, n_head_kv, head_dim].
// n_head_kv < n_head under GQA, which is why this can differ a lot from a naive
// params-based guess. fp16 KV cache is llama.cpp/node-llama-cpp's default.
const KV_CACHE_BYTES_PER_ELEMENT = 2;
function realKvCacheGb(metadata, arch, contextTokens) {
  const nLayer = Number(metadata[`${arch}.block_count`]);
  const nHead = Number(metadata[`${arch}.attention.head_count`]);
  const nHeadKv = Number(metadata[`${arch}.attention.head_count_kv`] ?? nHead);
  const embedLength = Number(metadata[`${arch}.embedding_length`]);
  const keyLength = Number(metadata[`${arch}.attention.key_length`] ?? (nHead ? embedLength / nHead : NaN));
  const valueLength = Number(metadata[`${arch}.attention.value_length`] ?? keyLength);
  if (!nLayer || !nHeadKv || !keyLength || !valueLength) return null;
  const totalBytes = nLayer * contextTokens * nHeadKv * (keyLength + valueLength) * KV_CACHE_BYTES_PER_ELEMENT;
  return totalBytes / 1024 ** 3;
}

/**
 * Real, ground-truth metadata for one specific GGUF file — architecture, exact
 * parameter count, and exact per-context-length KV-cache cost. Cached forever
 * once fetched: a given file's header content never changes.
 */
async function getRealGgufMeta(modelId, variantPath) {
  const cacheFile = path.join(paths.meta, `${safeName(modelId + '::' + variantPath)}.realmeta.json`);
  const cached = await readJson(cacheFile);
  if (cached) return cached;
  if (!(await online())) return { ok: false, offline: true, error: 'Offline — real metadata not cached yet for this file.' };

  const repo = encodeRepo(modelId);
  const url = `${HF}/${repo}/resolve/main/${variantPath.split('/').map(encodeURIComponent).join('/')}`;
  const header = await fetchGgufHeader(url);
  if (!header.ok) return { ok: false, error: header.error };

  const arch = header.metadata['general.architecture'] || null;
  const classification = classifyGgufHeader(header.metadata);
  const contextEstimates = arch
    ? [4096, 8192, 32768]
        .map((tokens) => {
          const kvGb = realKvCacheGb(header.metadata, arch, tokens);
          return kvGb == null ? null : { tokens, kvGb: round(kvGb, 3) };
        })
        .filter(Boolean)
    : null;

  const result = {
    ok: true,
    architecture: arch,
    exactParamsBillions: header.parameterCount ? round(header.parameterCount / 1e9, 3) : null,
    maxContextLength: arch ? Number(header.metadata[`${arch}.context_length`]) || null : null,
    classification,
    contextEstimates: contextEstimates && contextEstimates.length ? contextEstimates : null,
  };
  await writeJson(cacheFile, result);
  return result;
}

function estimateMemory(wm, hw) {
  const weightGb = wm.totalWeightGb || 0;
  const pB = wm.parameters?.billions;
  const tags = (wm.tags || []).join(' ').toLowerCase();
  const fromP = (bpp) => (pB != null ? round(pB * bpp * OVERHEAD, 3) : null);
  const fromW = (s) => (weightGb ? round(weightGb * s * OVERHEAD, 3) : null);

  let estimates = {};
  if (wm.format === 'gguf' && wm.variants?.length) {
    estimates = {
      as_stored: {
        label: 'GGUF variants',
        variants: wm.variants.map((v) => ({
          name: v.name,
          path: v.path,
          quant: v.quant,
          loadGb: round(v.sizeGb * OVERHEAD, 3),
          weightGb: v.sizeGb,
        })),
        minLoadGb: round(wm.variants[0].sizeGb * OVERHEAD, 3),
      },
    };
  } else if (tags.includes('4-bit') || tags.includes('gptq') || tags.includes('awq')) {
    estimates = { as_stored: { label: 'As stored (4-bit)', loadGb: fromW(1) } };
  } else if (tags.includes('8-bit')) {
    estimates = {
      as_stored: { label: 'As stored (8-bit)', loadGb: fromW(1) },
      int4: { label: 'INT4', loadGb: fromP(MEM.int4) ?? fromW(0.5) },
    };
  } else if (wm.format) {
    estimates = {
      fp16: { label: 'FP16/BF16', loadGb: fromP(MEM.fp16) ?? fromW(1) },
      int8: { label: 'INT8', loadGb: fromP(MEM.int8) ?? fromW(0.5) },
      int4: { label: 'INT4', loadGb: fromP(MEM.int4) ?? fromW(0.25) },
    };
  } else {
    estimates = { unknown: { label: 'Estimated', loadGb: fromW(1) ?? fromP(MEM.fp16) } };
  }

  const primaryLoadGb =
    estimates.fp16?.loadGb ??
    estimates.as_stored?.minLoadGb ??
    estimates.as_stored?.loadGb ??
    estimates.int4?.loadGb ??
    estimates.unknown?.loadGb ??
    null;

  // For multi-variant GGUF repos, only one quant is ever actually downloaded —
  // downloadGb/diskspace estimates must reflect that file, not every variant summed.
  const downloadGb =
    wm.format === 'gguf' && wm.variants?.length && hw
      ? pickGgufVariant(wm, hw).sizeGb
      : weightGb || null;

  return {
    weightGb,
    parametersBillions: pB,
    format: wm.format,
    estimates,
    primaryLoadGb,
    downloadGb,
    contextEstimates: estimateContextMemory(pB),
  };
}

// KV-cache memory grows with context length and isn't covered by the base weight
// estimate above — a model can fit at 4K context and not at 32K. Real KV-cache size
// depends on per-model architecture (layer count, attention heads, GQA) that we don't
// have without parsing the GGUF header, so this is a labeled approximation tuned for
// a typical modern GQA architecture (Llama/Qwen/Mistral-class), not an exact figure.
const KV_GB_PER_TOKEN_PER_BILLION = 0.00002;
function estimateContextMemory(paramsBillions) {
  if (paramsBillions == null) return null;
  return [4096, 8192, 32768].map((tokens) => ({
    tokens,
    kvGb: round(tokens * paramsBillions * KV_GB_PER_TOKEN_PER_BILLION, 2),
  }));
}

function matchHardware(hw, mem, kvCacheGb) {
  const b = hw.budgets;
  // A model that only fits at a trivial context window isn't really a fit — the
  // verdict has to include what it costs to actually hold a conversation, not
  // just what it costs to load the weights. Default context (4096 tokens) is
  // used here; kvCacheGb is real per-model math when available (see
  // getRealGgufMeta), or the rougher fallback estimate otherwise.
  const extraGb = kvCacheGb || 0;
  const modes = [];
  const push = (key, label, weightLoadGb, device) => {
    if (weightLoadGb == null || weightLoadGb <= 0) return;
    const loadGb = round(weightLoadGb + extraGb, 3);
    const budget =
      device === 'gpu'
        ? b.maxVramForModelGb
        : device === 'unified'
          ? Math.max(b.maxVramForModelGb, b.maxRamForModelGb)
          : b.maxRamForModelGb;
    let status = 'unlikely';
    if (budget <= 0 && device === 'gpu') status = 'no_gpu';
    else if (loadGb <= budget * 0.7) status = 'excellent';
    else if (loadGb <= budget) status = 'good';
    else if (loadGb <= budget * 1.15 || (device === 'gpu' && loadGb <= b.maxRamForModelGb))
      status = 'tight';
    modes.push({
      key,
      label,
      loadGb,
      weightLoadGb,
      kvCacheGb: extraGb,
      device,
      budgetGb: round(budget, 2),
      headroomGb: round(budget - loadGb, 2),
      status,
    });
  };

  const est = mem.estimates || {};
  const gpuDev = b.canUseGpu ? (b.unifiedMemory ? 'unified' : 'gpu') : 'cpu';

  if (est.as_stored?.variants) {
    for (const v of est.as_stored.variants.slice(0, 16)) {
      push(`g:${v.name}`, `GGUF ${v.quant || ''} · ${v.name}`, v.loadGb, gpuDev);
      if (gpuDev !== 'cpu') push(`c:${v.name}`, `CPU · ${v.name}`, v.loadGb, 'cpu');
    }
  } else {
    for (const [k, v] of Object.entries(est)) {
      if (!v?.loadGb) continue;
      push(k, v.label, v.loadGb, gpuDev);
      if (gpuDev !== 'cpu') push(`${k}_cpu`, `${v.label} (CPU)`, v.loadGb, 'cpu');
    }
  }

  const rank = { excellent: 4, good: 3, tight: 2, unlikely: 1, no_gpu: 0 };
  modes.sort((a, c) => rank[c.status] - rank[a.status] || a.loadGb - c.loadGb);
  const best = modes.find((m) => rank[m.status] >= 2) || modes[0] || null;
  const overall = best
    ? best.status === 'no_gpu'
      ? 'unlikely'
      : best.status
    : mem.primaryLoadGb == null
      ? 'unknown'
      : 'unknown';

  const diskOk =
    mem.downloadGb == null || b.freeDiskGb == null || mem.downloadGb <= b.freeDiskGb;

  const summary = {
    excellent: `Yes — comfortable fit${best ? ` via ${best.label}` : ''}.`,
    good: `Yes — should run${best ? ` via ${best.label}` : ''}.`,
    tight: `Maybe — tight fit${best ? ` via ${best.label}` : ''}.`,
    unlikely: 'Unlikely without more RAM/VRAM or a smaller quant.',
    unknown: 'Not enough Hub metadata to estimate.',
  }[overall];

  return {
    overall,
    bestMode: best,
    modes,
    disk: { downloadGb: mem.downloadGb, freeDiskGb: b.freeDiskGb, ok: diskOk },
    summary: diskOk ? summary : 'May fit memory, but free disk looks too low to download.',
  };
}

// Real data, when we have it, is authoritative for "is this even a text-gen
// model" — but it can't catch everything (MTP's architecture string looks
// perfectly normal), so a heuristic-only finding still surfaces if real data
// gives the all-clear.
function combineRunnabilityWarning(heuristicWarning, realMeta) {
  if (realMeta?.ok && realMeta.classification && realMeta.classification.runnable !== true) {
    return realMeta.classification.reason;
  }
  return heuristicWarning;
}

const DEFAULT_CONTEXT_TOKENS = 4096;
function kvCacheGbForVerdict(realMeta, paramsBillions) {
  if (realMeta?.ok && realMeta.contextEstimates) {
    const at = realMeta.contextEstimates.find((c) => c.tokens === DEFAULT_CONTEXT_TOKENS);
    if (at) return at.kvGb;
  }
  const fallback = estimateContextMemory(paramsBillions);
  const at = fallback?.find((c) => c.tokens === DEFAULT_CONTEXT_TOKENS);
  return at ? at.kvGb : 0;
}

async function analyze(modelId) {
  const [hw, weightMeta] = await Promise.all([detectHardware(), getWeightMeta(modelId)]);
  const pickedVariant = weightMeta.format === 'gguf' && weightMeta.variants?.length ? pickGgufVariant(weightMeta, hw) : null;
  const realMeta = pickedVariant ? await getRealGgufMeta(modelId, pickedVariant.path) : null;
  if (realMeta?.ok && realMeta.exactParamsBillions) weightMeta.parameters.exactBillions = realMeta.exactParamsBillions;
  const memory = estimateMemory(weightMeta, hw);
  const kvCacheGb = kvCacheGbForVerdict(realMeta, weightMeta.parameters?.exactBillions ?? memory.parametersBillions);
  const match = matchHardware(hw, memory, kvCacheGb);
  const heuristicWarning = detectExperimentalRisk(modelId, weightMeta.tags, pickedVariant, weightMeta.pipelineTag);
  const ir = fromWeightMeta(weightMeta, realMeta);
  // Both detectExperimentalRisk (pipeline-tag heuristic) and
  // combineRunnabilityWarning (GGUF header classification) exist to catch
  // "this isn't really a chat model" — a warning against the CHAT path
  // specifically. Now that asr/embed have real adapters, the exact same
  // "not text generation" fact is expected and correct for those families,
  // not something to warn about. An earlier version of this fix only gated
  // the header-classification half and left detectExperimentalRisk's
  // pipeline-tag check firing unconditionally — caught via a real
  // canRunHere check against a live embedding model (came back
  // export_only when it should have been yes), not just re-reading the code.
  const experimentalWarning = ir.family === 'llm' ? combineRunnabilityWarning(heuristicWarning, realMeta) : null;
  // Reranker/cross-encoder repos (pipeline_tag "text-ranking") aren't in
  // ir.js's PIPELINE_FAMILY map at all, so they'd fall through to family
  // 'other' / job 'chat' — which would incorrectly let them match the llama
  // adapter's CHAT path instead of the new rerank capability. Override job
  // explicitly rather than widen ir.js's family map for one pipeline tag.
  const jobOverride = weightMeta.pipelineTag === 'text-ranking' ? 'rerank' : undefined;
  const runPlan = plan(ir, hw, match, { experimentalWarning, job: jobOverride });
  const result = {
    modelId,
    weightMeta,
    ir,
    runPlan,
    memory,
    match,
    experimentalWarning,
    realMeta,
    hardwareSnapshot: {
      ramGb: hw.memory.totalGb,
      vramGb: hw.gpu.totalVramGb,
      budgets: hw.budgets,
      gpu: hw.gpu.primary?.model || null,
      cpu: hw.cpu.brand,
    },
    local: await localModelStatus(modelId),
  };
  // pin analysis for offline
  const file = path.join(paths.meta, `${safeName(modelId)}.json`);
  const prev = (await readJson(file)) || {};
  await writeJson(file, {
    ...prev,
    weightMeta,
    lastAnalysis: {
      overall: match.overall,
      summary: match.summary,
      primaryLoadGb: memory.primaryLoadGb,
      bestMode: match.bestMode,
      canRunHere: runPlan.canRunHere,
      family: ir.family,
      job: runPlan.job,
      at: new Date().toISOString(),
    },
  });
  return result;
}

// A short "why this pick" label pulled from Hub tags/id, for the Home screen —
// cheap signal for what a model is actually good at beyond just "it fits."
const USE_TAG_RULES = [
  [/coder|coding|code-generation/i, 'Good for coding'],
  [/function-calling|tool-use|agent/i, 'Agentic / tool use'],
  [/reasoning/i, 'Reasoning-focused'],
  [/roleplay/i, 'Roleplay / character chat'],
  [/uncensored|abliterated/i, 'Uncensored'],
  [/vision|multimodal|image-text-to-text/i, 'Vision / multimodal'],
  [/multilingual/i, 'Multilingual'],
  [/math/i, 'Math-focused'],
];
function deriveUseTag(modelId, tags) {
  const haystack = [modelId, ...(tags || [])].join(' ');
  for (const [re, label] of USE_TAG_RULES) {
    if (re.test(haystack)) return label;
  }
  return 'General chat';
}

async function analyzeIds(hw, ids) {
  const results = [];
  for (let i = 0; i < ids.length; i += 3) {
    const chunk = ids.slice(i, i + 3);
    const part = await Promise.all(
      chunk.map(async (modelId) => {
        try {
          const wm = await getWeightMeta(modelId);
          const pickedVariant = wm.format === 'gguf' && wm.variants?.length ? pickGgufVariant(wm, hw) : null;
          const realMeta = pickedVariant ? await getRealGgufMeta(modelId, pickedVariant.path) : null;
          if (realMeta?.ok && realMeta.exactParamsBillions) wm.parameters.exactBillions = realMeta.exactParamsBillions;
          const memory = estimateMemory(wm, hw);
          const kvCacheGb = kvCacheGbForVerdict(realMeta, wm.parameters?.exactBillions ?? memory.parametersBillions);
          const match = matchHardware(hw, memory, kvCacheGb);
          const heuristicWarning = detectExperimentalRisk(modelId, wm.tags, pickedVariant, wm.pipelineTag);
          const ir = fromWeightMeta(wm, realMeta);
          // See analyze()'s matching comment: BOTH warning sources only
          // apply to the chat path, not just the header-classification one.
          const experimentalWarning = ir.family === 'llm' ? combineRunnabilityWarning(heuristicWarning, realMeta) : null;
          // See analyze()'s matching comment on text-ranking job override.
          const jobOverride = wm.pipelineTag === 'text-ranking' ? 'rerank' : undefined;
          const runPlan = plan(ir, hw, match, { experimentalWarning, job: jobOverride });
          const file = path.join(paths.meta, `${safeName(modelId)}.json`);
          const prev = (await readJson(file)) || {};
          await writeJson(file, {
            ...prev,
            weightMeta: wm,
            lastAnalysis: {
              overall: match.overall,
              summary: match.summary,
              primaryLoadGb: memory.primaryLoadGb,
              bestMode: match.bestMode,
              canRunHere: runPlan.canRunHere,
              family: ir.family,
              job: runPlan.job,
              at: new Date().toISOString(),
            },
          });
          return {
            modelId,
            ok: true,
            overall: match.overall,
            summary: match.summary,
            bestMode: match.bestMode,
            primaryLoadGb: memory.primaryLoadGb,
            downloadGb: memory.downloadGb,
            format: wm.format,
            family: ir.family,
            runPlan,
            useTag: deriveUseTag(modelId, wm.tags),
            experimentalWarning,
            local: await localModelStatus(modelId),
          };
        } catch (err) {
          return { modelId, ok: false, error: String(err.message || err) };
        }
      })
    );
    results.push(...part);
  }
  return results;
}

const FIT_RANK = { excellent: 4, good: 3, tight: 2, unlikely: 1, no_gpu: 0, unknown: 0 };
// Below this, a "model" is almost certainly a broken/incomplete upload (config-only,
// stub, or partial GGUF) rather than something actually worth recommending.
const MIN_PLAUSIBLE_MODEL_GB = 0.05;

/**
 * Curated picks for the Home screen: popular GGUF chat models, ranked by how
 * well they actually fit this machine. Candidate order comes from the Hub's
 * own trending sort — Array.sort is stable, so filtering by fit rank preserves
 * that relevance order instead of second-guessing it by file size.
 */
const RECO_TTL_MS = 20 * 60 * 1000; // recomputing means ~40 live HF lookups — too slow to redo every launch

async function pickRunnable(hw, query, label) {
  const list = await listModels({ ...query, sort: 'trendingScore', limit: 12, page: 0 });
  const ids = (list.models || []).filter((m) => !m.gated).map((m) => m.id);
  const results = (await analyzeIds(hw, ids)).filter(
    (r) => r.ok && r.runPlan?.canRunHere === 'yes' && (r.primaryLoadGb == null || r.primaryLoadGb >= MIN_PLAUSIBLE_MODEL_GB)
  );
  const fits = results.filter((r) => FIT_RANK[r.overall] >= 3);
  const pick = fits[0] || results.sort((a, b) => FIT_RANK[b.overall] - FIT_RANK[a.overall])[0] || null;
  if (pick) pick.useTag = label;
  return { pick, offline: !!list.offline };
}

async function getRecommendations(force) {
  const hw = await detectHardware();
  const recoFile = path.join(paths.cache, 'recommendations.json');
  const cached = await readJson(recoFile);
  const cacheIsFresh = cached && Date.now() - new Date(cached.at).getTime() < RECO_TTL_MS;
  if (!force && cacheIsFresh && cached.data.picks && cached.data.picks.length) {
    return { ...cached.data, budgets: hw.budgets, cached: true };
  }

  const jobs = await Promise.all([
    pickRunnable(hw, { pipelineTag: 'text-generation', search: 'GGUF' }, 'Talk'),
    pickRunnable(hw, { pipelineTag: 'text-to-image', search: 'GGUF' }, 'See'),
    pickRunnable(hw, { pipelineTag: 'automatic-speech-recognition', search: 'ggml' }, 'Hear · transcribe'),
    pickRunnable(hw, { pipelineTag: 'text-to-speech', search: 'onnx piper' }, 'Hear · speak'),
    pickRunnable(hw, { pipelineTag: 'feature-extraction', search: 'GGUF' }, 'Embed'),
    pickRunnable(hw, { pipelineTag: 'text-ranking', search: 'GGUF' }, 'Rank'),
  ]);
  const picks = jobs.map((j) => j.pick).filter(Boolean);
  const offline = jobs.some((j) => j.offline);
  const data = { picks, offline };
  if (!offline && picks.length) await writeJson(recoFile, { at: new Date().toISOString(), data });
  return { ...data, budgets: hw.budgets, cached: false };
}

// ─── local downloads (offline models) ────────────────────────────────────────

function modelDir(modelId) {
  return path.join(paths.models, safeName(modelId));
}

async function localModelStatus(modelId) {
  const dir = modelDir(modelId);
  const manifest = await readJson(path.join(dir, 'manifest.json'));
  if (!manifest) return { downloaded: false, dir: null, files: [] };
  const files = [];
  for (const f of manifest.files || []) {
    const p = path.join(dir, f.localName || path.basename(f.path));
    let ok = false;
    try {
      const st = await fsp.stat(p);
      ok = st.size > 0 && (!f.sizeBytes || st.size === f.sizeBytes || st.size > f.sizeBytes * 0.99);
    } catch {
      ok = false;
    }
    files.push({ ...f, present: ok, localPath: p });
  }
  const complete = files.length > 0 && files.every((f) => f.present);
  return {
    downloaded: complete,
    partial: files.some((f) => f.present) && !complete,
    dir,
    files,
    manifest,
  };
}

async function listLocalModels() {
  const entries = await fsp.readdir(paths.models, { withFileTypes: true }).catch(() => []);
  const hw = await detectHardware();
  const out = [];
  for (const e of entries) {
    if (!e.isDirectory()) continue;
    const man = await readJson(path.join(paths.models, e.name, 'manifest.json'));
    if (!man?.modelId) continue;
    const status = await localModelStatus(man.modelId);
    const metaFile = path.join(paths.meta, `${safeName(man.modelId)}.json`);
    const meta = await readJson(metaFile);
    let lastAnalysis = meta?.lastAnalysis || null;
    // Re-score against current hardware (works fully offline if meta cached)
    if (meta?.weightMeta) {
      const memory = estimateMemory(meta.weightMeta, hw);
      const match = matchHardware(hw, memory);
      const ir = fromWeightMeta(meta.weightMeta, null);
      const jobOverride = meta.weightMeta.pipelineTag === 'text-ranking' ? 'rerank' : undefined;
      const runPlan = plan(ir, hw, match, { job: jobOverride });
      lastAnalysis = {
        overall: match.overall,
        summary: match.summary,
        primaryLoadGb: memory.primaryLoadGb,
        bestMode: match.bestMode,
        canRunHere: runPlan.canRunHere,
        family: ir.family,
        job: runPlan.job,
        at: new Date().toISOString(),
      };
    }
    out.push({
      modelId: man.modelId,
      ...status,
      lastAnalysis,
      totalWeightGb: man.totalWeightGb ?? null,
    });
  }
  return out;
}

const activeDownloads = new Map(); // modelId -> { req, canceled }

function cancelDownload(modelId) {
  const handle = activeDownloads.get(modelId);
  if (!handle) return { wasActive: false };
  handle.canceled = true;
  handle.req?.destroy(new Error('Canceled by user'));
  return { wasActive: true };
}

function downloadFile(url, dest, onProgress, handle) {
  return new Promise((resolve, reject) => {
    const tmp = `${dest}.part`;
    // A failed/canceled download's .part file is left in place on purpose — it's
    // the checkpoint the next attempt resumes from via a Range request, not junk
    // to clean up. It's only ever consumed (renamed to dest) or wiped by deleting
    // the model entirely.
    const fail = (e) => reject(e);
    let resumeFrom = 0;
    try {
      const st = fs.statSync(tmp);
      if (st.size > 0) resumeFrom = st.size;
    } catch {
      /* no partial file to resume from */
    }

    const doGet = (u, redirects = 0) => {
      if (handle?.canceled) return fail(new Error('Canceled'));
      if (redirects > 10) return fail(new Error('Too many redirects'));
      let parsed;
      try {
        parsed = new URL(u);
      } catch {
        return fail(new Error(`Invalid download URL: ${u}`));
      }
      const lib = parsed.protocol === 'http:' ? http : https;
      const headers = { 'User-Agent': 'CanYouRunAI/1.0' };
      if (resumeFrom > 0) headers.Range = `bytes=${resumeFrom}-`;
      const req = lib.get(parsed, { headers, timeout: 120000 }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          // HF often returns relative redirect paths
          const next = new URL(res.headers.location, parsed).href;
          return doGet(next, redirects + 1);
        }
        // Some servers ignore Range and just send the whole file again from byte 0.
        const resumed = resumeFrom > 0 && res.statusCode === 206;
        if (resumeFrom > 0 && !resumed) resumeFrom = 0;
        if (res.statusCode !== 200 && res.statusCode !== 206) {
          res.resume();
          return fail(new Error(`Download HTTP ${res.statusCode}`));
        }
        const total = resumed ? resumeFrom + (parseInt(res.headers['content-length'] || '0', 10) || 0) : parseInt(res.headers['content-length'] || '0', 10) || 0;
        let received = resumed ? resumeFrom : 0;
        fs.mkdirSync(path.dirname(dest), { recursive: true });
        const out = fs.createWriteStream(tmp, resumed ? { flags: 'a' } : undefined);
        res.on('data', (chunk) => {
          received += chunk.length;
          if (onProgress) onProgress({ received, total });
        });
        res.pipe(out);
        out.on('finish', () => {
          out.close(() => {
            try {
              fs.renameSync(tmp, dest);
              resolve({ received, total });
            } catch (e) {
              fail(e);
            }
          });
        });
        out.on('error', fail);
      });
      if (handle) handle.req = req;
      req.on('error', fail);
      req.on('timeout', () => {
        req.destroy();
        fail(new Error('Download timeout'));
      });
    };
    doGet(url);
  });
}

/**
 * Download smallest fitting weight set for offline use.
 * GGUF: smallest variant that fits, else smallest overall.
 * Otherwise: all safetensors/bin shards.
 */
async function downloadModel(modelId, opts = {}) {
  if (!(await online())) throw new Error('Need network to download models.');
  const hw = await detectHardware();
  const wm = await getWeightMeta(modelId);
  const pick = wm.format === 'gguf' && wm.variants?.length ? pickGgufVariant(wm, hw) : null;
  // Real per-file metadata, fetched (and cached forever) here if this is the first
  // time this exact file's been touched — a real pre-download check, not just a
  // display nicety, and free for any later analyze() call on the same file.
  const realMeta = pick ? await getRealGgufMeta(modelId, pick.path) : null;
  const mem = estimateMemory(wm, hw);
  const kvCacheGb = kvCacheGbForVerdict(realMeta, realMeta?.exactParamsBillions ?? mem.parametersBillions);
  const match = matchHardware(hw, mem, kvCacheGb);

  let files = [];
  if (wm.format === 'gguf' && wm.variants?.length) {
    const shardPaths = pick.paths && pick.paths.length ? pick.paths : [pick.path];
    // Shards must all land in the same folder under their original filenames —
    // that's how node-llama-cpp finds sibling parts when it loads shard 1.
    files = shardPaths.map((p) => {
      const wf = (wm.weightFiles || []).find((f) => f.path === p);
      return { path: p, sizeBytes: wf ? wf.sizeBytes : Math.round(pick.sizeBytes / shardPaths.length) };
    });
  } else {
    files = (wm.weightFiles || []).map((f) => ({
      path: f.path,
      sizeBytes: f.sizeBytes,
    }));
  }

  if (!files.length) throw new Error('No downloadable weight files found on Hub.');

  const dir = modelDir(modelId);
  await fsp.mkdir(dir, { recursive: true });
  const repo = encodeRepo(modelId);
  const saved = [];
  const handle = { req: null, canceled: false };
  activeDownloads.set(modelId, handle);

  try {
    for (let i = 0; i < files.length; i++) {
      if (handle.canceled) throw new Error('Canceled');
      const f = files[i];
      const localName = path.basename(f.path);
      const dest = path.join(dir, localName);
      // skip if already complete
      try {
        const st = await fsp.stat(dest);
        if (f.sizeBytes && st.size >= f.sizeBytes * 0.99) {
          saved.push({ ...f, localName, skipped: true });
          win?.webContents.send('download:progress', {
            modelId,
            file: localName,
            index: i,
            totalFiles: files.length,
            percent: 100,
            done: false,
          });
          continue;
        }
      } catch {
        /* download */
      }

      const url = `${HF}/${repo}/resolve/main/${f.path.split('/').map(encodeURIComponent).join('/')}`;
      await downloadFile(
        url,
        dest,
        ({ received, total }) => {
          const pct = total ? Math.round((received / total) * 100) : 0;
          win?.webContents.send('download:progress', {
            modelId,
            file: localName,
            index: i,
            totalFiles: files.length,
            received,
            total,
            percent: pct,
            done: false,
          });
        },
        handle
      );
      saved.push({ ...f, localName });
    }
  } finally {
    activeDownloads.delete(modelId);
  }

  const manifest = {
    modelId,
    format: wm.format,
    files: saved.map((f) => ({
      path: f.path,
      localName: f.localName,
      sizeBytes: f.sizeBytes,
    })),
    totalWeightGb: round(
      saved.reduce((s, f) => s + (f.sizeBytes || 0), 0) / 1024 ** 3,
      3
    ),
    downloadedAt: new Date().toISOString(),
    matchSnapshot: {
      overall: match.overall,
      bestMode: match.bestMode,
    },
  };
  await writeJson(path.join(dir, 'manifest.json'), manifest);
  win?.webContents.send('download:progress', {
    modelId,
    percent: 100,
    done: true,
  });

  return { ok: true, dir, manifest, local: await localModelStatus(modelId) };
}

async function deleteLocalModel(modelId) {
  const dir = modelDir(modelId);
  await fsp.rm(dir, { recursive: true, force: true });
  return { ok: true };
}

const CHAT_HISTORY_LIMIT = 200; // cap so a long-lived conversation doesn't grow the file forever

async function getChatHistory(modelId) {
  const log = await readJson(path.join(modelDir(modelId), 'chat.json'));
  return Array.isArray(log) ? log : [];
}

async function saveChatHistory(modelId, log) {
  const trimmed = Array.isArray(log) ? log.slice(-CHAT_HISTORY_LIMIT) : [];
  await writeJson(path.join(modelDir(modelId), 'chat.json'), trimmed);
  return { ok: true };
}

// ─── in-app inference (GGUF via out-of-process llama adapter) ────────────────
// node-llama-cpp used to load and run in THIS process. It now runs in its own
// child process (electron/adapters/llama/worker.js), spawned and supervised
// by engine/host.js, so a crash in there can't take the whole app down with
// it. Function names/signatures below are unchanged from the old in-process
// versions on purpose — every caller (the chat:* IPC handlers, the selftest)
// needed zero changes.

function readLocalManifest(modelId) {
  try {
    return JSON.parse(fs.readFileSync(path.join(modelDir(modelId), 'manifest.json'), 'utf8'));
  } catch {
    return null;
  }
}

function findGgufFile(modelId) {
  const dir = modelDir(modelId);
  const manifest = readLocalManifest(modelId);
  if (!manifest || manifest.format !== 'gguf' || !manifest.files?.length) return null;
  // A sharded model needs every part present in the same folder — node-llama-cpp
  // finds sibling shards next to whichever one it's pointed at.
  for (const f of manifest.files) {
    if (!fs.existsSync(path.join(dir, f.localName || path.basename(f.path)))) return null;
  }
  const first = manifest.files[0];
  return path.join(dir, first.localName || path.basename(first.path));
}

let activeChatModelId = null; // mirrors the worker's own state, for the parameterless unloadChat()

async function unloadChat() {
  const modelId = activeChatModelId;
  activeChatModelId = null;
  if (!modelId) return;
  await llamaAdapter.unload(modelId);
}

async function loadChatModel(modelId, opts = {}) {
  const file = findGgufFile(modelId);
  if (!file) throw new Error('GGUF file(s) missing or incomplete on disk. Remove and re-download this model.');
  if (activeChatModelId && activeChatModelId !== modelId) await unloadChat();
  const hw = await detectHardware();
  const result = await llamaAdapter.load(modelId, file, opts, {
    device: hw.budgets.canUseGpu ? 'gpu' : 'cpu',
  });
  activeChatModelId = modelId;
  return { modelId, ready: !!result.ready };
}

function stopGeneration(modelId) {
  return llamaAdapter.stop(modelId);
}

async function sendChatMessage(modelId, message, opts = {}) {
  if (activeChatModelId !== modelId) throw new Error('Model not loaded. Load it first.');
  const result = await llamaAdapter.send(modelId, message, opts, (text) =>
    win?.webContents.send('chat:chunk', { modelId, text })
  );
  return { answer: result.answer, tokensPerSecond: result.tokensPerSecond, tokenCount: result.tokenCount };
}

// ─── in-app image generation (sd.cpp adapter) ─────────────────────────────────
// sd-cli.exe is a one-shot-per-image native process, not a persistent worker,
// so there's no load/unload pair like chat — each call is its own job. See
// electron/adapters/sd/index.js for why this doesn't go through host.js.

function findSdModelFile(modelId) {
  const dir = modelDir(modelId);
  const manifest = readLocalManifest(modelId);
  if (!manifest || (manifest.format !== 'safetensors' && manifest.format !== 'gguf')) return null;
  const first = manifest.files?.[0];
  if (!first) return null;
  const full = path.join(dir, first.localName || path.basename(first.path));
  return fs.existsSync(full) ? full : null;
}

const activeGenerations = new Map(); // modelId -> { jobId, adapter }, so cancel can route to whichever adapter (sd.cpp or pydiffusion) actually handled this job
const canceledGenerations = new Set();

// Adapters return { ram, vram }; they reserve { ramGb, vramGb }. Alias here so
// the GPU lock actually sees a number. CPU jobs must not take the GPU lock.
function asHwEstimate(est, device) {
  const ramGb = est.ramGb ?? est.ram ?? 0;
  const vramGb = est.vramGb ?? est.vram ?? 0;
  const dev = device || est.device || 'cpu';
  return {
    ...est,
    ramGb,
    vramGb: dev === 'gpu' ? vramGb : 0,
    device: dev,
  };
}

// Multi-file diffusers pipelines never land a single local checkpoint via
// findSdModelFile (sd.cpp can't load them at all — see adapters/sd's canRun)
// — they go through pydiffusion instead, which downloads via Python's own
// huggingface_hub machinery (redirected into this app's own storage by
// pyhost's HF_HOME, see engine/pyhost.js) rather than this file's byte-level
// downloadModel(). That means no local-file precondition to check here; the
// repo id itself is what pydiffusion's worker needs.
async function generateImagePy(modelId, opts, hw) {
  const jobId = `t2i:py:${modelId}:${Date.now()}`;
  const outputPath = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}.png`);
  const onProgress = (text) => win?.webContents.send('sd:progress', { modelId, text });
  canceledGenerations.delete(modelId);
  activeGenerations.set(modelId, { jobId, adapter: pydiffusionAdapter });
  try {
    const { outputPath: written } = await pydiffusionAdapter.generate(
      jobId,
      {
        modelPath: modelId,
        prompt: opts.prompt,
        negativePrompt: opts.negativePrompt,
        width: opts.width || 512,
        height: opts.height || 512,
        steps: opts.steps,
        cfgScale: opts.cfgScale,
        seed: opts.seed,
        outputPath,
        hwEstimate: asHwEstimate(
          pydiffusionAdapter.estimateResources({ files: [] }, hw, { width: opts.width, height: opts.height }),
          hw.budgets?.canUseGpu ? 'gpu' : 'cpu'
        ),
        hardware: hw,
      },
      onProgress
    );
    return { outputPath: written, backend: 'python' };
  } finally {
    activeGenerations.delete(modelId);
    canceledGenerations.delete(modelId);
  }
}

async function generateImage(modelId, opts = {}) {
  const file = findSdModelFile(modelId);
  const hw = await detectHardware();
  // No single-checkpoint file on disk for this model — sd.cpp has nothing to
  // load (it never will; this is the multi-file-pipeline case, not "not
  // downloaded yet"). Route to the Python adapter instead of failing.
  if (!file) return generateImagePy(modelId, opts, hw);
  const width = opts.width || 512;
  const height = opts.height || 512;
  const jobId = `t2i:${modelId}:${Date.now()}`;
  const outputPath = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}.png`);
  // Never pass cuda — sd-cli cuda is not a pinned backend. Vulkan may DLL-fail
  // on some PCs; we retry CPU once. Cancel skips the retry.
  let backend = opts.backend || pickBackend(hw, 'diffusion') || 'cpu';
  if (backend === 'cuda') backend = hw.backends?.vulkan ? 'vulkan' : 'cpu';

  const est = sdAdapter.estimateResources(
    { files: [{ role: 'weights', size: fs.statSync(file).size }] },
    hw,
    { width, height }
  );
  const onProgress = (rawLogLine) => win?.webContents.send('sd:progress', { modelId, text: rawLogLine });

  const attempt = (b, jid) =>
    sdAdapter.generate(
      jid,
      {
        modelPath: file,
        prompt: opts.prompt,
        negativePrompt: opts.negativePrompt,
        width,
        height,
        steps: opts.steps,
        cfgScale: opts.cfgScale,
        seed: opts.seed,
        outputPath,
        backend: b,
        initImagePath: opts.initImagePath,
        maskImagePath: opts.maskImagePath,
        strength: opts.strength,
        hwEstimate: asHwEstimate(est, b === 'vulkan' ? 'gpu' : 'cpu'),
      },
      onProgress
    );

  canceledGenerations.delete(modelId);
  activeGenerations.set(modelId, { jobId, adapter: sdAdapter });
  try {
    try {
      const { outputPath: written } = await attempt(backend, jobId);
      return { outputPath: written, backend };
    } catch (e) {
      const startFail = /No verified sd-cli|STATUS_DLL|0xC0000135|3221225781|DLL_NOT_FOUND|ENOENT|spawn UNKNOWN|Download HTTP|sha256 mismatch|Expand-Archive|sd-cli.exe not found|sd-cli not found/i.test(String(e && e.message || e));
      if (backend === 'cpu' || canceledGenerations.has(modelId) || !startFail) throw e;
      onProgress('\nGPU image runner failed to start — retrying on CPU. First image can take several minutes.\n');
      const cpuJobId = `${jobId}:cpu`;
      activeGenerations.set(modelId, { jobId: cpuJobId, adapter: sdAdapter });
      const { outputPath: written } = await attempt('cpu', cpuJobId);
      return { outputPath: written, backend: 'cpu', fellBack: true };
    }
  } finally {
    activeGenerations.delete(modelId);
    canceledGenerations.delete(modelId);
  }
}

function cancelGeneration(modelId) {
  const entry = activeGenerations.get(modelId);
  canceledGenerations.add(modelId);
  if (!entry) return { stopped: false };
  return entry.adapter.cancel(entry.jobId);
}

// ─── in-app video/music generation (Python/diffusers + Python/transformers) ──
// Same "no local predownload, repo id goes straight to the Python worker"
// shape as generateImagePy above — see that function's comment for why.
const activeVideoJobs = new Map();
const activeMusicJobs = new Map();

async function generateVideo(modelId, opts = {}) {
  const hw = await detectHardware();
  const jobId = `t2v:${modelId}:${Date.now()}`;
  const outputPath = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}.mp4`);
  const onProgress = (text) => win?.webContents.send('video:progress', { modelId, text });
  activeVideoJobs.set(modelId, jobId);
  try {
    const { outputPath: written } = await pyvideoAdapter.generate(
      jobId,
      {
        modelPath: modelId,
        prompt: opts.prompt,
        negativePrompt: opts.negativePrompt,
        numFrames: opts.numFrames,
        fps: opts.fps,
        steps: opts.steps,
        seed: opts.seed,
        outputPath,
        hwEstimate: asHwEstimate(
          pyvideoAdapter.estimateResources({ files: [] }, hw, { width: opts.width, height: opts.height, numFrames: opts.numFrames }),
          hw.budgets?.canUseGpu ? 'gpu' : 'cpu'
        ),
        hardware: hw,
      },
      onProgress
    );
    return { outputPath: written };
  } finally {
    activeVideoJobs.delete(modelId);
  }
}

function cancelVideo(modelId) {
  const jobId = activeVideoJobs.get(modelId);
  if (!jobId) return { stopped: false };
  return pyvideoAdapter.cancel(jobId);
}

async function generateMusic(modelId, opts = {}) {
  const hw = await detectHardware();
  const jobId = `t2a:${modelId}:${Date.now()}`;
  const outputPath = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}.wav`);
  const onProgress = (text) => win?.webContents.send('music:progress', { modelId, text });
  activeMusicJobs.set(modelId, jobId);
  try {
    const { outputPath: written } = await pymusicAdapter.generate(
      jobId,
      {
        modelPath: modelId,
        prompt: opts.prompt,
        durationSeconds: opts.durationSeconds,
        outputPath,
        hwEstimate: asHwEstimate(
          pymusicAdapter.estimateResources({ files: [] }, hw, { durationSeconds: opts.durationSeconds }),
          hw.budgets?.canUseGpu ? 'gpu' : 'cpu'
        ),
        hardware: hw,
      },
      onProgress
    );
    return { outputPath: written };
  } finally {
    activeMusicJobs.delete(modelId);
  }
}

function cancelMusic(modelId) {
  const jobId = activeMusicJobs.get(modelId);
  if (!jobId) return { stopped: false };
  return pymusicAdapter.cancel(jobId);
}

// ─── in-app speech (whisper.cpp transcribe + piper speak) ─────────────────────
// Same one-shot-native-process shape as sd.cpp above — see
// electron/adapters/whisper|piper/index.js for why these don't go through
// host.js's fork-based worker model.

function findAsrModelFile(modelId) {
  const dir = modelDir(modelId);
  const manifest = readLocalManifest(modelId);
  if (!manifest || (manifest.format !== 'bin' && manifest.format !== 'gguf')) return null;
  const first = manifest.files?.[0];
  if (!first) return null;
  const full = path.join(dir, first.localName || path.basename(first.path));
  return fs.existsSync(full) ? full : null;
}

function findTtsVoiceFile(modelId) {
  const dir = modelDir(modelId);
  const manifest = readLocalManifest(modelId);
  if (!manifest || manifest.format !== 'onnx') return null;
  const first = manifest.files?.[0];
  if (!first) return null;
  const full = path.join(dir, first.localName || path.basename(first.path));
  return fs.existsSync(full) ? { path: full, hfRelativePath: first.path } : null;
}

const activeAsrJobs = new Map(); // modelId -> jobId
const activeTtsJobs = new Map(); // modelId -> jobId

async function transcribeAudio(modelId, audioPath, opts = {}) {
  const file = findAsrModelFile(modelId);
  if (!file) throw new Error('Model file missing or incomplete on disk. Remove and re-download this model.');
  if (!fs.existsSync(audioPath)) throw new Error('Audio file not found.');
  const hw = await detectHardware();
  const jobId = `asr:${modelId}:${Date.now()}`;
  const outputBase = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}`);

  activeAsrJobs.set(modelId, jobId);
  try {
    const { text } = await whisperAdapter.transcribe(
      jobId,
      {
        modelPath: file,
        audioPath,
        language: opts.language,
        outputBase,
        hwEstimate: asHwEstimate(
          whisperAdapter.estimateResources({ files: [{ role: 'weights', size: fs.statSync(file).size }] }, hw, {}),
          'cpu'
        ),
      },
      (rawLogLine) => win?.webContents.send('asr:progress', { modelId, text: rawLogLine })
    );
    return { text };
  } finally {
    activeAsrJobs.delete(modelId);
  }
}

function cancelTranscribe(modelId) {
  const jobId = activeAsrJobs.get(modelId);
  if (!jobId) return { stopped: false };
  return whisperAdapter.cancel(jobId);
}

async function speakText(modelId, text, opts = {}) {
  const voice = findTtsVoiceFile(modelId);
  if (!voice) throw new Error('Voice file missing or incomplete on disk. Remove and re-download this voice.');
  const hw = await detectHardware();
  const jobId = `tts:${modelId}:${Date.now()}`;
  const outputPath = path.join(paths.generated, `${safeName(modelId)}-${Date.now()}.wav`);

  activeTtsJobs.set(modelId, jobId);
  try {
    const { outputPath: written } = await piperAdapter.speak(
      jobId,
      {
        modelPath: voice.path,
        modelId,
        hfRelativePath: voice.hfRelativePath,
        text,
        outputPath,
        hwEstimate: asHwEstimate(
          piperAdapter.estimateResources({ files: [{ role: 'weights', size: fs.statSync(voice.path).size }] }, hw, {}),
          'cpu'
        ),
      },
      (rawLogLine) => win?.webContents.send('tts:progress', { modelId, text: rawLogLine })
    );
    return { outputPath: written };
  } finally {
    activeTtsJobs.delete(modelId);
  }
}

function cancelSpeak(modelId) {
  const jobId = activeTtsJobs.get(modelId);
  if (!jobId) return { stopped: false };
  return piperAdapter.cancel(jobId);
}

// ─── embeddings (llama adapter's own createEmbeddingContext) ──────────────────
// Same GGUF format/download path as chat (findGgufFile is already correct
// for this) — the only new thing is the vector-out job type, not a new
// runtime or download flow.
async function embedText(modelId, text) {
  const file = findGgufFile(modelId);
  if (!file) throw new Error('GGUF file(s) missing or incomplete on disk. Remove and re-download this model.');
  const { vector } = await llamaAdapter.embed(modelId, file, text);
  return { vector };
}

// ─── reranking (llama adapter's own createRankingContext) ─────────────────────
async function rerankDocuments(modelId, query, documents) {
  const file = findGgufFile(modelId);
  if (!file) throw new Error('GGUF file(s) missing or incomplete on disk. Remove and re-download this model.');
  if (!Array.isArray(documents) || !documents.length) throw new Error('documents array required');
  const { ranked } = await llamaAdapter.rerank(modelId, file, query, documents);
  return { ranked };
}

async function getCachedAnalyses() {
  const files = await fsp.readdir(paths.meta).catch(() => []);
  const out = [];
  for (const f of files) {
    if (!f.endsWith('.json')) continue;
    const data = await readJson(path.join(paths.meta, f));
    if (!data?.weightMeta?.modelId) continue;
    out.push({
      modelId: data.weightMeta.modelId,
      lastAnalysis: data.lastAnalysis || null,
      format: data.weightMeta.format,
      totalWeightGb: data.weightMeta.totalWeightGb,
      offline: true,
    });
  }
  return out;
}

// ─── window / ipc ────────────────────────────────────────────────────────────

function showWin() {
  if (!win || win.isDestroyed()) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
  win.moveTop();
  win.setAlwaysOnTop(true);
  setTimeout(() => {
    if (win && !win.isDestroyed()) win.setAlwaysOnTop(false);
  }, 2000);
}

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: 'Can You Run AI',
    backgroundColor: '#121416',
    center: true,
    autoHideMenuBar: true,
    show: true, // show immediately so user always sees something
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      autoplayPolicy: 'no-user-gesture-required',
    },
  });
  const indexHtml = path.join(__dirname, '..', 'src', 'index.html');
  win.loadFile(indexHtml).catch((err) => {
    console.error('loadFile failed', err);
    win.loadURL(
      'data:text/html,<h1 style="font-family:sans-serif;padding:40px">Failed to load UI</h1><pre>' +
        String(err) +
        '</pre>'
    );
  });
  win.once('ready-to-show', showWin);
  win.webContents.on('did-finish-load', showWin);
  win.webContents.on('did-fail-load', (_e, code, desc) => {
    console.error('did-fail-load', code, desc);
  });
  win.webContents.on('render-process-gone', (_e, details) => {
    console.error('render-process-gone', details);
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  // Fallback focus if ready-to-show is flaky
  setTimeout(showWin, 500);
}

const SELFTEST = process.argv.includes('--selftest');

async function runSelftest() {
  const results = [];
  const lines = [];
  const log = (msg) => {
    lines.push(msg);
    try {
      console.log(msg);
    } catch {
      /* ignore */
    }
  };
  const step = async (name, fn) => {
    try {
      const data = await fn();
      results.push({ name, ok: true, summary: summarize(data) });
      log(`[selftest] OK  ${name} ${JSON.stringify(summarize(data))}`);
      return data;
    } catch (e) {
      results.push({ name, ok: false, error: String(e.message || e) });
      log(`[selftest] FAIL ${name}: ${e.message || e}`);
      return null;
    }
  };
  const summarize = (data) => {
    if (!data || typeof data !== 'object') return data;
    if (data.cpu?.brand) {
      return {
        cpu: data.cpu.brand,
        ramGb: data.memory?.totalGb,
        vramGb: data.gpu?.totalVramGb,
      };
    }
    if (data.answerPreview) return { modelId: data.modelId, answerPreview: data.answerPreview, tokensPerSecond: data.tokensPerSecond };
    if (data.picks) return { pickCount: data.picks.length, offline: data.offline };
    if (data.tags) return { tagCount: data.tags.length, offline: data.offline };
    if (data.models) return { modelCount: data.models.length, offline: data.offline };
    if (data.match) {
      return {
        modelId: data.modelId,
        overall: data.match.overall,
        loadGb: data.memory?.primaryLoadGb,
      };
    }
    if (Array.isArray(data)) return { count: data.length };
    if (data.online != null) return data;
    if (data.ok != null) return data;
    return { keys: Object.keys(data).slice(0, 8) };
  };

  initPaths();
  log(`[selftest] userData=${paths.root}`);

  const hw = await step('hardware', () => detectHardware());
  await step('online', async () => ({ online: await online() }));
  const tags = await step('tags', () => getTags());
  const list = await step('listModels', () =>
    listModels({ pipelineTag: 'text-generation', sort: 'downloads', limit: 5, page: 0 })
  );
  const modelId =
    list?.models?.find((m) => !m.gated)?.id || list?.models?.[0]?.id || 'openai-community/gpt2';
  const analysis = await step('analyze', () => analyze(modelId));
  // Real regression check, not a synthetic one: caught a genuine bug live
  // (2026-08-17) where a pipeline-tag heuristic warning meant only for the
  // chat path was still forcing export_only on a real, adapter-supported
  // embedding model, silently defeating Grok's already-shipped Embed UI.
  // canRunHere for a real, known-good embedding repo must be 'yes'.
  await step('embedCanRun', async () => {
    const r = await analyze('nomic-ai/nomic-embed-text-v1.5-GGUF');
    const summary = { family: r.ir.family, job: r.ir.job, canRunHere: r.runPlan.canRunHere, runner: r.runPlan.runner, experimentalWarning: r.experimentalWarning };
    if (summary.canRunHere !== 'yes' || summary.runner !== 'llama') {
      throw new Error(`embedding model should be canRunHere:'yes' via llama, got ${JSON.stringify(summary)}`);
    }
    return { online: true, ...summary };
  });
  // Same class of check for rerank — pipeline_tag "text-ranking" isn't in
  // ir.js's family map at all (falls through to 'other'), so this guards
  // the job-override in analyze()/analyzeIds() rather than a warning gate,
  // but the risk (silently regressing to export_only/wrong job) is the same.
  await step('rerankCanRun', async () => {
    const r = await analyze('turingevo/bge-reranker-base-Q4_K_M-GGUF');
    const summary = { family: r.ir.family, job: r.ir.job, canRunHere: r.runPlan.canRunHere, runner: r.runPlan.runner };
    if (summary.job !== 'rerank' || summary.canRunHere !== 'yes' || summary.runner !== 'llama') {
      throw new Error(`reranker model should be job:'rerank' canRunHere:'yes' via llama, got ${JSON.stringify(summary)}`);
    }
    return { online: true, ...summary };
  });
  await step('localList', () => listLocalModels());
  await step('cached', () => getCachedAnalyses());
  await step('recommendations', () => getRecommendations());
  await step('sdBackendPlan', async () => {
    const gpu = { cuda: true, vulkan: true, metal: false, rocm: false, cpu: true };
    const see = pickBackend({ backends: gpu }, 'diffusion');
    if (see !== 'vulkan') throw new Error('See must not plan cuda (sd-cli unpinned), got ' + see);
    const talk = pickBackend({ backends: gpu }, 'llm');
    if (talk !== 'cuda') throw new Error('Talk should still prefer cuda, got ' + talk);
    const hear = pickBackend({ backends: gpu }, 'asr');
    if (hear !== 'cpu') throw new Error('Hear is CPU-only, got ' + hear);
    return { see, talk, hear };
  });

  // Real end-to-end in-app inference check with a tiny public GGUF model.
  const CHAT_TEST_MODEL = 'unsloth/SmolLM2-135M-Instruct-GGUF';
  await step('chat', async () => {
    await downloadModel(CHAT_TEST_MODEL);
    await loadChatModel(CHAT_TEST_MODEL);
    try {
      const { answer, tokensPerSecond } = await sendChatMessage(CHAT_TEST_MODEL, 'Reply with the single word: OK');
      if (!answer || !answer.trim()) throw new Error('empty response from model');
      return { modelId: CHAT_TEST_MODEL, answerPreview: answer.slice(0, 60), tokensPerSecond };
    } finally {
      await unloadChat();
    }
  });

  // The whole point of moving llama.cpp out-of-process (mixedplan.md §11
  // Phase 1): killing that worker must not take the app down, and the app
  // must self-heal on the next load — not require a restart.
  await step('crashIsolation', async () => {
    await loadChatModel(CHAT_TEST_MODEL);
    const host = require('./engine/host');
    const pidBefore = host.health('llama').pid;
    host.kill('llama');
    await new Promise((r) => setTimeout(r, 300));
    const afterKillHealth = host.health('llama');
    if (afterKillHealth.running) throw new Error('worker should be reported dead right after kill()');

    // This process (Electron's main process, running this very selftest) is
    // still alive to make this next call at all — that IS the isolation
    // proof. Now prove self-heal: a fresh load must transparently respawn.
    activeChatModelId = null; // worker's in-memory chat state died with the process
    const reload = await loadChatModel(CHAT_TEST_MODEL);
    const pidAfter = host.health('llama').pid;
    const { answer } = await sendChatMessage(CHAT_TEST_MODEL, 'Reply with the single word: OK');
    await unloadChat();
    return {
      pidBefore,
      pidAfter,
      respawnedNewProcess: pidAfter != null && pidAfter !== pidBefore,
      reloadedOk: !!reload.ready,
      generatedAfterRespawnOk: !!answer,
    };
  });

  await step('newFeatures', async () => {
    const results = {};

    // Phase 2: hardware.backends must be real detection, not a placeholder —
    // on this machine (no discrete NVIDIA/AMD GPU) it should show vulkan
    // available via the integrated GPU and cuda/metal false, not everything
    // true or everything false.
    const backends = (await detectHardware()).backends;
    results.backendsShapeOk =
      backends && typeof backends.cpu === 'boolean' && typeof backends.vulkan === 'boolean' &&
      typeof backends.cuda === 'boolean' && typeof backends.metal === 'boolean' && backends.rocm === false;
    if (!results.backendsShapeOk) throw new Error(`hardware.backends malformed: ${JSON.stringify(backends)}`);
    results.backendsNotAllFalseOk = backends.cpu === true; // cpu must always be true
    const ei = await llamaAdapter.ensureInstalled('cuda');
    results.ensureInstalledRejectsUnavailableOk = backends.cuda ? ei.ok === true : ei.ok === false;

    // Context-length memory estimate is present on a real analyze() call.
    const analysis = await analyze(CHAT_TEST_MODEL);
    results.contextEstimates = analysis.memory.contextEstimates;
    if (!Array.isArray(results.contextEstimates) || !results.contextEstimates.length) {
      throw new Error('contextEstimates missing from memory estimate');
    }

    // Advanced load options (system prompt / context size / gpu layers) shouldn't break loading.
    await loadChatModel(CHAT_TEST_MODEL, { systemPrompt: 'You are terse.', contextSize: 512, gpuLayers: 0 });
    const { answer } = await sendChatMessage(CHAT_TEST_MODEL, 'Say hi', { temperature: 0.9 });
    if (!answer) throw new Error('advanced-options generation returned nothing');
    results.advancedOptionsOk = true;

    // JSON-schema grammar mode (node-llama-cpp's createGrammarForJsonSchema)
    // must produce output that actually parses as JSON matching the shape —
    // not just "didn't crash". A 135M model has no business spontaneously
    // producing clean JSON on its own, so this only passes if the grammar
    // constraint is doing real work.
    const jsonSchema = { type: 'object', properties: { word: { type: 'string' }, count: { type: 'number' } }, required: ['word', 'count'] };
    const { answer: jsonAnswer } = await sendChatMessage(CHAT_TEST_MODEL, 'Give me any word and a count of your choice.', { jsonSchema });
    let parsedJson = null;
    try {
      parsedJson = JSON.parse(jsonAnswer);
    } catch {
      /* leave null, checked below */
    }
    results.grammarModeOk = !!parsedJson && typeof parsedJson.word === 'string' && typeof parsedJson.count === 'number';
    if (!results.grammarModeOk) throw new Error(`grammar mode did not produce schema-conformant JSON: ${jsonAnswer}`);

    // stopGeneration must be a safe no-op when nothing is generating.
    const stopResult = stopGeneration(CHAT_TEST_MODEL);
    results.stopNoopOk = stopResult.stopped === false;

    await unloadChat();

    // Chat history round-trips through disk (this is what survives an app restart).
    const sampleLog = [{ role: 'user', text: 'ping' }, { role: 'assistant', text: 'pong' }];
    await saveChatHistory(CHAT_TEST_MODEL, sampleLog);
    const loadedLog = await getChatHistory(CHAT_TEST_MODEL);
    results.historyRoundTripOk = JSON.stringify(loadedLog) === JSON.stringify(sampleLog);

    // Real-world regression check: a known custom-fork model must get flagged
    // before download, not discovered only after a failed load.
    const riskyAnalysis = await analyze('prism-ml/Ternary-Bonsai-27B-gguf');
    results.experimentalWarningOk = !!riskyAnalysis.experimentalWarning;

    // README fetch for the "why won't this load" moment returns real content.
    const readme = await getReadme('prism-ml/Ternary-Bonsai-27B-gguf');
    results.readmeOk = !!(readme.snippet && readme.snippet.length > 20);

    // Real GGUF header engine: ground-truth architecture + exact param count
    // for a normal model, not a filename/tag guess.
    const realMetaAnalysis = analysis; // already computed above for CHAT_TEST_MODEL
    results.realMetaOk = !!(realMetaAnalysis.realMeta?.ok && realMetaAnalysis.realMeta.architecture);
    results.exactParamsOk = realMetaAnalysis.weightMeta.parameters?.exactBillions != null;
    results.kvCacheInVerdictOk = (realMetaAnalysis.match.modes || []).some((m) => m.kvCacheGb > 0);

    // The imatrix/mmproj false-positive bugs from earlier this session: real
    // header parsing must catch these structurally (no architecture, or a
    // known-non-text-gen architecture), not just via filename heuristics.
    const imatrixHeader = await fetchGgufHeader(
      'https://huggingface.co/AtomicChat/Ling-3.0-flash-GGUF/resolve/main/imatrix/Ling-3.0-flash.imatrix.gguf'
    );
    const imatrixClassification = imatrixHeader.ok ? classifyGgufHeader(imatrixHeader.metadata) : null;
    results.imatrixCaughtOk = imatrixHeader.ok ? imatrixClassification.runnable !== true : true;

    const mmprojHeader = await fetchGgufHeader(
      'https://huggingface.co/saphvis/LLaVA_MORE-llama_3_1-8B-finetuning-FP16-mmproj-GGUF/resolve/main/mmproj-model-f16.gguf'
    );
    const mmprojClassification = mmprojHeader.ok ? classifyGgufHeader(mmprojHeader.metadata) : null;
    results.mmprojCaughtOk = mmprojHeader.ok ? mmprojClassification.runnable !== true : true;

    if (!results.advancedOptionsOk || !results.stopNoopOk || !results.historyRoundTripOk ||
        !results.experimentalWarningOk || !results.readmeOk || !results.realMetaOk ||
        !results.exactParamsOk || !results.kvCacheInVerdictOk || !results.imatrixCaughtOk ||
        !results.mmprojCaughtOk || !results.backendsShapeOk || !results.backendsNotAllFalseOk ||
        !results.ensureInstalledRejectsUnavailableOk || !results.grammarModeOk) {
      throw new Error(`checks failed: ${JSON.stringify(results)}`);
    }
    return results;
  });

  await step('uiLoad', async () => {
    return new Promise((resolve, reject) => {
      const testWin = new BrowserWindow({
        show: false,
        webPreferences: {
          preload: path.join(__dirname, 'preload.js'),
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: false,
        },
      });
      const t = setTimeout(() => {
        try {
          testWin.destroy();
        } catch {
          /* ignore */
        }
        reject(new Error('UI load timeout'));
      }, 300000);
      testWin.webContents.on('did-fail-load', (_e, code, desc) => {
        clearTimeout(t);
        try {
          testWin.destroy();
        } catch {
          /* ignore */
        }
        reject(new Error(`did-fail-load ${code} ${desc}`));
      });
      testWin.loadFile(path.join(__dirname, '..', 'src', 'index.html'), { hash: 'selftest' });
      testWin.webContents.on('did-finish-load', async () => {
        try {
          const probe = await testWin.webContents.executeJavaScript(`
            (async () => {
              if (!window.cyrai) return { ok: false, error: 'window.cyrai missing (preload broken)' };

              // Structural check: the new IA markup exists, independent of any JS timing.
              const tabViews = Array.from(document.querySelectorAll('.tab')).map((t) => t.getAttribute('data-view')).sort();
              const structureOk =
                JSON.stringify(tabViews) === JSON.stringify(['explore', 'home', 'library']) &&
                !!document.getElementById('view-home') &&
                !!document.getElementById('view-explore') &&
                !!document.getElementById('view-library') &&
                !!document.getElementById('hero-number') &&
                !!document.getElementById('chat-panel') &&
                !!document.getElementById('btn-chat-stop') &&
                !!document.getElementById('adv-system-prompt') &&
                !!document.getElementById('adv-context-size') &&
                !!document.getElementById('adv-gpu-layers') &&
                !!document.getElementById('adv-temperature') &&
                !!document.getElementById('compare-panel') &&
                !!document.getElementById('btn-compare-clear') &&
                !!document.getElementById('onboarding') &&
                !!document.getElementById('runnable-only') &&
                !!document.getElementById('gate') &&
                !!document.getElementById('btn-gate-enter') &&
                !!document.querySelector('#sort option[value="fit"]');

              const hw = await window.cyrai.detectHardware();
              const net = await window.cyrai.netStatus();
              const tags = await window.cyrai.getTags();
              const models = await window.cyrai.listModels({ limit: 3, sort: 'downloads', page: 0 });
              const reco = await window.cyrai.recommendations();
              const recoOk = !!(reco.ok && Array.isArray(reco.data.picks));

              const bridgeHasChat = ['loadChatModel', 'sendChatMessage', 'unloadChatModel', 'onChatChunk']
                .every((k) => typeof window.cyrai[k] === 'function');
              let chatBridge = null;
              if (bridgeHasChat) {
                const load = await window.cyrai.loadChatModel('${CHAT_TEST_MODEL}');
                const send = load.ok
                  ? await window.cyrai.sendChatMessage('${CHAT_TEST_MODEL}', 'Reply with the single word: OK')
                  : null;
                await window.cyrai.unloadChatModel();
                chatBridge = {
                  ok: !!(load.ok && send && send.ok && send.data.answer),
                  loadError: load.ok ? null : load.error,
                  sendError: send && !send.ok ? send.error : null,
                  answerPreview: send && send.ok ? String(send.data.answer).slice(0, 60) : null,
                };
              }

              // Behavioral check: the app's own init() actually ran and populated the hero,
              // not just that the IPC bridge works in isolation.
              const deadline = Date.now() + 45000;
              let heroText = document.getElementById('hero-sub').textContent;
              while (heroText === 'Detecting hardware…' && Date.now() < deadline) {
                await new Promise((r) => setTimeout(r, 300));
                heroText = document.getElementById('hero-sub').textContent;
              }
              const heroPopulated = document.getElementById('hero-number').textContent !== '—';

              // Click-driven UI test: real event handlers, not the IPC bridge directly.
              // Catches id/wiring typos that a pure-bridge check would miss.
              const waitFor = async (fn, ms) => {
                const deadline = Date.now() + ms;
                let v = fn();
                while (!v && Date.now() < deadline) {
                  await new Promise((r) => setTimeout(r, 250));
                  v = fn();
                }
                return v;
              };

              document.querySelector('.tab[data-view="explore"]').click();
              const exploreShown = !document.getElementById('view-explore').hidden;
              document.getElementById('q').value = 'smollm2';
              document.getElementById('btn-go').click();
              const modelsRendered = await waitFor(() => document.querySelectorAll('#models .item').length > 0, 45000);
              const firstItem = document.querySelector('#models .item');
              if (firstItem) firstItem.click();
              const detailRendered = await waitFor(() => !!document.querySelector('#detail .summary'), 15000);

              // Compare button + Advanced panel: pure DOM/state, no extra network calls.
              // renderDetail() replaces #detail's innerHTML on click, so re-query after —
              // the pre-click element reference goes stale/detached.
              let compareOk = false;
              if (document.getElementById('btn-compare')) {
                document.getElementById('btn-compare').click();
                const btnAfter = document.getElementById('btn-compare');
                compareOk = !!btnAfter && btnAfter.textContent === 'Remove from compare' &&
                  document.getElementById('compare-panel').hidden === true;
              }
              const advDetails = document.querySelector('.chat-advanced');
              let advancedPanelOk = false;
              if (advDetails) {
                advDetails.open = true;
                advancedPanelOk = advDetails.open === true && !!document.getElementById('adv-temperature');
              }

              // Regression check for the exact real-world case that motivated this
              // feature: a GGUF model flagged experimental (mtp tag) should show the
              // "can't run here" callout with a working, lazily-loaded setup-notes disclosure.
              document.getElementById('q').value = 'Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF';
              document.getElementById('btn-go').click();
              await waitFor(() => document.querySelectorAll('#models .item').length > 0, 30000);
              const riskyItem = document.querySelector('#models .item');
              let setupNotesOk = false;
              if (riskyItem) {
                riskyItem.click();
                await waitFor(() => !!document.querySelector('#detail .summary'), 15000);
                const setupNotesEl = document.getElementById('setup-notes');
                if (setupNotesEl) {
                  setupNotesEl.open = true; // <details> fires a real 'toggle' event on this automatically
                  const settled = await waitFor(() => {
                    const t = document.getElementById('setup-notes-body')?.textContent || '';
                    return t.length > 0 && t !== 'Click to load…' && t !== 'Loading…';
                  }, 15000);
                  setupNotesOk = !!settled && !document.getElementById('setup-notes-body').textContent.includes('<div');
                }
              }

              // Regression check: ticking fits-only then changing category must
              // not get stuck saying "Checking..." forever (it used to — loadModels()
              // cleared analyses on a new query but nothing re-triggered the check).
              const modelsEl = document.getElementById('models');
              const checkSettled = () => {
                const t = modelsEl.textContent;
                return modelsEl.querySelectorAll('.item').length > 0 ||
                  t.includes('None of these models fit') || t.includes('No models');
              };
              const fitsOnlyBox = document.getElementById('fits-only');
              fitsOnlyBox.checked = true;
              fitsOnlyBox.dispatchEvent(new Event('change'));
              const firstCheckDone = await waitFor(checkSettled, 25000);
              let categorySwitchOk = null;
              const catSelect = document.getElementById('cat');
              if (catSelect.options.length > 1) {
                catSelect.selectedIndex = catSelect.selectedIndex === 0 ? 1 : 0;
                catSelect.dispatchEvent(new Event('change'));
                // A category switch re-checks a full ~20-model page (vs. whatever a
                // narrow search returned), so give it real room before calling it stuck.
                categorySwitchOk = await waitFor(checkSettled, 60000);
              }
              fitsOnlyBox.checked = false;
              fitsOnlyBox.dispatchEvent(new Event('change'));
              const fitsOnlyOk = !!firstCheckDone && (catSelect.options.length <= 1 || !!categorySwitchOk);

              document.querySelector('.tab[data-view="library"]').click();
              const libraryShown = !document.getElementById('view-library').hidden;
              const libItem = document.querySelector('#local-list .item[data-id="${CHAT_TEST_MODEL}"]');
              let chatUiOk = false;
              let chatUiAnswer = null;
              let visibilityOk = null;
              if (libItem) {
                libItem.click();
                const runnableShown = await waitFor(() => document.getElementById('chat-panel').hidden === false && document.getElementById('chat-runnable').hidden === false, 5000);
                // The [hidden] attribute being set is not proof anything is actually
                // invisible — components with their own \`display\` rule can silently
                // beat the UA [hidden]{display:none} rule. Check real computed style.
                const dispNone = (id) => getComputedStyle(document.getElementById(id)).display === 'none';
                const dispShown = (id) => getComputedStyle(document.getElementById(id)).display !== 'none';
                visibilityOk = dispNone('chat-empty') && dispNone('chat-noninfer') && dispShown('chat-panel') && dispShown('chat-runnable');
                if (runnableShown) {
                  document.getElementById('chat-input').value = 'Reply with the single word: OK';
                  document.getElementById('btn-chat-send').click();
                  await waitFor(() => document.querySelectorAll('#chat-log .msg.assistant').length > 0, 5000);
                  const answered = await waitFor(() => {
                    const bubbles = document.querySelectorAll('#chat-log .msg.assistant');
                    const last = bubbles[bubbles.length - 1];
                    return last && last.textContent.trim().length > 0;
                  }, 40000);
                  chatUiOk = !!answered;
                  const bubbles = document.querySelectorAll('#chat-log .msg.assistant');
                  chatUiAnswer = bubbles.length ? bubbles[bubbles.length - 1].textContent.slice(0, 60) : null;
                }
              }

              const uiInteraction = {
                ok: exploreShown && !!modelsRendered && !!detailRendered && compareOk && advancedPanelOk && setupNotesOk &&
                  fitsOnlyOk && libraryShown && !!libItem && !!visibilityOk && chatUiOk,
                exploreShown, modelsRendered: !!modelsRendered, detailRendered: !!detailRendered,
                compareOk, advancedPanelOk, setupNotesOk,
                fitsOnlyOk, firstCheckDone: !!firstCheckDone, categorySwitchOk,
                libraryShown, libItemFound: !!libItem, visibilityOk, chatUiOk, chatUiAnswer,
              };

              return {
                ok: !!(
                  hw.ok && net.ok && tags.ok && models.ok && recoOk &&
                  bridgeHasChat && chatBridge && chatBridge.ok &&
                  structureOk && heroPopulated && uiInteraction.ok
                ),
                hw: hw.ok ? hw.data.cpu.brand : hw.error,
                online: net.ok ? net.data.online : net.error,
                tagCount: tags.ok ? (tags.data.tags || []).length : tags.error,
                modelCount: models.ok ? (models.data.models || []).length : models.error,
                modelError: models.ok ? null : models.error,
                recoOk,
                recoCount: reco.ok ? reco.data.picks.length : null,
                recoError: reco.ok ? null : reco.error,
                bridgeHasChat,
                chatBridge,
                structureOk,
                heroPopulated,
                heroText,
                uiInteraction,
              };
            })()
          `);
          clearTimeout(t);
          try {
            testWin.destroy();
          } catch {
            /* ignore */
          }
          if (!probe.ok) throw new Error(JSON.stringify(probe));
          resolve(probe);
        } catch (err) {
          clearTimeout(t);
          try {
            testWin.destroy();
          } catch {
            /* ignore */
          }
          reject(err);
        }
      });
    });
  });

  const failed = results.filter((r) => !r.ok);
  const summary = {
    passed: results.filter((r) => r.ok).length,
    failed: failed.length,
    hardware: hw?.cpu?.brand || null,
    vramGb: hw?.gpu?.totalVramGb ?? null,
    tags: tags?.tags?.length ?? null,
    sampleModel: modelId,
    overall: analysis?.match?.overall ?? null,
    primaryLoadGb: analysis?.memory?.primaryLoadGb ?? null,
    failures: failed,
    at: new Date().toISOString(),
  };
  log(`[selftest] summary ${JSON.stringify(summary)}`);
  log(failed.length ? '[selftest] ALL FAILED' : '[selftest] ALL PASSED');

  // Always write report where we can read it (stdout is flaky on Windows Electron)
  const reportPath = path.join(process.cwd(), 'selftest-report.json');
  const textPath = path.join(process.cwd(), 'selftest-report.txt');
  try {
    fs.writeFileSync(reportPath, JSON.stringify({ summary, results, lines }, null, 2));
    fs.writeFileSync(textPath, lines.join('\n') + '\n');
    log(`[selftest] wrote ${reportPath}`);
  } catch (e) {
    log(`[selftest] could not write report: ${e.message}`);
  }

  app.exit(failed.length ? 1 : 0);
}

app.whenReady().then(() => {
  if (SELFTEST) {
    runSelftest().catch((e) => {
      console.error('[selftest] crash', e);
      app.exit(1);
    });
    return;
  }
  initPaths();
  createWindow();
  app.on('activate', () => {
    if (!BrowserWindow.getAllWindows().length) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (SELFTEST) return;
  if (process.platform !== 'darwin') app.quit();
});

const ok = (data) => ({ ok: true, data });
const fail = (e) => ({ ok: false, error: String(e.message || e) });

ipcMain.handle('hw:detect', async () => {
  try {
    return ok(await detectHardware());
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('net:status', async () => {
  const isOn = await online();
  return ok({ online: isOn, modelsDir: paths.models, cacheDir: paths.cache });
});

ipcMain.handle('hf:tags', async () => {
  try {
    return ok(await getTags());
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('hf:list', async (_e, opts) => {
  try {
    return ok(await listModels(opts || {}));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('hf:analyze', async (_e, modelId) => {
  try {
    if (!modelId) throw new Error('modelId required');
    return ok(await analyze(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('hf:readme', async (_e, modelId) => {
  try {
    if (!modelId) throw new Error('modelId required');
    return ok(await getReadme(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('hf:analyzeBatch', async (_e, ids) => {
  try {
    const hw = await detectHardware();
    const results = await analyzeIds(hw, (ids || []).slice(0, 30));
    return ok({ results, budgets: hw.budgets });
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('hf:recommendations', async (_e, force) => {
  try {
    return ok(await getRecommendations(!!force));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:list', async () => {
  try {
    return ok(await listLocalModels());
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:cached', async () => {
  try {
    return ok(await getCachedAnalyses());
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:download', async (_e, modelId) => {
  try {
    return ok(await downloadModel(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:cancelDownload', async (_e, modelId) => {
  try {
    return ok(cancelDownload(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:delete', async (_e, modelId) => {
  try {
    return ok(await deleteLocalModel(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:getChatHistory', async (_e, modelId) => {
  try {
    return ok(await getChatHistory(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:saveChatHistory', async (_e, modelId, log) => {
  try {
    return ok(await saveChatHistory(modelId, log));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('local:openDir', async (_e, modelId) => {
  try {
    const dir = modelId ? modelDir(modelId) : paths.models;
    await fsp.mkdir(dir, { recursive: true });
    shell.openPath(dir);
    return ok(true);
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('shell:open', async (_e, url) => {
  if (typeof url === 'string' && /^https?:\/\//.test(url)) {
    await shell.openExternal(url);
    return ok(true);
  }
  return fail(new Error('Invalid URL'));
});

const DATA_URL_MIME = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.wav': 'audio/wav' };
// Renderer-requested file reads must stay scoped to this app's own output
// dirs — without this check, a compromised renderer could ask to read any
// file on disk (SSH keys, etc.) through an otherwise-innocuous "show me
// this image" channel.
ipcMain.handle('local:readDataUrl', async (_e, filePath) => {
  try {
    const resolved = path.resolve(String(filePath || ''));
    const allowedRoots = [paths.generated, paths.models].map((p) => path.resolve(p) + path.sep);
    if (!allowedRoots.some((root) => resolved.startsWith(root))) {
      throw new Error('File is outside the app-managed output directories.');
    }
    const ext = path.extname(resolved).toLowerCase();
    const mime = DATA_URL_MIME[ext];
    if (!mime) throw new Error(`Unsupported file type: ${ext}`);
    const buf = await fsp.readFile(resolved);
    return ok(`data:${mime};base64,${buf.toString('base64')}`);
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('chat:load', async (_e, modelId, opts) => {
  try {
    if (!modelId) throw new Error('modelId required');
    return ok(await loadChatModel(modelId, opts || {}));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('chat:send', async (_e, { modelId, message, opts } = {}) => {
  try {
    if (!modelId || !message) throw new Error('modelId and message required');
    return ok(await sendChatMessage(modelId, message, opts || {}));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('chat:stop', async (_e, modelId) => {
  try {
    return ok(stopGeneration(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('chat:unload', async () => {
  try {
    await unloadChat();
    return ok(true);
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('sd:generate', async (_e, { modelId, opts } = {}) => {
  try {
    if (!modelId) throw new Error('modelId required');
    if (!opts?.prompt) throw new Error('prompt required');
    return ok(await generateImage(modelId, opts));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('sd:cancel', async (_e, modelId) => {
  try {
    return ok(cancelGeneration(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('video:generate', async (_e, { modelId, opts } = {}) => {
  try {
    if (!modelId) throw new Error('modelId required');
    if (!opts?.prompt) throw new Error('prompt required');
    return ok(await generateVideo(modelId, opts));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('video:cancel', async (_e, modelId) => {
  try {
    return ok(cancelVideo(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('music:generate', async (_e, { modelId, opts } = {}) => {
  try {
    if (!modelId) throw new Error('modelId required');
    if (!opts?.prompt) throw new Error('prompt required');
    return ok(await generateMusic(modelId, opts));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('music:cancel', async (_e, modelId) => {
  try {
    return ok(cancelMusic(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('asr:transcribe', async (_e, { modelId, audioPath, opts } = {}) => {
  try {
    if (!modelId || !audioPath) throw new Error('modelId and audioPath required');
    return ok(await transcribeAudio(modelId, audioPath, opts || {}));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('asr:cancel', async (_e, modelId) => {
  try {
    return ok(cancelTranscribe(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('tts:speak', async (_e, { modelId, text, opts } = {}) => {
  try {
    if (!modelId || !text) throw new Error('modelId and text required');
    return ok(await speakText(modelId, text, opts || {}));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('tts:cancel', async (_e, modelId) => {
  try {
    return ok(cancelSpeak(modelId));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('dialog:pickAudio', async () => {
  try {
    const res = await dialog.showOpenDialog(win, {
      properties: ['openFile'],
      filters: [{ name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg'] }],
    });
    if (res.canceled || !res.filePaths?.length) return ok(null);
    return ok(res.filePaths[0]);
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('dialog:pickImage', async () => {
  try {
    const res = await dialog.showOpenDialog(win, {
      properties: ['openFile'],
      filters: [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
    });
    if (res.canceled || !res.filePaths?.length) return ok(null);
    return ok(res.filePaths[0]);
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('embed:text', async (_e, { modelId, text } = {}) => {
  try {
    if (!modelId || !text) throw new Error('modelId and text required');
    return ok(await embedText(modelId, text));
  } catch (e) {
    return fail(e);
  }
});

ipcMain.handle('rerank:documents', async (_e, { modelId, query, documents } = {}) => {
  try {
    if (!modelId || !query) throw new Error('modelId and query required');
    return ok(await rerankDocuments(modelId, query, documents));
  } catch (e) {
    return fail(e);
  }
});

app.on('before-quit', () => {
  unloadChat().catch(() => {});
});
