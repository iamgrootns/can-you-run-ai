/* Can You Run AI — renderer */
(function () {
  const $ = (id) => document.getElementById(id);
  const rpc = () => {
    if (!window.cyrai) throw new Error('Bridge missing — preload failed');
    return window.cyrai;
  };

  const state = {
    view: 'home',
    online: true,
    hw: null,
    // explore
    page: 0,
    limit: 20,
    models: [],
    analyses: new Map(),
    selected: null,
    downloadingId: null,
    compareIds: [],
    compareData: new Map(),
    hearAudioPath: null,
    seeInitPath: null,
    seeMaskPath: null,
    // library / chat
    localModels: [],
    librarySelected: null,
    chatLogs: new Map(),
    chatHistoryLoaded: new Set(),
    chatLoadedModelId: null,
    chatLoadedSettings: null,
    chatBusy: false,
    // lazy-load guards
    exploreLoaded: false,
    recos: [],
  };

  const FIT_RANK = { excellent: 4, good: 3, tight: 2, unlikely: 1, unknown: 0 };

  function esc(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function friendlyError(msg) {
    const s = String(msg || '');
    if (/HTTP 429/.test(s)) return 'Hugging Face is rate-limiting requests right now — wait a moment and try again.';
    if (/HTTP 5\d\d/.test(s)) return "Hugging Face is temporarily unavailable — it's not you, try again shortly.";
    if (/ENOTFOUND|ECONNREFUSED|fetch failed|network/i.test(s)) return "Can't reach the network right now.";
    if (/No verified sd-cli|STATUS_DLL|0xC0000135|3221225781|DLL_NOT_FOUND/i.test(s)) {
      return 'Image runner could not start on GPU. We retry on CPU automatically — first image can take several minutes.';
    }
    if (/sd-cli exited/i.test(s)) return 'Image generation failed. Stay at 512×512, or try another checkpoint.';
    if (/GPU is in use/i.test(s)) return 'GPU is busy with another job. Unload chat first, or wait.';
    return s;
  }

  function gb(n) {
    if (n == null || Number.isNaN(+n)) return '—';
    return (+n).toFixed(+n >= 10 ? 1 : 2) + ' GB';
  }

  const dlSpeedTracker = { time: 0, bytes: 0, file: null, lastText: '' };

  function formatEta(sec) {
    if (sec == null || !isFinite(sec)) return '';
    if (sec < 60) return Math.round(sec) + 's left';
    const m = Math.round(sec / 60);
    return m < 60 ? m + 'm left' : Math.round(m / 60) + 'h left';
  }

  // Real MB/s measured from consecutive progress events, not a synthetic estimate.
  function trackDownloadSpeed(p) {
    if (p.received == null) return '';
    const now = Date.now();
    if (dlSpeedTracker.file !== p.file) {
      dlSpeedTracker.file = p.file;
      dlSpeedTracker.time = now;
      dlSpeedTracker.bytes = p.received;
      dlSpeedTracker.lastText = '';
      return '';
    }
    const dt = (now - dlSpeedTracker.time) / 1000;
    if (dt < 0.4) return dlSpeedTracker.lastText; // avoid noisy sub-400ms samples
    const bps = (p.received - dlSpeedTracker.bytes) / dt;
    dlSpeedTracker.time = now;
    dlSpeedTracker.bytes = p.received;
    if (bps <= 0) return dlSpeedTracker.lastText;
    const mbps = (bps / 1024 / 1024).toFixed(1) + ' MB/s';
    const remain = (p.total || 0) - p.received;
    const eta = formatEta(remain / bps);
    dlSpeedTracker.lastText = mbps + (eta ? ' · ' + eta : '');
    return dlSpeedTracker.lastText;
  }

  function estimateMinutes(dlGb) {
    if (dlGb == null) return null;
    const MBPS = 12.5; // ~100 Mbps typical broadband, stated as an estimate
    const mins = Math.max(1, Math.round((dlGb * 1024) / MBPS / 60));
    return mins;
  }

  const BACKEND_PREF = ['cuda', 'metal', 'vulkan', 'rocm', 'cpu'];
  const BACKEND_LABEL = { cuda: 'CUDA', vulkan: 'Vulkan', metal: 'Metal', rocm: 'ROCm', cpu: 'CPU', gpu: 'GPU' };

  function backendLabel(name) {
    return BACKEND_LABEL[name] || String(name || '');
  }

  function pickBackend(hw, family) {
    const b = hw && hw.backends;
    if (!b) return null;
    if (family === 'asr' || family === 'tts') return 'cpu';
    const pref = family === 'diffusion' || family === 't2i' ? ['vulkan', 'cpu'] : BACKEND_PREF;
    for (let i = 0; i < pref.length; i++) {
      if (b[pref[i]]) return pref[i];
    }
    return 'cpu';
  }

  function formatDlSize(bytes) {
    if (bytes == null || !isFinite(+bytes) || bytes < 0) return '';
    return gb(bytes / 1024 / 1024 / 1024);
  }

  const dlResume = { file: null, active: false };
  function resumeNote(p) {
    if (p.done) {
      dlResume.file = null;
      dlResume.active = false;
      return '';
    }
    if (p.file !== dlResume.file) {
      dlResume.file = p.file;
      dlResume.active = !!(p.received > 0 && p.percent > 0 && p.percent < 100);
    }
    return dlResume.active ? 'Resuming · ' : '';
  }

  function backendsHtml(h) {
    const b = h && h.backends;
    if (!b) return '';
    const pref = pickBackend(h);
    const parts = ['cuda', 'vulkan', 'metal', 'rocm', 'cpu']
      .filter(function (k) { return b[k]; })
      .map(function (k) {
        return k === pref
          ? '<em title="Preferred on this machine">' + backendLabel(k) + '</em>'
          : backendLabel(k);
      });
    if (!parts.length) return '';
    return '<span><b>Backend</b> ' + parts.join(' · ') + '</span>';
  }

  function downloadFactsHtml(d) {
    const local = d.local || {};
    const mins = estimateMinutes(d.memory && d.memory.downloadGb);
    const backend = (d.runPlan && d.runPlan.backend) || pickBackend(state.hw, d.ir && d.ir.family);
    const bits = [];
    if (d.memory && d.memory.downloadGb != null) {
      bits.push(local.downloaded ? gb(d.memory.downloadGb) + ' on disk' : gb(d.memory.downloadGb) + ' download');
    }
    if (!local.downloaded && mins) bits.push('~' + mins + ' min (est. 100 Mbps)');
    if (local.partial) bits.push('partial — will resume');
    if (backend && d.runPlan && d.runPlan.canRunHere === 'yes') bits.push('runs on ' + backendLabel(backend));
    else if (backend) bits.push(backendLabel(backend) + ' on this PC');
    if (!bits.length) return '';
    return '<div class="dl-facts">' + bits.map(function (t) { return '<span>' + esc(t) + '</span>'; }).join('') + '</div>';
  }

  function contextEstimatesHtml(contextEstimates) {
    if (!contextEstimates || !contextEstimates.length) return '';
    const parts = contextEstimates
      .map(function (c) { return (c.tokens / 1024) + 'K ctx: +' + gb(c.kvGb); })
      .join(' · ');
    return '<p class="sm-text muted">Longer conversations use more memory (approx.) — ' + esc(parts) + '.</p>';
  }

  function pill(status) {
    const s = status || 'unknown';
    return '<span class="pill ' + esc(s) + '">' + esc(s) + '</span>';
  }

  function toast(msg, ms) {
    const el = $('toast');
    if (!el) return;
    el.textContent = msg || '';
    el.hidden = !msg;
    clearTimeout(toast._t);
    if (msg) toast._t = setTimeout(function () { el.hidden = true; }, ms || 4000);
  }

  /* ───────── nav ───────── */

  function switchView(view) {
    state.view = view;
    ['home', 'explore', 'library'].forEach(function (v) {
      $('view-' + v).hidden = v !== view;
    });
    document.querySelectorAll('.tab').forEach(function (t) {
      t.classList.toggle('active', t.getAttribute('data-view') === view);
    });
    if (view === 'explore' && !state.exploreLoaded) {
      state.exploreLoaded = true;
      loadTags();
      loadModels();
    }
    if (view === 'library') {
      loadLocal();
    }
  }

  /* ───────── net / hardware ───────── */

  async function refreshNet() {
    try {
      const r = await rpc().netStatus();
      state.online = !!(r.ok && r.data && r.data.online);
      const p = $('net-pill');
      if (p) {
        p.textContent = state.online ? 'Online' : 'Offline';
        p.className = 'pill ' + (state.online ? 'on' : 'off');
      }
    } catch (e) {
      const p = $('net-pill');
      if (p) {
        p.textContent = 'Error';
        p.className = 'pill off';
      }
    }
  }

  async function loadHardware() {
    try {
      const r = await rpc().detectHardware();
      if (!r.ok) {
        $('hero-sub').textContent = friendlyError(r.error);
        return;
      }
      const h = r.data;
      state.hw = h;
      const b = h.budgets;
      $('hero-number').textContent = b.approxMaxParamsB != null ? '~' + b.approxMaxParamsB : '—';
      $('hero-sub').textContent =
        b.approxMaxParamsB > 0
          ? 'Comfortably runs Q4 chat models up to about this size. Use Talk / See / Hear / Embed below — we only Run what an adapter can actually load.'
          : "We couldn't size a comfortable model for this hardware — check the specs below.";

      const g = h.gpu && h.gpu.primary;
      const gpuCount = h.gpu.gpuCount || 0;
      const gpuLabel = (g && g.model) || 'none';
      const vram =
        h.gpu.totalVramGb != null
          ? gb(h.gpu.totalVramGb) + (gpuCount > 1 ? ' (' + gpuCount + ' GPUs combined)' : '')
          : h.gpu.unifiedMemory
            ? 'unified'
            : 'n/a';
      $('hero-specs').innerHTML =
        '<span><b>CPU</b> ' + esc(h.cpu.brand) + '</span>' +
        '<span><b>RAM</b> ' + gb(h.memory.totalGb) + '</span>' +
        '<span><b>GPU</b> ' + esc(gpuLabel) + (gpuCount > 1 ? esc(' +' + (gpuCount - 1) + ' more') : '') + '</span>' +
        '<span><b>VRAM</b> ' + esc(vram) + '</span>' +
        '<span><b>Disk free</b> ' + (h.disk.primary ? gb(h.disk.primary.availableGb) : '—') + '</span>' +
        backendsHtml(h);
    } catch (e) {
      $('hero-sub').textContent = friendlyError(e.message || e);
    }
  }

  /* ───────── home / recommendations ───────── */

  function hearKind(d) {
    const rp = d && d.runPlan;
    if (!rp) return null;
    const family = rp.family || (d.ir && d.ir.family);
    if (rp.runner === 'whisper' || rp.job === 'asr' || family === 'asr') return 'asr';
    if (rp.runner === 'piper' || rp.job === 'tts' || family === 'tts') return 'tts';
    return null;
  }

  // Only call embedText when job/family is embed — a chat GGUF will load and then fail.
  function isEmbedModel(d) {
    const rp = d && d.runPlan;
    if (!rp) return false;
    const family = rp.family || (d.ir && d.ir.family);
    return rp.job === 'embed' || family === 'embed';
  }

  // Family stays 'other' — ir.js has no text-ranking map. Route on job only.
  // Never call rerankDocuments on a chat/embed GGUF.
  function isRerankModel(d) {
    const rp = d && d.runPlan;
    if (rp && rp.job === 'rerank') return true;
    return !!(d && d.weightMeta && d.weightMeta.pipelineTag === 'text-ranking');
  }

  function looksLikeReranker(id) {
    return /rerank|text-ranking|cross-?encoder/i.test(String(id || ''));
  }

  function cosineSim(a, b) {
    if (!a || !b || a.length !== b.length || !a.length) return null;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const den = Math.sqrt(na) * Math.sqrt(nb);
    return den ? dot / den : 0;
  }

  function vecPreview(v) {
    const n = v.length;
    const head = [];
    for (let i = 0; i < Math.min(8, n); i++) head.push(Number(v[i]).toFixed(4));
    return n + '-d  ' + head.join(' ') + (n > 8 ? ' …' : '');
  }

  function fileName(p) {
    const s = String(p || '');
    const i = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return i >= 0 ? s.slice(i + 1) : s;
  }

  function recoActionFor(r) {
    if (r.local && r.local.downloaded && r.runPlan && r.runPlan.canRunHere === 'yes') {
      const kind = hearKind(r);
      if (kind === 'asr') return { label: 'Transcribe', kind: 'view' };
      if (kind === 'tts') return { label: 'Speak', kind: 'view' };
      if (r.runPlan.runner === 'sd' || r.runPlan.job === 't2i') return { label: 'Generate', kind: 'view' };
      if (isEmbedModel(r)) return { label: 'Embed', kind: 'view' };
      if (isRerankModel(r)) return { label: 'Rank', kind: 'view' };
      return { label: 'Chat', kind: 'chat' };
    }
    if (r.local && r.local.downloaded && r.format === 'gguf') return { label: 'Chat', kind: 'chat' };
    if (r.local && r.local.downloaded) return { label: 'View', kind: 'view' };
    return { label: 'Download', kind: 'download' };
  }

  function recoCardHtml(r) {
    const mins = estimateMinutes(r.downloadGb);
    const already = r.local && r.local.downloaded;
    const backend = (r.runPlan && r.runPlan.backend) || pickBackend(state.hw, r.runPlan && r.runPlan.family);
    const whyBits = already
      ? [r.summary, backend ? backendLabel(backend) : '']
      : [r.summary, r.downloadGb != null ? gb(r.downloadGb) : '', mins ? '~' + mins + ' min' : '', backend ? backendLabel(backend) : ''];
    const why = whyBits.filter(Boolean).join(' · ');
    const action = recoActionFor(r);
    return (
      '<div class="reco-card" data-id="' + esc(r.modelId) + '">' +
      '<div><div class="name">' + esc(r.modelId) +
      (r.useTag ? '<span class="use-tag">' + esc(r.useTag) + '</span>' : '') + '</div>' +
      '<div class="why">' + esc(why) + '</div></div>' +
      '<div class="meta">' + pill(r.overall) +
      '<button type="button" class="btn ' + (action.kind === 'download' ? 'primary' : 'ghost') +
      '" data-action="' + action.kind + '" data-id="' + esc(r.modelId) + '">' + action.label + '</button>' +
      '</div></div>'
    );
  }

  function wireRecoActions(container) {
    container.querySelectorAll('[data-action]').forEach(function (btn) {
      btn.onclick = function (e) {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        const action = btn.getAttribute('data-action');
        if (action === 'download') homeDownload(id, btn);
        else if (action === 'cancel') rpc().cancelDownload(id);
        else if (action === 'chat') {
          switchView('library');
          selectLibraryModel(id);
        } else {
          switchView('explore');
          selectModel(id);
        }
      };
    });
  }

  async function homeDownload(id, btn) {
    const card = btn.closest('.reco-card');
    const whyEl = card && card.querySelector('.why');
    btn.textContent = 'Cancel';
    btn.classList.remove('primary');
    btn.classList.add('ghost');
    btn.setAttribute('data-action', 'cancel');
    wireRecoActions(card.parentElement);
    toast('Downloading ' + id + '…');
    try {
      const r = await rpc().download(id);
      if (!r.ok) {
        toast(friendlyError(r.error));
        btn.textContent = 'Download';
        btn.classList.add('primary');
        btn.classList.remove('ghost');
        btn.setAttribute('data-action', 'download');
        wireRecoActions(card.parentElement);
        return;
      }
      toast('Saved for offline use');
      try {
        const a = await rpc().analyze(id);
        if (a.ok) {
          const action = recoActionFor({
            modelId: id,
            local: a.data.local,
            runPlan: a.data.runPlan,
            format: a.data.weightMeta && a.data.weightMeta.format,
          });
          if (whyEl) whyEl.textContent = (a.data.runPlan && a.data.runPlan.fitLine) || 'Downloaded — ready.';
          btn.textContent = action.label;
          btn.setAttribute('data-action', action.kind);
          wireRecoActions(card.parentElement);
          loadLocal();
          // First-run: don't leave them on Home after a successful download.
          if (action.kind === 'chat') {
            switchView('library');
            selectLibraryModel(id);
          } else if (action.kind === 'view') {
            switchView('explore');
            selectModel(id);
          }
          return;
        }
      } catch (e2) { /* fall through to Chat */ }
      if (whyEl) whyEl.textContent = 'Downloaded — ready to chat.';
      btn.textContent = 'Chat';
      btn.setAttribute('data-action', 'chat');
      wireRecoActions(card.parentElement);
      loadLocal();
    } catch (e) {
      toast(friendlyError(e.message || e));
      btn.textContent = 'Download';
      btn.classList.add('primary');
      btn.classList.remove('ghost');
      btn.setAttribute('data-action', 'download');
      wireRecoActions(card.parentElement);
    }
  }

  async function loadRecommendations(force) {
    try {
      const r = await rpc().recommendations(!!force);
      if (!r.ok) {
        $('reco-list').innerHTML = '<div class="muted sm-text">' + esc(friendlyError(r.error)) + '</div>';
        return;
      }
      const picks = r.data.picks || [];
      state.recos = picks;
      $('reco-meta').textContent = r.data.offline ? 'offline cache' : 'matched to your hardware';
      renderCapabilityMap();
      if (!picks.length) {
        $('reco-list').innerHTML = '<div class="muted sm-text">No fitting models found right now — try Explore.</div>';
        return;
      }
      $('reco-list').innerHTML = picks.map(recoCardHtml).join('');
      wireRecoActions($('reco-list'));
    } catch (e) {
      $('reco-list').innerHTML = '<div class="muted sm-text">' + esc(friendlyError(e.message || e)) + '</div>';
    }
  }

  function readLastUsed() {
    try {
      const v = JSON.parse(localStorage.getItem('cyrai_last_used') || '[]');
      return Array.isArray(v) ? v : [];
    } catch (e) {
      return [];
    }
  }

  function touchLastUsed(id, job) {
    if (!id) return;
    try {
      const next = [{ id: id, job: job || '', at: Date.now() }]
        .concat(readLastUsed().filter(function (x) { return x.id !== id; }))
        .slice(0, 12);
      localStorage.setItem('cyrai_last_used', JSON.stringify(next));
    } catch (e) { /* best-effort */ }
  }

  function continueActionFor(m) {
    const job = m.lastAnalysis && m.lastAnalysis.job;
    const family = m.lastAnalysis && m.lastAnalysis.family;
    if (job === 'asr' || family === 'asr') return { label: 'Transcribe', kind: 'view' };
    if (job === 'tts' || family === 'tts') return { label: 'Speak', kind: 'view' };
    if (job === 't2i' || family === 'diffusion') return { label: 'Generate', kind: 'view' };
    if (job === 'embed' || family === 'embed') return { label: 'Embed', kind: 'view' };
    if (job === 'rerank' || looksLikeReranker(m.modelId)) return { label: 'Rank', kind: 'view' };
    return { label: 'Chat', kind: 'chat' };
  }

  function renderContinue() {
    let runnable = state.localModels.filter(function (m) {
      const cr = m.lastAnalysis && m.lastAnalysis.canRunHere;
      if (cr) return m.downloaded && cr === 'yes';
      const fmt = m.manifest && m.manifest.format;
      return m.downloaded && (fmt === 'gguf' || fmt === 'bin' || fmt === 'onnx' || fmt === 'safetensors');
    });
    const used = readLastUsed();
    const usedAt = {};
    used.forEach(function (x) { usedAt[x.id] = x.at; });
    runnable.sort(function (a, b) {
      return (usedAt[b.modelId] || 0) - (usedAt[a.modelId] || 0);
    });
    runnable = runnable.slice(0, 4);
    const section = $('continue-section');
    if (!runnable.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    $('continue-list').innerHTML = runnable
      .map(function (m) {
        const overall = m.lastAnalysis && m.lastAnalysis.overall;
        const action = continueActionFor(m);
        const recent = usedAt[m.modelId] ? ' · last used' : '';
        return (
          '<div class="reco-card" data-id="' + esc(m.modelId) + '">' +
          '<div><div class="name">' + esc(m.modelId) + '</div><div class="why">' + gb(m.totalWeightGb) + ' on disk · ready offline' + recent + '</div></div>' +
          '<div class="meta">' + (overall ? pill(overall) : '') +
          '<button type="button" class="btn ghost" data-action="' + action.kind + '" data-id="' + esc(m.modelId) + '">' + action.label + '</button>' +
          '</div></div>'
        );
      })
      .join('');
    wireRecoActions($('continue-list'));
    renderCapabilityMap();
  }

  function localReady(pred) {
    return (state.localModels || []).some(function (m) {
      if (!m.downloaded) return false;
      const cr = m.lastAnalysis && m.lastAnalysis.canRunHere;
      if (cr && cr !== 'yes') return false;
      return pred(m);
    });
  }

  function recoUnlock(pred) {
    let best = null;
    (state.recos || []).forEach(function (p) {
      if (!p || !p.runPlan || p.runPlan.canRunHere !== 'yes') return;
      if (p.local && p.local.downloaded) return;
      if (!pred(p)) return;
      if (!best || (p.downloadGb || 99) < (best.downloadGb || 99)) best = p;
    });
    return best;
  }

  function jobPredLocal(job) {
    return function (m) {
      const j = m.lastAnalysis && m.lastAnalysis.job;
      const f = m.lastAnalysis && m.lastAnalysis.family;
      if (job === 'chat') return j === 'chat' || f === 'llm' || (!j && m.manifest && m.manifest.format === 'gguf' && !looksLikeReranker(m.modelId));
      if (job === 't2i') return j === 't2i' || f === 'diffusion';
      if (job === 'asr') return j === 'asr' || f === 'asr';
      if (job === 'tts') return j === 'tts' || f === 'tts';
      if (job === 'embed') return j === 'embed' || f === 'embed';
      if (job === 'rerank') return j === 'rerank' || looksLikeReranker(m.modelId);
      return false;
    };
  }

  function jobPredReco(job) {
    return function (p) {
      const j = p.runPlan && p.runPlan.job;
      const tag = p.useTag || '';
      if (job === 'chat') return j === 'chat' || tag === 'Talk';
      if (job === 't2i') return j === 't2i' || tag === 'See';
      if (job === 'asr') return j === 'asr' || /transcribe/i.test(tag);
      if (job === 'tts') return j === 'tts' || /speak/i.test(tag);
      if (job === 'embed') return j === 'embed' || tag === 'Embed';
      if (job === 'rerank') return j === 'rerank' || tag === 'Rank';
      return false;
    };
  }

  // Invention: this PC's AI capability passport — not a model list, a skill map.
  function renderCapabilityMap() {
    const el = $('cap-map');
    if (!el) return;
    const jobs = [
      { job: 'chat', label: 'Talk', pipeline: 'text-generation' },
      { job: 't2i', label: 'See', pipeline: 'text-to-image' },
      { job: 'asr', label: 'Transcribe', pipeline: 'automatic-speech-recognition' },
      { job: 'tts', label: 'Speak', pipeline: 'text-to-speech' },
      { job: 'embed', label: 'Embed', pipeline: 'feature-extraction' },
      { job: 'rerank', label: 'Rank', pipeline: 'text-ranking' },
    ];
    const cells = [];
    let unlock = null;
    jobs.forEach(function (j) {
      const ready = localReady(jobPredLocal(j.job));
      const u = recoUnlock(jobPredReco(j.job));
      let cls = 'cant';
      let sub = 'Not on this PC';
      if (ready) {
        cls = 'ready';
        sub = 'Ready offline';
      } else if (u) {
        cls = 'unlock';
        sub = 'Unlock · ' + (u.downloadGb != null ? gb(u.downloadGb) : 'download');
        if (!unlock || (u.downloadGb || 99) < (unlock.downloadGb || 99)) unlock = { job: j.label, pick: u };
      }
      cells.push(
        '<button type="button" class="cap-cell ' + cls + '" data-job="' + esc(j.job) +
        '" data-pipeline="' + esc(j.pipeline) + '"><b>' + esc(j.label) + '</b>' + esc(sub) + '</button>'
      );
    });
    el.innerHTML = cells.join('');
    el.querySelectorAll('.cap-cell').forEach(function (btn) {
      btn.onclick = function () { openJob(btn.getAttribute('data-job'), btn.getAttribute('data-pipeline')); };
    });
    const line = $('unlock-line');
    if (line) {
      line.textContent = unlock
        ? 'Next unlock: ' + unlock.job + ' — ' + unlock.pick.modelId + (unlock.pick.downloadGb != null ? ' (' + gb(unlock.pick.downloadGb) + ')' : '')
        : '';
    }
  }

  function openJob(job, pipeline) {
    const ready = (state.localModels || []).find(function (m) {
      return m.downloaded && jobPredLocal(job)(m);
    });
    if (ready) {
      const action = continueActionFor(ready);
      if (action.kind === 'chat') {
        switchView('library');
        selectLibraryModel(ready.modelId);
        return;
      }
      switchView('explore');
      selectModel(ready.modelId);
      return;
    }
    switchView('explore');
    state.exploreLoaded = true;
    loadTags().then(function () {
      $('cat').value = pipeline || '';
      $('runnable-only').checked = true;
      $('fits-only').checked = false;
      state.page = 0;
      loadModels();
    });
  }

  function copyPassport() {
    const hw = state.hw || {};
    const jobs = ['chat', 't2i', 'asr', 'tts', 'embed', 'rerank'];
    const ready = jobs.filter(function (j) { return localReady(jobPredLocal(j)); });
    const card = {
      kind: 'cyrai-passport-v1',
      at: new Date().toISOString(),
      machine: {
        cpu: hw.cpu && hw.cpu.brand,
        ramGb: hw.memory && hw.memory.totalGb,
        gpu: hw.gpu && hw.gpu.primary && hw.gpu.primary.model,
        vramGb: hw.gpu && hw.gpu.totalVramGb,
        backends: hw.backends,
        approxMaxParamsB: hw.budgets && hw.budgets.approxMaxParamsB,
      },
      ready: ready,
      recos: (state.recos || []).map(function (p) {
        return { id: p.modelId, job: p.runPlan && p.runPlan.job, canRunHere: p.runPlan && p.runPlan.canRunHere, downloadGb: p.downloadGb };
      }),
    };
    const text = JSON.stringify(card, null, 2);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { toast('Machine passport copied'); }).catch(function () { toast('Copy failed'); });
    } else toast('Copy not available');
  }

  function headroomCoachHtml(d) {
    const best = d.match && d.match.bestMode;
    if (!best || best.headroomGb == null) return '';
    const h = best.headroomGb;
    const where = best.device === 'cpu' || best.device === 'unified' ? 'RAM' : 'VRAM';
    if (h < 0) {
      return '<div class="callout tight"><span class="callout-label">Headroom</span>Need ~' +
        esc(gb(-h)) + ' more ' + where + ' for a comfortable fit (' + esc(best.label) + ').</div>';
    }
    if (best.status === 'tight') {
      return '<div class="callout tight"><span class="callout-label">Headroom</span>Tight — ~' +
        esc(gb(h)) + ' left. Close other apps before a long run.</div>';
    }
    return '';
  }

  /* ───────── explore ───────── */

  async function loadTags() {
    try {
      const r = await rpc().getTags();
      if (!r.ok) {
        $('cat').innerHTML = '<option value="">' + esc(friendlyError(r.error)) + '</option>';
        return;
      }
      const d = r.data;
      let html = '<option value="">All categories</option>';
      const groups = d.bySubType || {};
      const keys = Object.keys(groups).sort();
      for (let i = 0; i < keys.length; i++) {
        const sub = keys[i];
        html += '<optgroup label="' + esc(sub) + '">';
        for (let j = 0; j < groups[sub].length; j++) {
          const t = groups[sub][j];
          html += '<option value="' + esc(t.id) + '">' + esc(t.label) + '</option>';
        }
        html += '</optgroup>';
      }
      $('cat').innerHTML = html;

      let libs = '<option value="">Any library</option>';
      const libraries = d.libraries || [];
      for (let i = 0; i < libraries.length; i++) {
        const l = libraries[i];
        libs += '<option value="' + esc(l.id) + '">' + esc(l.label) + '</option>';
      }
      $('lib').innerHTML = libs;
      if (d.offline) toast('Using cached categories (offline)');
    } catch (e) {
      $('cat').innerHTML = '<option value="">' + esc(friendlyError(e.message || e)) + '</option>';
    }
  }

  async function loadModels() {
    $('meta').textContent = 'Loading…';
    $('models').innerHTML = '';
    $('btn-go').disabled = true;
    try {
      const sortVal = $('sort').value;
      // "Best fit" isn't a sort Hugging Face understands — fetch a solid default-
      // ordered candidate set, then re-sort it client-side once we've checked fit.
      const hfSort = sortVal === 'fit' ? 'downloads' : sortVal;
      let search = $('q').value.trim();
      // "Runnable only" filters client-side after fetching. Talk/See are GGUF-
      // shaped; Hear is ggml.bin (whisper) or onnx (piper) — GGUF bias hides both.
      const pipe = $('cat').value || '';
      const hearPipe = pipe === 'automatic-speech-recognition' || pipe === 'audio-to-audio'
        || pipe === 'text-to-speech' || pipe === 'text-to-audio';
      if ($('runnable-only').checked && hearPipe) {
        if ((pipe === 'automatic-speech-recognition' || pipe === 'audio-to-audio') && !/ggml|whisper/i.test(search)) {
          search = (search + ' ggml').trim();
        } else if ((pipe === 'text-to-speech' || pipe === 'text-to-audio') && !/onnx|piper/i.test(search)) {
          search = (search + ' onnx').trim();
        }
      } else if ($('runnable-only').checked && !/gguf/i.test(search)) {
        search = (search + ' GGUF').trim();
      }
      const r = await rpc().listModels({
        pipelineTag: $('cat').value || null,
        search: search,
        sort: hfSort,
        library: $('lib').value || null,
        limit: state.limit,
        page: state.page,
      });
      $('btn-go').disabled = false;
      if (!r.ok) {
        $('meta').textContent = friendlyError(r.error);
        $('models').innerHTML = '<div class="muted sm-text">' + esc(friendlyError(r.error)) + '</div>';
        $('btn-check').disabled = true;
        return;
      }
      state.models = r.data.models || [];
      state.analyses.clear();
      $('meta').textContent =
        state.models.length + ' models · page ' + (state.page + 1) + (r.data.offline ? ' · offline cache' : '');
      $('page').textContent = String(state.page + 1);
      $('prev').disabled = state.page <= 0;
      $('next').disabled = state.models.length < state.limit;
      $('btn-check').disabled = !state.models.length;
      const needsCheck = sortVal === 'fit' || $('fits-only').checked || $('runnable-only').checked;
      if (needsCheck && state.models.length) checkPage();
      else renderModels();
    } catch (e) {
      $('btn-go').disabled = false;
      $('meta').textContent = friendlyError(e.message || e);
      $('models').innerHTML = '<div class="muted sm-text">' + esc(friendlyError(e.message || e)) + '</div>';
    }
  }

  function renderModels() {
    if (!state.models.length) {
      $('models').innerHTML = '<div class="muted sm-text">No models for this search.</div>';
      return;
    }
    const fitsOnly = $('fits-only').checked;
    const runnableOnly = $('runnable-only').checked;
    const needsAnalysis = fitsOnly || runnableOnly;
    const checked = state.models.every(function (m) { return state.analyses.has(m.id); });
    let list = state.models;
    if (fitsOnly) {
      list = list.filter(function (m) {
        const a = state.analyses.get(m.id);
        return a && a.ok && ['excellent', 'good', 'tight'].indexOf(a.overall) >= 0;
      });
    }
    if (runnableOnly) {
      list = list.filter(function (m) {
        const a = state.analyses.get(m.id);
        return a && a.ok && a.runPlan && a.runPlan.canRunHere === 'yes';
      });
    }
    if (!list.length) {
      $('models').innerHTML =
        '<div class="muted sm-text">' +
        (needsAnalysis
          ? checked
            ? 'No models on this page match. Try a different category or page.'
            : 'Checking…'
          : 'No models.') +
        '</div>';
      return;
    }
    if ($('sort').value === 'fit' && checked) {
      list = list.slice().sort(function (a, b) {
        const ra = FIT_RANK[(state.analyses.get(a.id) || {}).overall] ?? 0;
        const rb = FIT_RANK[(state.analyses.get(b.id) || {}).overall] ?? 0;
        return rb - ra;
      });
    }
    if (checked && state.models.length) {
      let n = 0;
      state.models.forEach(function (m) {
        const a = state.analyses.get(m.id);
        if (a && a.ok && a.runPlan && a.runPlan.canRunHere === 'yes') n++;
      });
      const meta = $('meta');
      if (meta && meta.textContent.indexOf('run in this app') < 0) {
        meta.textContent += ' · ' + n + '/' + state.models.length + ' run in this app';
      }
    }
    $('models').innerHTML = list
      .map(function (m) {
        const a = state.analyses.get(m.id);
        let right = '<span class="muted sm-text">tap to check</span>';
        if (a && a.ok) {
          right =
            pill(a.overall) +
            '<div class="sub">' +
            (a.primaryLoadGb != null ? '~' + gb(a.primaryLoadGb) : '') +
            (a.local && a.local.downloaded ? ' · offline' : '') +
            '</div>';
        } else if (a && !a.ok) {
          right = pill('unknown');
        }
        return (
          '<article class="item ' + (state.selected === m.id ? 'on' : '') + '" data-id="' + esc(m.id) + '">' +
          '<div><div class="t">' + esc(m.id) + '</div><div class="sub">' +
          esc(m.pipelineTag || '—') + ' · ↓' + esc(m.downloads) + ' · ♥' + esc(m.likes) +
          '</div></div><div>' + right + '</div></article>'
        );
      })
      .join('');
    document.querySelectorAll('#models .item').forEach(function (el) {
      el.onclick = function () { selectModel(this.getAttribute('data-id')); };
    });
  }

  async function checkPage() {
    const ids = state.models.map(function (m) { return m.id; });
    if (!ids.length) return;
    $('btn-check').disabled = true;
    $('meta').textContent = 'Checking ' + ids.length + ' models…';
    // Real network work (one Hub lookup per model) — show visible progress so it
    // doesn't read as frozen, especially when a filter hid the list entirely.
    if ($('fits-only').checked || $('runnable-only').checked) {
      $('models').innerHTML = '<div class="skeleton-row"></div><div class="skeleton-row"></div><div class="skeleton-row"></div>';
    }
    try {
      const r = await rpc().analyzeBatch(ids);
      $('btn-check').disabled = false;
      if (!r.ok) {
        toast(friendlyError(r.error));
        $('meta').textContent = friendlyError(r.error);
        renderModels();
        return;
      }
      (r.data.results || []).forEach(function (res) { state.analyses.set(res.modelId, res); });
      $('meta').textContent = 'Checked ' + ids.length + ' against your hardware';
      renderModels();
    } catch (e) {
      $('btn-check').disabled = false;
      toast(friendlyError(e.message || e));
      $('meta').textContent = friendlyError(e.message || e);
      renderModels();
    }
  }

  async function selectModel(id) {
    state.selected = id;
    switchView('explore');
    renderModels();
    $('detail').innerHTML = '<div class="muted">Analyzing ' + esc(id) + '…</div>';
    try {
      const r = await rpc().analyze(id);
      if (!r.ok) {
        $('detail').innerHTML = '<div class="muted">' + esc(friendlyError(r.error)) + '</div>';
        return;
      }
      const d = r.data;
      state.analyses.set(id, {
        modelId: id,
        ok: true,
        overall: d.match.overall,
        primaryLoadGb: d.memory.primaryLoadGb,
        local: d.local,
        format: d.weightMeta && d.weightMeta.format,
        runPlan: d.runPlan,
        experimentalWarning: d.experimentalWarning,
      });
      renderModels();
      renderDetail(d);
    } catch (e) {
      $('detail').innerHTML = '<div class="muted">' + esc(friendlyError(e.message || e)) + '</div>';
    }
  }

  function renderDetail(d) {
    const modes = (d.match.modes || []).slice(0, 12);
    const local = d.local || {};
    const mins = estimateMinutes(d.memory.downloadGb);
    const rows = modes
      .map(function (m) {
        return (
          '<tr><td>' + esc(m.label) + '</td><td class="mono">' + gb(m.loadGb) + '</td>' +
          '<td class="mono">' + gb(m.budgetGb) + '</td><td>' + pill(m.status) + '</td></tr>'
        );
      })
      .join('');

    const canRunHere = local.downloaded && d.runPlan && d.runPlan.canRunHere === 'yes';
    const isSee = canRunHere && (d.runPlan.runner === 'sd' || d.runPlan.job === 't2i' || d.ir && d.ir.family === 'diffusion');
    const hear = canRunHere ? hearKind(d) : null;
    const isEmbed = canRunHere && isEmbedModel(d);
    const isRerank = canRunHere && isRerankModel(d);
    const isTalk = canRunHere && !isSee && !hear && !isEmbed && !isRerank;
    const notRunnable = !d.runPlan || d.runPlan.canRunHere !== 'yes';
    const runnerPending = d.runPlan && d.runPlan.canRunHere === 'download_runner';

    $('detail').innerHTML =
      '<div class="t" style="font-weight:700;word-break:break-all;margin-bottom:10px;font-size:15px">' + esc(d.modelId) + '</div>' +
      '<div class="summary">' + esc((d.runPlan && d.runPlan.fitLine) || d.match.summary) + '</div>' +
      headroomCoachHtml(d) +
      downloadFactsHtml(d) +
      (d.experimentalWarning
        ? '<div class="callout bad"><span class="callout-label">Before you download</span>' + esc(d.experimentalWarning) + '</div>'
        : '') +
      '<div class="actions">' +
      '<button type="button" class="btn primary" id="btn-dl"' + (local.downloaded ? ' disabled' : '') + '>' +
      (local.downloaded ? 'Downloaded' : 'Download for offline') + '</button>' +
      (isTalk ? '<button type="button" class="btn ghost" id="btn-chat-jump">Chat in Library</button>' : '') +
      (local.downloaded || local.partial ? '<button type="button" class="btn danger" id="btn-rm">Remove</button>' : '') +
      '<button type="button" class="btn ghost" id="btn-folder">Open folder</button>' +
      '<button type="button" class="btn ghost" id="btn-compare">' +
      (state.compareIds.indexOf(d.modelId) >= 0 ? 'Remove from compare' : 'Compare') + '</button>' +
      '<button type="button" class="link" id="btn-hf">View on Hub</button></div>' +
      (isSee ? generatePanelHtml() : '') +
      (hear ? hearPanelHtml(hear) : '') +
      (isEmbed ? embedPanelHtml() : '') +
      (isRerank ? rerankPanelHtml() : '') +
      (notRunnable
        ? '<div class="callout neutral"><span class="callout-label">' +
          (runnerPending ? 'Runner not in the app yet' : "Can't run in this app") +
          '</span>' +
          esc(runnerPending
            ? 'Fits as a file, but no in-app adapter for this exact format or layout — download for your own tools.'
            : d.runPlan && d.runPlan.canRunHere === 'export_only'
              ? 'No in-app runner for this format — download and use your own tooling.'
              : "Flagged as likely needing custom tooling.") +
          '<details class="tech" id="setup-notes" style="margin-top:8px"><summary>Setup notes from the model card</summary>' +
          '<div id="setup-notes-body" class="sm-text" style="margin-top:6px">Click to load…</div></details>' +
          '</div>'
        : '') +
      '<details class="tech"><summary>Technical details</summary>' +
      '<dl class="kv">' +
      '<dt>Format</dt><dd>' + esc(d.weightMeta.format || '?') + '</dd>' +
      (d.realMeta?.architecture ? '<dt>Architecture</dt><dd class="mono">' + esc(d.realMeta.architecture) + '</dd>' : '') +
      '<dt>Weights</dt><dd>' + gb(d.weightMeta.totalWeightGb) + '</dd>' +
      '<dt>Params' + (d.weightMeta.parameters?.exactBillions != null ? ' <span class="muted">(exact)</span>' : '') + '</dt><dd>' +
      (d.weightMeta.parameters?.exactBillions != null
        ? d.weightMeta.parameters.exactBillions + 'B'
        : d.weightMeta.parameters?.billions != null
          ? '~' + d.weightMeta.parameters.billions + 'B'
          : '—') +
      '</dd>' +
      '<dt>Offline copy</dt><dd>' + (local.downloaded ? 'Yes' : local.partial ? 'Partial — resume supported' : 'No') + '</dd>' +
      '<dt>Download size</dt><dd>' + gb(d.memory.downloadGb) + (!local.downloaded && mins ? ' · ~' + mins + ' min' : '') + '</dd>' +
      '<dt>Backend</dt><dd>' + esc(backendLabel((d.runPlan && d.runPlan.backend) || pickBackend(state.hw) || '—')) + '</dd>' +
      '</dl>' +
      (modes.length
        ? '<table><thead><tr><th>Mode</th><th>Load</th><th>Budget</th><th>Fit</th></tr></thead><tbody>' + rows + '</tbody></table>'
        : '') +
      '<p class="sm-text muted">' +
      (modes.length && modes[0].kvCacheGb
        ? 'Load figures include ~4K tokens of real conversation memory' + (d.realMeta?.contextEstimates ? ' (exact, from the model\'s own header)' : ' (estimated)') + ', plus ~20% runtime overhead.'
        : '~20% runtime overhead included in load estimates.') +
      '</p>' +
      contextEstimatesHtml(d.realMeta?.contextEstimates || d.memory.contextEstimates) +
      '</details>';

    const setupNotes = $('setup-notes');
    if (setupNotes) {
      let loaded = false;
      setupNotes.addEventListener('toggle', async function () {
        if (!setupNotes.open || loaded) return;
        loaded = true;
        const body = $('setup-notes-body');
        body.textContent = 'Loading…';
        try {
          const r = await rpc().getReadme(d.modelId);
          if (r.ok && r.data.snippet) {
            body.textContent = r.data.snippet + (r.data.truncated ? '…' : '');
            body.style.whiteSpace = 'pre-wrap';
          } else {
            body.textContent = 'No setup notes found in the model card — check the Hub page directly.';
          }
        } catch (e) {
          body.textContent = friendlyError(e.message || e);
        }
      });
    }
    const btnDl = $('btn-dl');
    if (btnDl) btnDl.onclick = function () { downloadModel(d.modelId); };
    const btnChatJump = $('btn-chat-jump');
    if (btnChatJump) btnChatJump.onclick = function () { switchView('library'); selectLibraryModel(d.modelId); };
    const btnRm = $('btn-rm');
    if (btnRm) {
      btnRm.onclick = async function () {
        await rpc().deleteLocal(d.modelId);
        toast('Removed local files');
        selectModel(d.modelId);
        loadLocal();
      };
    }
    const btnFolder = $('btn-folder');
    if (btnFolder) btnFolder.onclick = function () { rpc().openDir(d.modelId); };
    const btnHf = $('btn-hf');
    if (btnHf) btnHf.onclick = function () { rpc().openUrl(d.weightMeta.cardUrl); };
    const btnCompare = $('btn-compare');
    if (btnCompare) btnCompare.onclick = function () { toggleCompare(d); };
    if (isSee) wireGenerate(d.modelId);
    if (hear) wireHear(d.modelId, hear);
    if (isEmbed) wireEmbed(d.modelId);
    if (isRerank) wireRerank(d.modelId);
  }

  function genButtonLabel() {
    if (state.seeInitPath && state.seeMaskPath) return 'Inpaint 512×512';
    if (state.seeInitPath) return 'Image-to-image';
    return 'Generate 512×512';
  }

  function generatePanelHtml() {
    return (
      '<div class="gen-panel" id="gen-panel">' +
      '<p class="sm-text muted">512×512. First image can take several minutes on CPU. If Vulkan fails to start, we retry on CPU automatically.</p>' +
      '<p class="sm-text muted" id="gen-elapsed" hidden></p>' +
      '<label class="sm-text">Prompt<textarea id="gen-prompt" rows="3" placeholder="a cat on a windowsill, photograph"></textarea></label>' +
      '<div class="row gap" style="margin-top:8px">' +
      '<button type="button" class="btn ghost" id="btn-pick-init">Init image</button>' +
      '<span id="gen-init-name" class="sm-text muted">' +
      esc(state.seeInitPath ? fileName(state.seeInitPath) : 'optional — img2img / inpaint') + '</span>' +
      (state.seeInitPath ? '<button type="button" class="link sm-text" id="btn-clear-init">Clear</button>' : '') +
      '</div>' +
      '<div class="row gap" style="margin-top:6px">' +
      '<button type="button" class="btn ghost" id="btn-pick-mask">Mask</button>' +
      '<span id="gen-mask-name" class="sm-text muted">' +
      esc(state.seeMaskPath ? fileName(state.seeMaskPath) : 'optional — inpaint with init') + '</span>' +
      (state.seeMaskPath ? '<button type="button" class="link sm-text" id="btn-clear-mask">Clear</button>' : '') +
      '</div>' +
      '<label class="sm-text" id="gen-strength-wrap"' + (state.seeInitPath ? '' : ' hidden') +
      '>Strength: <span id="gen-strength-val">0.75</span>' +
      '<input type="range" id="gen-strength" min="0.05" max="1" step="0.05" value="0.75" /></label>' +
      '<div class="row gap" style="margin-top:8px">' +
      '<button type="button" class="btn primary" id="btn-gen">' + esc(genButtonLabel()) + '</button>' +
      '<button type="button" class="btn danger" id="btn-gen-stop" hidden>Cancel</button>' +
      '</div>' +
      '<pre id="gen-log" class="gen-log sm-text muted" hidden></pre>' +
      '<img id="gen-img" class="gen-img" alt="" hidden />' +
      '</div>'
    );
  }

  function refreshGenMode() {
    const btn = $('btn-gen');
    if (btn) btn.textContent = genButtonLabel();
    const wrap = $('gen-strength-wrap');
    if (wrap) wrap.hidden = !state.seeInitPath;
    const initName = $('gen-init-name');
    if (initName) initName.textContent = state.seeInitPath ? fileName(state.seeInitPath) : 'optional — img2img / inpaint';
    const maskName = $('gen-mask-name');
    if (maskName) maskName.textContent = state.seeMaskPath ? fileName(state.seeMaskPath) : 'optional — inpaint with init';
  }

  function ensureClearBtn(id, parent, kind) {
    if ($(id) || !parent) return;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'link sm-text';
    b.id = id;
    b.textContent = 'Clear';
    parent.appendChild(b);
    b.onclick = function () { clearSeeImage(kind); };
  }

  function clearSeeImage(kind) {
    if (kind === 'mask') state.seeMaskPath = null;
    else state.seeInitPath = null;
    const clearId = kind === 'mask' ? 'btn-clear-mask' : 'btn-clear-init';
    const btn = $(clearId);
    if (btn) btn.remove();
    refreshGenMode();
  }

  async function pickSeeImage(kind) {
    try {
      const r = await rpc().pickImageFile();
      if (!r.ok) throw new Error(r.error);
      if (!r.data) return;
      if (kind === 'mask') state.seeMaskPath = r.data;
      else state.seeInitPath = r.data;
      refreshGenMode();
      if (kind === 'mask') ensureClearBtn('btn-clear-mask', $('gen-mask-name') && $('gen-mask-name').parentElement, 'mask');
      else ensureClearBtn('btn-clear-init', $('gen-init-name') && $('gen-init-name').parentElement, 'init');
    } catch (e) {
      toast(friendlyError(e.message || e));
    }
  }

  function wireGenerate(modelId) {
    const promptEl = $('gen-prompt');
    const btn = $('btn-gen');
    const stop = $('btn-gen-stop');
    const logEl = $('gen-log');
    const img = $('gen-img');
    const strength = $('gen-strength');
    const strengthVal = $('gen-strength-val');
    if (!btn) return;
    if (strength && strengthVal) {
      strength.oninput = function () { strengthVal.textContent = strength.value; };
    }
    const pickInit = $('btn-pick-init');
    const pickMask = $('btn-pick-mask');
    if (pickInit) pickInit.onclick = function () { pickSeeImage('init'); };
    if (pickMask) pickMask.onclick = function () { pickSeeImage('mask'); };
    const clearInit = $('btn-clear-init');
    const clearMask = $('btn-clear-mask');
    if (clearInit) clearInit.onclick = function () { clearSeeImage('init'); };
    if (clearMask) clearMask.onclick = function () { clearSeeImage('mask'); };
    btn.onclick = async function () {
      const prompt = (promptEl && promptEl.value.trim()) || '';
      if (!prompt) {
        toast('Write a prompt first.');
        return;
      }
      if (state.seeMaskPath && !state.seeInitPath) {
        toast('Mask needs an init image — generating from the prompt only.');
      }
      btn.disabled = true;
      stop.hidden = false;
      logEl.hidden = false;
      logEl.textContent = 'Starting… first image on CPU can take several minutes.\n';
      img.hidden = true;
      const elapsedEl = $('gen-elapsed');
      const t0 = Date.now();
      let tick = null;
      if (elapsedEl) {
        elapsedEl.hidden = false;
        const paint = function () {
          const s = Math.round((Date.now() - t0) / 1000);
          elapsedEl.textContent = s < 60
            ? 'Working… ' + s + 's'
            : 'Working… ' + Math.floor(s / 60) + 'm ' + (s % 60) + 's — CPU first image is slow';
        };
        paint();
        tick = setInterval(paint, 1000);
      }
      const opts = { prompt: prompt, width: 512, height: 512 };
      if (state.seeInitPath) {
        opts.initImagePath = state.seeInitPath;
        opts.strength = strength ? parseFloat(strength.value) : 0.75;
        if (state.seeMaskPath) opts.maskImagePath = state.seeMaskPath;
      }
      try {
        const r = await rpc().generateImage(modelId, opts);
        if (!r.ok) throw new Error(r.error);
        const out = r.data && r.data.outputPath;
        const used = r.data && r.data.backend;
        const note = r.data && r.data.fellBack ? ' (Vulkan failed — used CPU)' : (used ? ' on ' + used : '');
        logEl.textContent += out ? 'Saved ' + out + note + '\n' : 'Done' + note + '.\n';
        if (out) {
          const prev = await rpc().readDataUrl(out);
          if (prev.ok) {
            img.src = prev.data;
            img.hidden = false;
          }
        }
        touchLastUsed(modelId, 't2i');
        toast(r.data && r.data.fellBack ? 'Image saved on CPU' : 'Image saved');
      } catch (e) {
        logEl.textContent += friendlyError(e.message || e) + '\n';
        toast(friendlyError(e.message || e));
      } finally {
        if (tick) clearInterval(tick);
        btn.disabled = false;
        stop.hidden = true;
      }
    };
    stop.onclick = function () { rpc().cancelGenerate(modelId); };
  }

  function embedPanelHtml() {
    return (
      '<div class="gen-panel" id="embed-panel">' +
      '<label class="sm-text">Text<textarea id="embed-a" rows="3" placeholder="A sentence to embed"></textarea></label>' +
      '<label class="sm-text" style="display:block;margin-top:8px">Compare with (optional)' +
      '<textarea id="embed-b" rows="2" placeholder="Another sentence — shows cosine similarity"></textarea></label>' +
      '<div class="row gap" style="margin-top:8px">' +
      '<button type="button" class="btn primary" id="btn-embed">Embed</button>' +
      '<button type="button" class="btn ghost" id="btn-embed-copy" hidden>Copy vector</button>' +
      '</div>' +
      '<pre id="embed-out" class="embed-out sm-text" hidden></pre>' +
      '</div>'
    );
  }

  function wireEmbed(modelId) {
    const btn = $('btn-embed');
    const copy = $('btn-embed-copy');
    const out = $('embed-out');
    if (!btn) return;
    let lastVec = null;
    btn.onclick = async function () {
      const a = (($('embed-a') && $('embed-a').value) || '').trim();
      const b = (($('embed-b') && $('embed-b').value) || '').trim();
      if (!a) {
        toast('Write some text to embed first.');
        return;
      }
      btn.disabled = true;
      lastVec = null;
      if (copy) copy.hidden = true;
      if (out) {
        out.hidden = false;
        out.textContent = 'Embedding… first load can take a bit.\n';
      }
      try {
        const r1 = await rpc().embedText(modelId, a);
        if (!r1.ok) throw new Error(r1.error);
        const v1 = r1.data && r1.data.vector;
        if (!v1 || !v1.length) throw new Error('No vector returned.');
        lastVec = v1;
        let text = vecPreview(v1);
        if (b) {
          const r2 = await rpc().embedText(modelId, b);
          if (!r2.ok) throw new Error(r2.error);
          const v2 = r2.data && r2.data.vector;
          if (!v2 || !v2.length) throw new Error('No vector returned for the second text.');
          const sim = cosineSim(v1, v2);
          text += '\n' + vecPreview(v2);
          text += '\ncosine ' + (sim == null ? '—' : sim.toFixed(3));
        }
        if (out) out.textContent = text;
        if (copy) copy.hidden = false;
        touchLastUsed(modelId, 'embed');
        toast(b ? 'Compared' : v1.length + '-d vector');
      } catch (e) {
        if (out) out.textContent = friendlyError(e.message || e);
        toast(friendlyError(e.message || e));
      } finally {
        btn.disabled = false;
      }
    };
    if (copy) {
      copy.onclick = async function () {
        if (!lastVec) return;
        try {
          await navigator.clipboard.writeText(JSON.stringify(lastVec));
          toast('Vector copied');
        } catch (e) {
          toast('Could not copy');
        }
      };
    }
  }

  function rerankPanelHtml() {
    return (
      '<div class="gen-panel" id="rerank-panel">' +
      '<label class="sm-text">Query<textarea id="rerank-q" rows="2" placeholder="What should these documents be ranked against?"></textarea></label>' +
      '<label class="sm-text" style="display:block;margin-top:8px">Documents (one per line)' +
      '<textarea id="rerank-docs" rows="6" placeholder="Paris is the capital of France.&#10;The Eiffel Tower is in Paris.&#10;Bananas are a fruit.&#10;Photosynthesis happens in plants."></textarea></label>' +
      '<div class="row gap" style="margin-top:8px">' +
      '<button type="button" class="btn primary" id="btn-rerank">Rank</button>' +
      '</div>' +
      '<ol id="rerank-out" class="rerank-out sm-text" hidden></ol>' +
      '</div>'
    );
  }

  function wireRerank(modelId) {
    const btn = $('btn-rerank');
    const out = $('rerank-out');
    if (!btn) return;
    btn.onclick = async function () {
      const query = (($('rerank-q') && $('rerank-q').value) || '').trim();
      const raw = ($('rerank-docs') && $('rerank-docs').value) || '';
      const documents = raw.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean);
      if (!query) {
        toast('Write a query first.');
        return;
      }
      if (documents.length < 2) {
        toast('Add at least two documents, one per line.');
        return;
      }
      btn.disabled = true;
      if (out) {
        out.hidden = false;
        out.innerHTML = '<li class="muted">Ranking… first load can take a bit.</li>';
      }
      try {
        const r = await rpc().rerankDocuments(modelId, query, documents);
        if (!r.ok) throw new Error(r.error);
        const ranked = r.data && r.data.ranked;
        if (!ranked || !ranked.length) throw new Error('No ranking returned.');
        if (out) {
          out.innerHTML = ranked.map(function (row) {
            const score = row && typeof row.score === 'number' ? row.score.toFixed(4) : '—';
            return '<li><span class="score">' + esc(score) + '</span> ' + esc(row && row.document) + '</li>';
          }).join('');
        }
        touchLastUsed(modelId, 'rerank');
        toast('Ranked ' + ranked.length);
      } catch (e) {
        if (out) out.innerHTML = '<li>' + esc(friendlyError(e.message || e)) + '</li>';
        toast(friendlyError(e.message || e));
      } finally {
        btn.disabled = false;
      }
    };
  }

  function hearPanelHtml(kind) {
    if (kind === 'tts') {
      return (
        '<div class="gen-panel" id="hear-panel">' +
        '<label class="sm-text">Text to speak<textarea id="hear-text-in" rows="3" placeholder="Type something to speak"></textarea></label>' +
        '<div class="row gap" style="margin-top:8px">' +
        '<button type="button" class="btn primary" id="btn-speak">Speak</button>' +
        '<button type="button" class="btn danger" id="btn-speak-stop" hidden>Cancel</button>' +
        '</div>' +
        '<pre id="hear-log" class="gen-log sm-text muted" hidden></pre>' +
        '<audio id="hear-audio" class="gen-audio" controls hidden></audio>' +
        '</div>'
      );
    }
    return (
      '<div class="gen-panel" id="hear-panel">' +
      '<div class="row gap">' +
      '<button type="button" class="btn ghost" id="btn-pick-audio">Choose audio file</button>' +
      '<span id="hear-file" class="sm-text muted">wav / mp3 / flac / ogg</span>' +
      '</div>' +
      '<label class="sm-text" style="display:block;margin-top:8px">Language (optional)' +
      '<input id="hear-lang" type="text" placeholder="auto" autocomplete="off" /></label>' +
      '<div class="row gap" style="margin-top:8px">' +
      '<button type="button" class="btn primary" id="btn-asr">Transcribe</button>' +
      '<button type="button" class="btn danger" id="btn-asr-stop" hidden>Cancel</button>' +
      '</div>' +
      '<pre id="hear-log" class="gen-log sm-text muted" hidden></pre>' +
      '<div id="hear-out" class="hear-out" hidden></div>' +
      '</div>'
    );
  }

  function appendHearLog(text) {
    const logEl = $('hear-log');
    if (!logEl || !text) return;
    logEl.hidden = false;
    logEl.textContent += text + (/\n$/.test(text) ? '' : '\n');
    logEl.scrollTop = logEl.scrollHeight;
  }

  function wireHear(modelId, kind) {
    const logEl = $('hear-log');
    if (kind === 'tts') {
      const textEl = $('hear-text-in');
      const btn = $('btn-speak');
      const stop = $('btn-speak-stop');
      const audio = $('hear-audio');
      if (!btn) return;
      btn.onclick = async function () {
        const text = (textEl && textEl.value.trim()) || '';
        if (!text) {
          toast('Write something to speak first.');
          return;
        }
        btn.disabled = true;
        stop.hidden = false;
        if (logEl) {
          logEl.hidden = false;
          logEl.textContent = 'Speaking…\n';
        }
        if (audio) audio.hidden = true;
        try {
          const r = await rpc().speakText(modelId, text, {});
          if (!r.ok) throw new Error(r.error);
          const out = r.data && r.data.outputPath;
          appendHearLog(out ? 'Saved ' + out : 'Done.');
          if (out && audio) {
            const prev = await rpc().readDataUrl(out);
            if (prev.ok) {
              audio.src = prev.data;
              audio.hidden = false;
              audio.play().catch(function () { /* user gesture already spent */ });
            }
          }
          touchLastUsed(modelId, 'tts');
          toast('Speech saved');
        } catch (e) {
          appendHearLog(friendlyError(e.message || e));
          toast(friendlyError(e.message || e));
        } finally {
          btn.disabled = false;
          stop.hidden = true;
        }
      };
      stop.onclick = function () { rpc().cancelSpeak(modelId); };
      return;
    }

    const pick = $('btn-pick-audio');
    const fileEl = $('hear-file');
    const btn = $('btn-asr');
    const stop = $('btn-asr-stop');
    const outEl = $('hear-out');
    if (pick) {
      pick.onclick = async function () {
        try {
          const r = await rpc().pickAudioFile();
          if (!r.ok) throw new Error(r.error);
          state.hearAudioPath = r.data || null;
          if (fileEl) fileEl.textContent = state.hearAudioPath ? fileName(state.hearAudioPath) : 'wav / mp3 / flac / ogg';
        } catch (e) {
          toast(friendlyError(e.message || e));
        }
      };
    }
    if (!btn) return;
    btn.onclick = async function () {
      if (!state.hearAudioPath) {
        toast('Choose an audio file first.');
        return;
      }
      const lang = ($('hear-lang') && $('hear-lang').value.trim()) || '';
      btn.disabled = true;
      stop.hidden = false;
      if (logEl) {
        logEl.hidden = false;
        logEl.textContent = 'Transcribing…\n';
      }
      if (outEl) {
        outEl.hidden = true;
        outEl.textContent = '';
      }
      try {
        const r = await rpc().transcribeAudio(modelId, state.hearAudioPath, lang ? { language: lang } : {});
        if (!r.ok) throw new Error(r.error);
        const text = r.data && r.data.text;
        appendHearLog(text ? 'Done.' : 'Done (empty transcript).');
        if (outEl) {
          outEl.textContent = text || '';
          outEl.hidden = !text;
        }
        touchLastUsed(modelId, 'asr');
        toast(text ? 'Transcript ready' : 'No text in transcript');
      } catch (e) {
        appendHearLog(friendlyError(e.message || e));
        toast(friendlyError(e.message || e));
      } finally {
        btn.disabled = false;
        stop.hidden = true;
      }
    };
    stop.onclick = function () { rpc().cancelTranscribe(modelId); };
  }

  function toggleCompare(d) {
    state.compareData.set(d.modelId, d);
    const i = state.compareIds.indexOf(d.modelId);
    if (i >= 0) {
      state.compareIds.splice(i, 1);
    } else {
      state.compareIds.push(d.modelId);
      if (state.compareIds.length > 2) state.compareIds.shift(); // keep the 2 most recent
    }
    renderDetail(d); // refresh the button label
    renderCompare();
  }

  function compareRow(label, fn) {
    return '<tr><th>' + esc(label) + '</th>' + state.compareIds.map(function (id) {
      const d = state.compareData.get(id);
      return '<td>' + (d ? fn(d) : '—') + '</td>';
    }).join('') + '</tr>';
  }

  function renderCompare() {
    const panel = $('compare-panel');
    if (state.compareIds.length < 2) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    const names = '<tr><th></th>' + state.compareIds.map(function (id) {
      return '<td class="mono" style="word-break:break-all">' + esc(id) + '</td>';
    }).join('') + '</tr>';
    $('compare-table').innerHTML =
      '<div class="table-scroll"><table>' +
      names +
      compareRow('Fit', function (d) { return pill(d.match.overall); }) +
      compareRow('Format', function (d) { return esc(d.weightMeta.format || '?'); }) +
      compareRow('Params', function (d) {
        return d.weightMeta.parameters && d.weightMeta.parameters.billions != null ? d.weightMeta.parameters.billions + 'B' : '—';
      }) +
      compareRow('Load size', function (d) { return gb(d.memory.primaryLoadGb); }) +
      compareRow('Download size', function (d) { return gb(d.memory.downloadGb); }) +
      compareRow('Download ETA', function (d) {
        const mins = estimateMinutes(d.memory.downloadGb);
        return mins ? '~' + mins + ' min' : '—';
      }) +
      compareRow('Backend', function (d) {
        return esc(backendLabel((d.runPlan && d.runPlan.backend) || pickBackend(state.hw) || '—'));
      }) +
      compareRow('Downloaded', function (d) { return d.local && d.local.downloaded ? 'Yes' : 'No'; }) +
      '</table></div>';
  }

  async function downloadModel(id) {
    state.downloadingId = id;
    toast('Downloading ' + id + '…');
    try {
      const r = await rpc().download(id);
      state.downloadingId = null;
      if (!r.ok) {
        toast(friendlyError(r.error));
        await selectModel(id);
        return;
      }
      toast('Saved for offline use');
      await selectModel(id);
      await loadLocal();
    } catch (e) {
      state.downloadingId = null;
      toast(friendlyError(e.message || e));
    }
  }

  async function showCached() {
    try {
      const r = await rpc().listCached();
      if (!r.ok) return toast(friendlyError(r.error));
      const items = r.data || [];
      if (!items.length) return toast('No cached checks yet');
      state.models = items.map(function (m) { return { id: m.modelId, pipelineTag: 'cached', downloads: 0, likes: 0 }; });
      state.analyses.clear();
      items.forEach(function (m) {
        if (m.lastAnalysis) {
          state.analyses.set(m.modelId, {
            modelId: m.modelId, ok: true, overall: m.lastAnalysis.overall, primaryLoadGb: m.lastAnalysis.primaryLoadGb,
          });
        }
      });
      state.page = 0;
      $('meta').textContent = items.length + ' cached checks (offline-ready)';
      $('btn-check').disabled = true;
      $('prev').disabled = true;
      $('next').disabled = true;
      renderModels();
      toast('Showing cached model checks');
    } catch (e) {
      toast(friendlyError(e.message || e));
    }
  }

  /* ───────── library / chat ───────── */

  async function loadLocal() {
    try {
      const r = await rpc().listLocal();
      if (!r.ok) {
        $('local-list').textContent = friendlyError(r.error);
        return;
      }
      state.localModels = r.data || [];
      renderContinue();
      const items = state.localModels;
      if (!items.length) {
        $('local-list').innerHTML = '<div class="sm-text muted">No downloads yet. Find one in Explore.</div>';
        return;
      }
      $('local-list').innerHTML = items
        .map(function (m) {
          const st = m.lastAnalysis && m.lastAnalysis.overall;
          return (
            '<div class="item ' + (state.librarySelected === m.modelId ? 'on' : '') + '" data-id="' + esc(m.modelId) + '">' +
            '<div><div class="t">' + esc(m.modelId) + '</div><div class="sub">' +
            (m.downloaded ? 'ready offline' : 'partial') + ' · ' + gb(m.totalWeightGb) +
            (m.lastAnalysis && m.lastAnalysis.canRunHere !== 'yes' ? ' · download only' : '') +
            '</div></div><div>' + (st ? pill(st) : '') + '</div></div>'
          );
        })
        .join('');
      document.querySelectorAll('#local-list .item').forEach(function (el) {
        el.onclick = function () { selectLibraryModel(this.getAttribute('data-id')); };
      });
    } catch (e) {
      $('local-list').textContent = friendlyError(e.message || e);
    }
  }

  async function selectLibraryModel(id) {
    state.librarySelected = id;
    document.querySelectorAll('#local-list .item').forEach(function (el) {
      el.classList.toggle('on', el.getAttribute('data-id') === id);
    });
    const m = state.localModels.find(function (x) { return x.modelId === id; });
    if (!m) return;
    $('chat-empty').hidden = true;
    $('chat-panel').hidden = false;
    $('chat-title').textContent = id;

    const isGguf = m.manifest && m.manifest.format === 'gguf';
    const fmt = m.manifest && m.manifest.format;
    const family = m.lastAnalysis && m.lastAnalysis.family;
    const canRun = !m.lastAnalysis || m.lastAnalysis.canRunHere === 'yes';
    let job = (m.lastAnalysis && m.lastAnalysis.job) || null;
    const knownStudio = family === 'embed' || family === 'diffusion' || family === 'asr' || family === 'tts';
    // Rerankers are family 'other'. Re-analyze only if job is missing.
    if (isGguf && !job && !knownStudio && family !== 'llm') {
      try {
        const r = await rpc().analyze(id);
        if (state.librarySelected !== id) return;
        if (r.ok && r.data && r.data.runPlan) job = r.data.runPlan.job;
      } catch (e) { /* offline / uncached — id heuristic below */ }
    }
    const studioKind = job === 'rerank' || looksLikeReranker(id) ? 'rerank'
      : job === 'embed' || family === 'embed' ? 'embed'
      : job === 't2i' || family === 'diffusion' ? 'diffusion'
      : job === 'asr' || family === 'asr' ? 'asr'
      : job === 'tts' || family === 'tts' ? 'tts'
      : !m.lastAnalysis && fmt === 'onnx' ? 'tts'
      : !m.lastAnalysis && fmt === 'bin' ? 'asr'
      : !m.lastAnalysis && fmt === 'safetensors' ? 'diffusion'
      : null;
    const studio = !!(studioKind && canRun);
    const studioEl = $('lib-studio');
    if (studioEl) {
      studioEl.hidden = !studio;
      studioEl.innerHTML = '';
    }
    $('chat-runnable').hidden = studio || !isGguf;
    $('chat-noninfer').hidden = studio || isGguf;
    if (studio && studioEl) {
      const jump = studioKind === 'rerank' ? 'Rank in Explore'
        : studioKind === 'embed' ? 'Embed in Explore'
        : studioKind === 'diffusion' ? 'Generate in Explore'
        : studioKind === 'tts' ? 'Speak in Explore'
        : 'Transcribe in Explore';
      const copy = studioKind === 'rerank'
        ? "This is a reranker — it doesn't chat."
        : studioKind === 'embed'
        ? "This is an embedding model — it doesn't chat."
        : 'This model runs from Explore, not chat.';
      studioEl.innerHTML =
        '<p>' + esc(copy) + '</p>' +
        '<button type="button" class="btn primary" id="btn-lib-studio">' + esc(jump) + '</button>';
      const btnStudio = $('btn-lib-studio');
      if (btnStudio) btnStudio.onclick = function () { switchView('explore'); selectModel(id); };
    } else if (!isGguf) {
      const btnHub = $('btn-lib-hub');
      if (btnHub) btnHub.onclick = function () { rpc().openUrl('https://huggingface.co/' + id); };
    } else {
      if (!state.chatHistoryLoaded.has(id)) {
        state.chatHistoryLoaded.add(id);
        const r = await rpc().getChatHistory(id);
        if (r.ok && r.data.length && !state.chatLogs.has(id)) state.chatLogs.set(id, r.data);
      }
      // selecting a different model in between could have raced ahead of this await
      if (state.librarySelected !== id) return;
      renderChatLog(id);
      updateChatStatus(id);
    }

    $('btn-lib-folder').onclick = function () { rpc().openDir(id); };
    $('btn-chat-remove').onclick = async function () {
      if (state.chatLoadedModelId === id) {
        await rpc().unloadChatModel();
        state.chatLoadedModelId = null;
        state.chatLoadedSettings = null;
      }
      await rpc().deleteLocal(id);
      toast('Removed local files');
      state.librarySelected = null;
      $('chat-panel').hidden = true;
      $('chat-empty').hidden = false;
      loadLocal();
    };
  }

  function renderChatLog(modelId) {
    const log = state.chatLogs.get(modelId) || [];
    const log$ = $('chat-log');
    log$.innerHTML = log
      .map(function (m) {
        const speed = m.tokensPerSecond ? '<div class="msg-speed">' + m.tokensPerSecond + ' tok/s on this machine</div>' : '';
        const extra = m.role === 'error' ? errorExtrasHtml(m) : '';
        return '<div class="msg ' + esc(m.role) + '">' + esc(m.text) + speed + extra + '</div>';
      })
      .join('');
    document.querySelectorAll('#chat-log [data-open-card]').forEach(function (el) {
      el.onclick = function () { rpc().openUrl(this.getAttribute('data-open-card')); };
    });
    log$.scrollTop = log$.scrollHeight;
  }

  // When loading a model fails, offer the model card / README — often the only
  // place that explains a non-standard model needs special tooling to run.
  function errorExtrasHtml(m) {
    if (!m.cardUrl) return '';
    if (m.readmeLoading) return '<div class="msg-extra sm-text muted">Checking the model card…</div>';
    const snippet = m.readmeSnippet
      ? '<details class="tech"><summary>Notes from the model card</summary><p class="sm-text" style="white-space:pre-wrap">' +
        esc(m.readmeSnippet) + '</p></details>'
      : '';
    return (
      '<div class="msg-extra">' +
      '<button type="button" class="link sm-text" data-open-card="' + esc(m.cardUrl) + '">View model card on Hugging Face</button>' +
      snippet +
      '</div>'
    );
  }

  function updateChatStatus(modelId) {
    const loaded = state.chatLoadedModelId === modelId;
    const be = pickBackend(state.hw);
    $('chat-status').textContent = state.chatBusy
      ? 'Generating…'
      : loaded
        ? 'Loaded in memory' + (be ? ' · ' + backendLabel(be) : '')
        : 'Not loaded — send a message to load';
    $('btn-chat-unload').hidden = !loaded;
  }

  function readAdvancedSettings() {
    const ctxVal = $('adv-context-size').value;
    const gpuVal = $('adv-gpu-layers').value;
    return {
      systemPrompt: $('adv-system-prompt').value.trim(),
      contextSize: ctxVal ? parseInt(ctxVal, 10) : null,
      gpuLayers: gpuVal === '' ? 'auto' : gpuVal === 'max' ? 'max' : parseInt(gpuVal, 10),
    };
  }

  function persistChatLog(modelId) {
    rpc().saveChatHistory(modelId, state.chatLogs.get(modelId) || []).catch(function () { /* best-effort */ });
  }

  function loadSettingsEqual(a, b) {
    return !!a && !!b && a.systemPrompt === b.systemPrompt && a.contextSize === b.contextSize && a.gpuLayers === b.gpuLayers;
  }

  async function sendChat(e) {
    e.preventDefault();
    const modelId = state.librarySelected;
    if (!modelId || state.chatBusy) return;
    const input = $('chat-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    const log = state.chatLogs.get(modelId) || [];
    log.push({ role: 'user', text: text });
    state.chatLogs.set(modelId, log);
    renderChatLog(modelId);

    state.chatBusy = true;
    $('btn-chat-send').disabled = true;
    $('btn-chat-stop').hidden = false;
    updateChatStatus(modelId);

    try {
      const loadOpts = readAdvancedSettings();
      const needsLoad = state.chatLoadedModelId !== modelId || !loadSettingsEqual(state.chatLoadedSettings, loadOpts);
      if (needsLoad) {
        $('chat-status').textContent = 'Loading model into memory…';
        const r = await rpc().loadChatModel(modelId, loadOpts);
        if (!r.ok) throw new Error(friendlyError(r.error));
        state.chatLoadedModelId = modelId;
        state.chatLoadedSettings = loadOpts;
      }
      log.push({ role: 'assistant', text: '' });
      renderChatLog(modelId);
      const temperature = parseFloat($('adv-temperature').value);
      const schemaRaw = $('adv-json-schema') && $('adv-json-schema').value.trim();
      let sendOpts = { temperature };
      if (schemaRaw) {
        try {
          sendOpts.jsonSchema = JSON.parse(schemaRaw);
        } catch (pe) {
          throw new Error('JSON schema is not valid JSON.');
        }
      }
      const r2 = await rpc().sendChatMessage(modelId, text, sendOpts);
      if (!r2.ok) throw new Error(r2.error);
      log[log.length - 1].text = r2.data.answer;
      log[log.length - 1].tokensPerSecond = r2.data.tokensPerSecond;
      renderChatLog(modelId);
      touchLastUsed(modelId, 'chat');
    } catch (err) {
      const failedDuringLoad = state.chatLoadedModelId !== modelId;
      const errEntry = { role: 'error', text: friendlyError(err.message || err) };
      if (failedDuringLoad) {
        errEntry.cardUrl = 'https://huggingface.co/' + modelId;
        errEntry.readmeLoading = true;
      }
      log.push(errEntry);
      renderChatLog(modelId);
      if (failedDuringLoad) {
        rpc().getReadme(modelId).then(function (r) {
          errEntry.readmeLoading = false;
          if (r.ok) {
            if (r.data.snippet) errEntry.readmeSnippet = r.data.snippet;
            if (r.data.cardUrl) errEntry.cardUrl = r.data.cardUrl;
          }
          if (state.librarySelected === modelId) renderChatLog(modelId);
          // The earlier finally-block save ran before this resolved and persisted
          // readmeLoading:true — save again now so a reload doesn't show it stuck.
          persistChatLog(modelId);
        }).catch(function () {
          errEntry.readmeLoading = false;
          persistChatLog(modelId);
        });
      }
    } finally {
      state.chatBusy = false;
      $('btn-chat-send').disabled = false;
      $('btn-chat-stop').hidden = true;
      updateChatStatus(modelId);
      persistChatLog(modelId);
    }
  }

  async function unloadChat() {
    const modelId = state.librarySelected;
    await rpc().unloadChatModel();
    state.chatLoadedModelId = null;
    state.chatLoadedSettings = null;
    updateChatStatus(modelId);
    toast('Model unloaded from memory');
  }

  /* ───────── wiring ───────── */

  function wire() {
    document.querySelectorAll('.tab').forEach(function (t) {
      t.onclick = function () { switchView(this.getAttribute('data-view')); };
    });
    $('link-to-explore').onclick = function () { switchView('explore'); };
    if ($('btn-copy-passport')) $('btn-copy-passport').onclick = copyPassport;
    document.querySelectorAll('#job-tiles .job-tile').forEach(function (btn) {
      btn.onclick = function () {
        const pipe = btn.getAttribute('data-pipeline');
        switchView('explore');
        state.exploreLoaded = true;
        loadTags().then(function () {
          $('cat').value = pipe || '';
          $('runnable-only').checked = true;
          $('fits-only').checked = false;
          state.page = 0;
          loadModels();
        });
      };
    });
    $('btn-onboarding-dismiss').onclick = function () {
      $('onboarding').hidden = true;
      try { localStorage.setItem('cyrai_onboarded', '1'); } catch (e) { /* best-effort */ }
    };
    $('btn-rescan').onclick = function () { loadHardware(); loadRecommendations(true); };

    $('btn-go').onclick = function () { state.page = 0; loadModels(); };
    $('q').onkeydown = function (e) { if (e.key === 'Enter') { state.page = 0; loadModels(); } };
    $('cat').onchange = function () { state.page = 0; loadModels(); };
    $('sort').onchange = function () { state.page = 0; loadModels(); };
    $('lib').onchange = function () { state.page = 0; loadModels(); };
    const onAnalysisFilterChange = function () {
      const needsCheck = this.checked && state.models.some(function (m) { return !state.analyses.has(m.id); });
      if (needsCheck) checkPage();
      else renderModels();
    };
    $('fits-only').onchange = onAnalysisFilterChange;
    // Unlike fits-only, toggling this changes the underlying search query (GGUF
    // bias) — re-checking the already-fetched page isn't enough, it needs a refetch.
    $('runnable-only').onchange = function () { state.page = 0; loadModels(); };
    $('btn-check').onclick = function () { checkPage(); };
    $('prev').onclick = function () { if (state.page > 0) { state.page--; loadModels(); } };
    $('next').onclick = function () { state.page++; loadModels(); };
    $('btn-cached').onclick = function () { showCached(); };

    $('btn-open-models').onclick = function () { rpc().openDir(null); };
    $('chat-form').onsubmit = sendChat;
    $('btn-chat-stop').onclick = function () {
      if (state.librarySelected) rpc().stopGeneration(state.librarySelected);
    };
    $('btn-chat-unload').onclick = unloadChat;
    $('adv-temperature').oninput = function () { $('adv-temp-val').textContent = this.value; };
    $('btn-cancel-dl').onclick = function () {
      if (state.downloadingId) rpc().cancelDownload(state.downloadingId);
    };
    $('btn-compare-clear').onclick = function () {
      state.compareIds = [];
      renderCompare();
      if (state.selected) selectModel(state.selected);
    };

    try {
      rpc().onGenerateProgress(function (p) {
        const logEl = $('gen-log');
        if (!logEl || !p || !p.text) return;
        logEl.hidden = false;
        logEl.textContent += p.text + (/\n$/.test(p.text) ? '' : '\n');
        logEl.scrollTop = logEl.scrollHeight;
      });
      function hearProgress(p) {
        if (!p || !p.text) return;
        appendHearLog(p.text);
      }
      rpc().onTranscribeProgress(hearProgress);
      rpc().onSpeakProgress(hearProgress);
      rpc().onDownloadProgress(function (p) {
        const wrap = $('dl-progress');
        const bar = document.querySelector('#dl-progress > i');
        const row = $('dl-status-row');
        const txt = $('dl-txt');
        if (wrap) wrap.hidden = false;
        if (row) row.hidden = false;
        if (bar) bar.style.width = (p.percent || 0) + '%';

        const speed = !p.done ? trackDownloadSpeed(p) : '';
        const resume = !p.done ? resumeNote(p) : '';
        const sizeBit = (p.received != null && p.total)
          ? formatDlSize(p.received) + ' / ' + formatDlSize(p.total)
          : '';
        if (txt) {
          txt.textContent = p.done
            ? 'Done'
            : resume + (p.file || '') + ' ' + (p.percent || 0) + '% (' + ((p.index || 0) + 1) + '/' + (p.totalFiles || 1) + ')' +
              (sizeBit ? ' · ' + sizeBit : '') +
              (speed ? ' · ' + speed : '');
        }
        if (p.done) {
          setTimeout(function () {
            if (wrap) wrap.hidden = true;
            if (row) row.hidden = true;
            if (txt) txt.textContent = '';
          }, 1500);
        }
        const card = document.querySelector('.reco-card[data-id="' + (p.modelId || '').replace(/"/g, '') + '"] .why');
        if (card && !p.done) {
          card.textContent = (resume || 'Downloading… ') + (p.percent || 0) + '%' +
            (sizeBit ? ' · ' + sizeBit : '') +
            (speed ? ' · ' + speed : '');
        }
      });
    } catch (e) { /* ignore */ }

    try {
      rpc().onChatChunk(function (data) {
        if (!data || data.modelId !== state.librarySelected) return;
        const log = state.chatLogs.get(data.modelId);
        if (!log || !log.length) return;
        const last = log[log.length - 1];
        if (last.role !== 'assistant') return;
        last.text += data.text;
        renderChatLog(data.modelId);
      });
    } catch (e) { /* ignore */ }
  }

  function reduceMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  function hideGate() {
    const gate = $('gate');
    if (!gate || gate.hidden) return;
    document.body.classList.remove('gate-open');
    gate.hidden = true;
    const vid = $('gate-seq');
    if (vid) vid.pause();
    const homeTab = document.querySelector('.tab[data-view="home"]');
    if (homeTab) homeTab.focus();
  }

  function wireGate() {
    const gate = $('gate');
    const vid = $('gate-seq');
    const wrap = $('gate-seq-wrap');
    const fill = $('gate-scrub-fill');
    const enterBtn = $('btn-gate-enter');
    const skipBtn = $('btn-gate-skip');
    if (!gate || !enterBtn) return;
    if (location.hash === '#selftest') {
      hideGate();
      return;
    }

    let entering = false;
    let closing = false;
    let progress = 0;
    let raf = 0;
    const reduce = reduceMotion();

    function paint(p) {
      progress = Math.max(0, Math.min(1, p));
      if (fill) fill.style.transform = 'scaleX(' + progress + ')';
      if (vid && vid.duration && !entering && !reduce) {
        const t = progress * Math.max(0, vid.duration - 0.04);
        if (Math.abs(vid.currentTime - t) > 0.03) vid.currentTime = t;
      }
    }

    function onPointer(e) {
      if (entering || reduce) return;
      const r = gate.getBoundingClientRect();
      const nx = (e.clientX - r.left) / r.width;
      const ny = (e.clientY - r.top) / r.height;
      if (wrap) {
        wrap.style.setProperty('--ry', ((0.5 - nx) * 14).toFixed(2) + 'deg');
        wrap.style.setProperty('--rx', ((ny - 0.5) * 8).toFixed(2) + 'deg');
      }
      if (!raf) {
        raf = requestAnimationFrame(function () {
          raf = 0;
          paint(nx);
        });
      }
    }

    function onWheel(e) {
      if (entering || reduce) return;
      e.preventDefault();
      paint(progress + e.deltaY * 0.0007);
    }

    function finishEnter() {
      if (closing || gate.hidden) return;
      closing = true;
      gate.classList.add('gate-out');
      window.setTimeout(hideGate, reduce ? 0 : 700);
    }

    function enter() {
      if (entering) return;
      entering = true;
      enterBtn.disabled = true;
      enterBtn.setAttribute('aria-busy', 'true');
      if (reduce || !vid || !vid.duration) {
        finishEnter();
        return;
      }
      const remain = vid.duration - vid.currentTime;
      if (remain > 2.2) vid.currentTime = vid.duration - 2.2;
      vid.playbackRate = 1.25;
      const done = function () {
        vid.removeEventListener('ended', done);
        finishEnter();
      };
      vid.addEventListener('ended', done);
      vid.play().catch(finishEnter);
      window.setTimeout(done, 3200);
    }

    function skip() {
      if (entering) return;
      entering = true;
      finishEnter();
    }

    gate.addEventListener('pointermove', onPointer, { passive: true });
    gate.addEventListener('wheel', onWheel, { passive: false });
    enterBtn.addEventListener('click', enter);
    if (skipBtn) skipBtn.addEventListener('click', skip);
    document.addEventListener('keydown', function (e) {
      if (gate.hidden || entering) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        skip();
      }
    });
    if (vid) {
      vid.muted = true;
      vid.addEventListener('loadedmetadata', function () {
        if (!entering && !reduce) paint(0.08);
      });
      // Electron file:// will not paint a frame until play() has run once.
      vid.play().then(function () {
        if (!entering) vid.pause();
      }).catch(function () { /* poster still shows */ });
    }
    window.setTimeout(function () {
      if (!gate.hidden) enterBtn.focus();
    }, 40);
  }

  async function init() {
    try {
      document.body.classList.add('ready');
      wireGate();
      if (!window.cyrai) {
        $('hero-sub').textContent = 'App bridge failed. Restart with START.bat';
        return;
      }
      wire();
      switchView('home');
      try {
        if (!localStorage.getItem('cyrai_onboarded')) $('onboarding').hidden = false;
      } catch (e) { /* localStorage unavailable — skip onboarding banner, not fatal */ }
      await refreshNet();
      await loadHardware();
      await loadLocal();
      await loadRecommendations();
      setInterval(refreshNet, 30000);
    } catch (e) {
      toast(friendlyError(e.message || e));
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
