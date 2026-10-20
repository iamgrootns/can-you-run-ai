/**
 * Headless MVP smoke test (no Electron UI).
 * Run: node scripts/smoke.js
 * Optional: node scripts/smoke.js --download
 */
const si = require('systeminformation');
const fs = require('fs');
const path = require('path');
const os = require('os');
const https = require('https');
const http = require('http');

const HF_API = 'https://huggingface.co/api';
const cacheDir = path.join(os.tmpdir(), 'cyrai-smoke');
const wantDl = process.argv.includes('--download');
fs.mkdirSync(cacheDir, { recursive: true });

async function fetchJson(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'CanYouRunAI-smoke/1' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${url}`);
  return res.json();
}

function encodeRepo(id) {
  return id.split('/').map(encodeURIComponent).join('/');
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const go = (u, n = 0) => {
      if (n > 10) return reject(new Error('redirects'));
      const parsed = new URL(u);
      const lib = parsed.protocol === 'http:' ? http : https;
      lib
        .get(parsed, { headers: { 'User-Agent': 'CanYouRunAI-smoke/1' }, timeout: 120000 }, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            res.resume();
            return go(new URL(res.headers.location, parsed).href, n + 1);
          }
          if (res.statusCode !== 200) {
            res.resume();
            return reject(new Error('HTTP ' + res.statusCode));
          }
          fs.mkdirSync(path.dirname(dest), { recursive: true });
          const out = fs.createWriteStream(dest);
          res.pipe(out);
          out.on('finish', () => out.close(() => resolve(fs.statSync(dest).size)));
          out.on('error', reject);
        })
        .on('error', reject);
    };
    go(url);
  });
}

async function main() {
  const log = (s) => console.log('✓', s);
  const fail = (s) => {
    console.error('✗', s);
    process.exitCode = 1;
  };

  const [cpu, mem, graphics] = await Promise.all([si.cpu(), si.mem(), si.graphics()]);
  const gpus = (graphics.controllers || []).map((g) => `${g.model} (${g.vram || '?'}MB)`);
  log(`CPU ${cpu.brand} · RAM ${(mem.total / 1024 ** 3).toFixed(1)}GB · GPU ${gpus.join(' | ') || 'none'}`);

  const tags = await fetchJson(`${HF_API}/models-tags-by-type`);
  const nTags = (tags.pipeline_tag || []).length;
  if (nTags < 10) fail(`expected many pipeline tags, got ${nTags}`);
  else log(`${nTags} live pipeline tags`);
  fs.writeFileSync(path.join(cacheDir, 'tags.json'), JSON.stringify({ tags: tags.pipeline_tag }));

  const models = await fetchJson(
    `${HF_API}/models?limit=5&sort=downloads&pipeline_tag=text-generation&full=true`
  );
  if (!Array.isArray(models) || !models.length) fail('no models');
  else log(`sample models: ${models.map((m) => m.id).join(', ')}`);

  const id = models.find((m) => !m.gated)?.id || models[0].id;
  const tree = await fetchJson(`${HF_API}/models/${encodeRepo(id)}/tree/main?recursive=true`);
  const weights = (Array.isArray(tree) ? tree : []).filter(
    (f) => f.type === 'file' && /\.(safetensors|gguf)$/i.test(f.path)
  );
  const bytes = weights.reduce((s, f) => s + (f.lfs?.size || f.size || 0), 0);
  const loadGb = (bytes / 1024 ** 3) * 1.2;
  log(`${id}: ${weights.length} weight files, est load ~${loadGb.toFixed(2)} GB`);

  const totalVram = Math.max(0, ...(graphics.controllers || []).map((g) => g.vram || 0)) / 1024;
  const maxVram = totalVram * 0.9;
  const status =
    loadGb <= maxVram * 0.7
      ? 'excellent'
      : loadGb <= maxVram
        ? 'good'
        : loadGb <= maxVram * 1.15
          ? 'tight'
          : 'unlikely';
  log(`match on ${totalVram.toFixed(1)}GB VRAM → ${status}`);

  const cached = JSON.parse(fs.readFileSync(path.join(cacheDir, 'tags.json'), 'utf8'));
  if (!cached.tags?.length) fail('cache readback failed');
  else log(`offline cache ok (${cached.tags.length} tags)`);

  if (wantDl) {
    const tiny = 'sshleifer/tiny-gpt2';
    const t = await fetchJson(`${HF_API}/models/${encodeRepo(tiny)}/tree/main`);
    const bin = t.find((f) => f.type === 'file' && /pytorch_model\.bin$/i.test(f.path));
    if (!bin) fail('tiny model weight missing');
    else {
      const dest = path.join(cacheDir, 'models', tiny.replace('/', '__'), path.basename(bin.path));
      const url = `https://huggingface.co/${tiny}/resolve/main/${bin.path}`;
      const sz = await downloadFile(url, dest);
      fs.writeFileSync(
        path.join(path.dirname(dest), 'manifest.json'),
        JSON.stringify({ modelId: tiny, files: [{ path: bin.path, sizeBytes: sz }] })
      );
      log(`offline download ok ${sz} bytes (${tiny})`);
    }
  }

  console.log(process.exitCode ? '\nSMOKE FAILED' : '\nSMOKE OK — MVP core paths work');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
