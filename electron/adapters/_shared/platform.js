'use strict';
/**
 * Cross-platform helpers shared by the native-binary adapters (sd, whisper,
 * piper). Factored out because all three previously duplicated an
 * almost-identical download/hash/extract block that only ever worked on
 * Windows (PowerShell Expand-Archive, .exe paths). Real upstream releases
 * for all three projects also ship Linux builds (and piper/sd ship macOS
 * builds too) — verified against the real GitHub Releases API, not assumed.
 */
const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const https = require('https');
const { spawn } = require('child_process');

/** 'win' | 'mac' | 'linux'. Anything else (freebsd, etc.) falls through to 'linux' handling, which is the closest match for a posix native binary. */
function osKey() {
  if (process.platform === 'win32') return 'win';
  if (process.platform === 'darwin') return 'mac';
  return 'linux';
}

/** 'x64' | 'arm64'. 32-bit (ia32/arm) is deliberately not mapped — none of these upstreams ship a build for it, and guessing one would just fail later at spawn time instead of with a clear reason now. */
function archKey() {
  if (process.arch === 'arm64') return 'arm64';
  if (process.arch === 'x64') return 'x64';
  return process.arch;
}

/** e.g. 'linux-x64', 'mac-arm64', 'win-x64' — the natural key for a per-platform ASSETS map. */
function platformArchKey() {
  return `${osKey()}-${archKey()}`;
}

function exeSuffix() {
  return osKey() === 'win' ? '.exe' : '';
}

/**
 * Env override to make a bundled-alongside-the-binary shared library
 * findable. Windows resolves sibling DLLs via the process's cwd (which every
 * caller already sets to the binary's own directory) — nothing extra needed.
 * Linux/macOS do not search cwd for shared libraries, so without this every
 * extracted binary here would fail to start with a missing libggml.so /
 * libpiper_phonemize.so / etc. even though the file sits right next to it
 * (confirmed by actually extracting and running whisper-cli/sd-cli/piper on
 * Linux during development — they do not start without this).
 */
function libraryPathEnv(dir) {
  if (osKey() === 'win') return {};
  const key = osKey() === 'mac' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
  const existing = process.env[key];
  return { [key]: existing ? `${dir}:${existing}` : dir };
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('data', (d) => hash.update(d));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function downloadFile(url, dest, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 10) return reject(new Error('Too many redirects'));
    https.get(url, { headers: { 'User-Agent': 'CanYouRunAI/1.0' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return resolve(downloadFile(res.headers.location, dest, redirects + 1));
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`Download HTTP ${res.statusCode} for ${url}`));
      }
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      const out = fs.createWriteStream(dest);
      res.pipe(out);
      out.on('finish', () => out.close(() => resolve()));
      out.on('error', reject);
    }).on('error', reject);
  });
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args);
    let stderr = '';
    proc.stderr.on('data', (d) => { stderr += d.toString(); });
    proc.on('error', reject);
    proc.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}: ${stderr || '(no stderr)'}`))));
  });
}

/**
 * Extracts a .zip or .tar.gz/.tgz into destDir. Windows uses PowerShell's
 * Expand-Archive (built in, no extra dependency) for .zip and bsdtar
 * (ships as tar.exe on Windows 10 1803+ / Windows 11) for .tar.gz.
 * Linux/macOS use the system `unzip`/`tar` — both are effectively always
 * present (coreutils-level tools), so this avoids pulling in a JS zip
 * library for what upstream already gives every target OS a CLI for.
 */
async function extractArchive(archivePath, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  const isZip = /\.zip$/i.test(archivePath);
  if (osKey() === 'win') {
    if (isZip) {
      await run('powershell.exe', [
        '-NoProfile', '-NonInteractive', '-Command',
        `Expand-Archive -Path '${archivePath}' -DestinationPath '${destDir}' -Force`,
      ]);
    } else {
      await run('tar.exe', ['-xzf', archivePath, '-C', destDir]);
    }
    return;
  }
  if (isZip) {
    try {
      await run('unzip', ['-o', archivePath, '-d', destDir]);
      return;
    } catch (e) {
      // Fall through to tar: modern bsdtar (macOS default, many Linux
      // distros) auto-detects zip format even without a .tar.gz extension.
      try {
        await run('tar', ['-xf', archivePath, '-C', destDir]);
        return;
      } catch {
        throw e; // report the original unzip error, it's the more informative one
      }
    }
  }
  await run('tar', ['-xzf', archivePath, '-C', destDir]);
}

/** chmod +x — archive extraction doesn't reliably preserve the exec bit across every zip/tar implementation. No-op on Windows. */
async function makeExecutable(filePath) {
  if (osKey() === 'win') return;
  await fsp.chmod(filePath, 0o755);
}

/**
 * electron-builder's asarUnpack extracts matching files to a real
 * `app.asar.unpacked/` directory alongside the (still virtual) `app.asar`
 * archive — but that transparent asar->real-file redirection is an Electron
 * patch on Node's own `fs` module only. A path handed to `child_process.spawn`
 * goes straight to the OS with no such patch, so an external process (the
 * Python worker scripts here) literally cannot open a path still pointing
 * inside app.asar, even though `fs.existsSync` on that same path from this
 * process would happily say true. Anything passed to `spawn()` as a script
 * path for an asarUnpack'd file must go through this first. No-op outside a
 * packaged build (dev mode has no asar at all, path is already real).
 */
function unpackedPath(p) {
  const marker = `${path.sep}app.asar${path.sep}`;
  return p.includes(marker) ? p.replace(marker, `${path.sep}app.asar.unpacked${path.sep}`) : p;
}

module.exports = {
  osKey, archKey, platformArchKey, exeSuffix, libraryPathEnv,
  sha256File, downloadFile, extractArchive, makeExecutable, unpackedPath,
};
