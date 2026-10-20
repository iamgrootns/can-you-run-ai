'use strict';
/**
 * Runner host — Claude-owned (see Desktop/mixedplan.md §11).
 *
 * Spawns one child process per adapter worker and normalizes events to the
 * PIPE_EVENTS shape in contract.js, regardless of transport. For a
 * Node-based adapter (llama, via node-llama-cpp) this uses child_process
 * .fork(), which gives real OS-process crash isolation plus a structured
 * message channel for free — no reason to hand-roll NDJSON framing over a
 * named pipe for something that's already a Node process talking to a Node
 * parent. Native-binary adapters (sd.cpp, whisper.cpp in later phases) are
 * NOT Node processes and will need an actual stdio/socket transport — this
 * module's public surface (spawn/on/off/send/health/kill) is written so
 * that swap stays inside this file and never ripples into registry.js,
 * scheduler.js, or main.js.
 */
const { fork } = require('child_process');
const { PIPE_EVENTS } = require('./contract');

const workers = new Map(); // workerId -> { proc, listeners:Set, alive }

function spawn(workerId, modulePath, args = []) {
  if (workers.has(workerId) && workers.get(workerId).alive) {
    throw new Error(`worker ${workerId} already running`);
  }
  const proc = fork(modulePath, args, { stdio: ['pipe', 'pipe', 'pipe', 'ipc'] });
  const entry = { proc, listeners: new Set(), alive: true };
  workers.set(workerId, entry);

  proc.on('message', (msg) => {
    if (!msg || !PIPE_EVENTS.includes(msg.type)) return;
    for (const fn of entry.listeners) fn(msg);
  });
  proc.on('exit', (code, signal) => {
    entry.alive = false;
    // A crash mid-job must be a reported 'exit' event, not silence — callers
    // (the llama adapter's pending load()/send() promises) need this to
    // reject cleanly instead of hanging forever.
    for (const fn of entry.listeners) fn({ type: 'exit', code, signal });
    workers.delete(workerId);
  });
  proc.stderr?.on('data', (d) => {
    for (const fn of entry.listeners) fn({ type: 'error', message: d.toString() });
  });
  return entry;
}

function on(workerId, fn) {
  workers.get(workerId)?.listeners.add(fn);
  return fn;
}

function off(workerId, fn) {
  workers.get(workerId)?.listeners.delete(fn);
}

function send(workerId, msg) {
  const entry = workers.get(workerId);
  if (!entry?.alive) throw new Error(`worker ${workerId} not running`);
  entry.proc.send(msg);
}

function health(workerId) {
  const entry = workers.get(workerId);
  return { running: !!entry?.alive, pid: entry?.proc?.pid ?? null };
}

function kill(workerId) {
  const entry = workers.get(workerId);
  if (!entry) return false;
  entry.proc.kill();
  workers.delete(workerId);
  return true;
}

module.exports = { spawn, on, off, send, health, kill };
