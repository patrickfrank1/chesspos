import Stockfish from "./stockfish/sf_19.js";

let engine = null;
let initializing = false;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function loadNnue() {
  const response = await fetch("/api/stockfish/nnue", { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`NNUE download failed (${response.status})`);
  }
  const total = Number(response.headers.get("content-length") || 0);
  if (!response.body) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    post("nnue-progress", { received: buffer.byteLength, total: buffer.byteLength });
    return buffer;
  }
  const reader = response.body.getReader();
  const buffer = total > 0 ? new Uint8Array(total) : null;
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buffer) buffer.set(value, received);
    else chunks.push(value);
    received += value.length;
    post("nnue-progress", { received, total });
  }
  if (buffer) return buffer;
  const joined = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  return joined;
}

async function init() {
  if (engine || initializing) return;
  initializing = true;
  try {
    engine = await Stockfish({
      listen: (line) => {
        post("line", { line });
        if (line === "readyok") post("ready");
      },
      onError: (message) => post("error", { message }),
      onExit: (code) => post("exit", { code }),
    });
    const nnue = await loadNnue();
    engine.setNnueBuffer(nnue, 0);
    engine.uci("uci");
    engine.uci("setoption name Threads value 1");
    engine.uci("setoption name Hash value 32");
    engine.uci("isready");
  } catch (error) {
    post("error", { message: error?.message || String(error) });
  } finally {
    initializing = false;
  }
}

self.onmessage = async (event) => {
  const message = event.data || {};
  if (message.type === "init") {
    await init();
    return;
  }
  if (!engine) {
    post("error", { message: "Stockfish is not initialized" });
    return;
  }
  if (message.type === "command") {
    engine.uci(message.command);
    return;
  }
  if (message.type === "quit") {
    engine.uci("quit");
    self.close();
  }
};
