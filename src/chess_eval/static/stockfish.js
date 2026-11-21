export function parseStockfishInfo(line, turn = "white") {
  if (!line.startsWith("info ") || !line.includes(" pv ")) return null;
  const depth = line.match(/ depth (\d+)/);
  const seldepth = line.match(/ seldepth (\d+)/);
  const nodes = line.match(/ nodes (\d+)/);
  const nps = line.match(/ nps (\d+)/);
  const cp = line.match(/ score cp (-?\d+)/);
  const mate = line.match(/ score mate (-?\d+)/);
  if (!cp && !mate) return null;
  const pv = line.match(/ pv (.+)$/);
  const scoreType = cp ? "cp" : "mate";
  let scoreValue = Number(cp ? cp[1] : mate[1]);
  if (turn === "black") scoreValue = -scoreValue;
  return {
    depth: Number(depth?.[1] || 0),
    seldepth: Number(seldepth?.[1] || 0),
    nodes: Number(nodes?.[1] || 0),
    nps: Number(nps?.[1] || 0),
    scoreType,
    scoreValue,
    pv: pv?.[1] || "",
  };
}

export class StockfishController {
  constructor({ onReady, onLine, onError, onProgress } = {}) {
    this.onReady = onReady;
    this.onLine = onLine;
    this.onError = onError;
    this.onProgress = onProgress;
    this.ready = false;
    this.searching = false;
    this.pending = null;
    this.worker = new Worker(new URL("./stockfish-worker.js", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event) => this.handleMessage(event.data);
    this.worker.onerror = (event) =>
      this.onError?.(event.message || "Stockfish worker failed");
    this.worker.postMessage({ type: "init" });
  }

  handleMessage(message) {
    if (!message || !message.type) return;
    if (message.type === "ready") {
      this.ready = true;
      this.onReady?.(message);
      if (this.pending) {
        const pending = this.pending;
        this.pending = null;
        this.start(pending.fen, pending.depth);
      }
      return;
    }
    if (message.type === "line") {
      this.onLine?.(message.line);
      if (message.line.startsWith("bestmove ")) {
        this.searching = false;
        if (this.pending) {
          const pending = this.pending;
          this.pending = null;
          this.start(pending.fen, pending.depth);
        }
      }
      return;
    }
    if (message.type === "error") {
      this.onError?.(message.message);
      return;
    }
    if (message.type === "nnue-progress") {
      this.onProgress?.(message);
      return;
    }
    if (message.type === "exit") {
      this.onError?.(`Stockfish exited with code ${message.code}`);
    }
  }

  evaluate(fen, depth) {
    if (!this.ready) {
      this.pending = { fen, depth };
      return;
    }
    if (this.searching) {
      this.pending = { fen, depth };
      this.worker.postMessage({ type: "command", command: "stop" });
      return;
    }
    this.start(fen, depth);
  }

  start(fen, depth) {
    this.searching = true;
    this.worker.postMessage({
      type: "command",
      command: `position fen ${fen}`,
    });
    this.worker.postMessage({ type: "command", command: `go depth ${depth}` });
  }

  stop() {
    this.pending = null;
    if (this.searching) {
      this.worker.postMessage({ type: "command", command: "stop" });
    }
  }

  dispose() {
    this.worker.postMessage({ type: "quit" });
    this.worker.terminate();
  }
}
