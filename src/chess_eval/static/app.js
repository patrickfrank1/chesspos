import { StockfishController, parseStockfishInfo } from "./stockfish.js";

const START_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const PIECES = {
  P: "♙",
  N: "♘",
  B: "♗",
  R: "♖",
  Q: "♕",
  K: "♔",
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛",
  k: "♚",
};

const state = {
  models: [],
  selected: new Set(),
  samples: [],
  sampleMap: new Map(),
  startFen: START_FEN,
  moves: [],
  position: null,
  evaluations: [],
  orientation: "white",
  selectedSquare: null,
  busy: false,
  stockfish: {
    controller: null,
    ready: false,
    error: null,
    info: null,
    bestmove: null,
    progress: null,
    thinking: false,
  },
};

let boardRequestId = 0;

const elements = {
  board: document.getElementById("board"),
  deviceLine: document.getElementById("device-line"),
  statusLine: document.getElementById("status-line"),
  materialLine: document.getElementById("material-line"),
  fenInput: document.getElementById("fen-input"),
  sampleSelect: document.getElementById("sample-select"),
  moveList: document.getElementById("move-list"),
  evalCards: document.getElementById("eval-cards"),
  evalStatus: document.getElementById("eval-status"),
  modelList: document.getElementById("model-list"),
  modelCount: document.getElementById("model-count"),
  promotion: document.getElementById("promotion-select"),
  toast: document.getElementById("toast"),
  sfDot: document.getElementById("sf-dot"),
  sfEval: document.getElementById("sf-eval"),
  sfMeta: document.getElementById("sf-meta"),
  sfPv: document.getElementById("sf-pv"),
  sfDepth: document.getElementById("sf-depth"),
  sfAuto: document.getElementById("sf-auto"),
  sfRun: document.getElementById("sf-run"),
};

function pieceColor(piece) {
  return piece === piece.toUpperCase() ? "white" : "black";
}

function parseFen(fen) {
  const [placement] = String(fen).split(" ");
  const board = {};
  if (!placement) return board;
  let rank = 7;
  let file = 0;
  for (const char of placement) {
    if (char === "/") {
      rank -= 1;
      file = 0;
      continue;
    }
    if (/[1-8]/.test(char)) {
      file += Number(char);
      continue;
    }
    if (file < 8 && rank >= 0) {
      board[FILES[file] + RANKS[rank]] = char;
    }
    file += 1;
  }
  return board;
}

function fenTurn(fen) {
  const parts = String(fen).split(" ");
  return parts[1] === "b" ? "black" : "white";
}

function squareName(fileIndex, rankIndex) {
  return FILES[fileIndex] + RANKS[rankIndex];
}

function displaySquares() {
  const squares = [];
  const ranks =
    state.orientation === "white"
      ? [7, 6, 5, 4, 3, 2, 1, 0]
      : [0, 1, 2, 3, 4, 5, 6, 7];
  const orderedFiles =
    state.orientation === "white"
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];
  for (const rankIndex of ranks) {
    for (const fileIndex of orderedFiles) {
      squares.push({
        name: squareName(fileIndex, rankIndex),
        fileIndex,
        rankIndex,
      });
    }
  }
  return squares;
}

function legalTargetsFrom(source) {
  if (!state.position) return new Set();
  return new Set(
    state.position.legal_moves
      .filter((move) => move.startsWith(source))
      .map((move) => move.slice(2, 4)),
  );
}

function isOwnPiece(piece, turn) {
  return Boolean(piece) && pieceColor(piece) === turn;
}

function findKingSquare(board, color) {
  const king = color === "white" ? "K" : "k";
  return Object.keys(board).find((square) => board[square] === king) || null;
}

function renderBoard() {
  const fen = state.position ? state.position.fen : state.startFen;
  const board = parseFen(fen);
  const turn = state.position ? state.position.turn : fenTurn(fen);
  const lastMove = state.position ? state.position.last_move : null;
  const lastSquares = new Set();
  if (lastMove && lastMove.length >= 4) {
    lastSquares.add(lastMove.slice(0, 2));
    lastSquares.add(lastMove.slice(2, 4));
  }
  const targets = state.selectedSquare
    ? legalTargetsFrom(state.selectedSquare)
    : new Set();
  const checkSquare =
    state.position && state.position.is_check
      ? findKingSquare(board, turn)
      : null;

  elements.board.innerHTML = "";
  const squares = displaySquares();
  for (const square of squares) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "square";
    button.dataset.square = square.name;
    const isLight = (square.fileIndex + square.rankIndex) % 2 === 1;
    button.classList.add(isLight ? "light" : "dark");
    if (lastSquares.has(square.name)) button.classList.add("last-move");
    if (state.selectedSquare === square.name) button.classList.add("selected");
    if (targets.has(square.name)) {
      button.classList.add("legal");
      if (board[square.name]) button.classList.add("capture");
    }
    if (checkSquare === square.name) button.classList.add("in-check");

    const piece = board[square.name];
    if (piece) {
      const span = document.createElement("span");
      span.className = `piece ${pieceColor(piece)}`;
      span.textContent = PIECES[piece] || "?";
      button.appendChild(span);
    }

    if (state.orientation === "white") {
      if (square.rankIndex === 0) {
        const coord = document.createElement("span");
        coord.className = "coord file";
        coord.textContent = FILES[square.fileIndex];
        button.appendChild(coord);
      }
      if (square.fileIndex === 0) {
        const coord = document.createElement("span");
        coord.className = "coord rank";
        coord.textContent = RANKS[square.rankIndex];
        button.appendChild(coord);
      }
    } else {
      if (square.rankIndex === 7) {
        const coord = document.createElement("span");
        coord.className = "coord file";
        coord.textContent = FILES[square.fileIndex];
        button.appendChild(coord);
      }
      if (square.fileIndex === 7) {
        const coord = document.createElement("span");
        coord.className = "coord rank";
        coord.textContent = RANKS[square.rankIndex];
        button.appendChild(coord);
      }
    }

    button.addEventListener("click", () => onSquareClick(square.name));
    elements.board.appendChild(button);
  }
}

function statusText(position) {
  if (!position) return "Loading…";
  if (position.is_checkmate) {
    return `Checkmate — ${position.turn === "white" ? "Black" : "White"} wins`;
  }
  if (position.is_stalemate) return "Stalemate";
  if (position.status === "insufficient material") return "Draw — insufficient material";
  if (position.status === "seventy-five-move rule") return "Draw — 75-move rule";
  if (position.status === "fivefold repetition") return "Draw — fivefold repetition";
  const turn = position.turn === "white" ? "White" : "Black";
  if (position.is_check) return `${turn} to move — check`;
  return `${turn} to move`;
}

function renderStatus() {
  const position = state.position;
  elements.statusLine.innerHTML = statusText(position)
    .replace("check", '<span class="danger">check</span>')
    .replace("Checkmate", '<span class="danger">Checkmate</span>');
  if (position) {
    const balance = position.material_balance;
    const text =
      balance === 0 ? "material even" : `material ${balance > 0 ? "+" : ""}${balance} White`;
    elements.materialLine.textContent = text;
    elements.fenInput.value = position.fen;
  }
}

function renderMoves() {
  const position = state.position;
  const sans = position ? position.san_moves : [];
  elements.moveList.innerHTML = "";
  if (!sans.length) {
    elements.moveList.textContent = "No moves yet";
    return;
  }
  const start = document.createElement("button");
  start.type = "button";
  start.className = "move-chip";
  start.textContent = "start";
  start.addEventListener("click", () => {
    state.moves = [];
    state.selectedSquare = null;
    refreshPosition();
  });
  elements.moveList.appendChild(start);
  sans.forEach((san, index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "move-chip";
    chip.textContent = san;
    chip.addEventListener("click", () => {
      state.moves = state.moves.slice(0, index + 1);
      state.selectedSquare = null;
      refreshPosition();
    });
    elements.moveList.appendChild(chip);
  });
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

function metric(name, value, digits = 3) {
  return `<span class="metric">${name} <strong>${formatNumber(value, digits)}</strong></span>`;
}

function renderEvaluations() {
  elements.evalCards.innerHTML = "";
  if (!state.selected.size) {
    elements.evalCards.innerHTML =
      '<div class="empty-card">Select one or more checkpoints to evaluate the position.</div>';
    return;
  }
  if (!state.evaluations.length) {
    elements.evalCards.innerHTML =
      '<div class="empty-card">Evaluation pending…</div>';
    return;
  }
  for (const evaluation of state.evaluations) {
    const card = document.createElement("div");
    card.className = "eval-card";
    if (evaluation.error) {
      card.innerHTML = `<div class="error-card"><strong>${escapeHtml(
        evaluation.label || evaluation.model_id,
      )}</strong><br />${escapeHtml(evaluation.error)}</div>`;
      elements.evalCards.appendChild(card);
      continue;
    }
    const isScore = evaluation.task === "score";
    const value = evaluation.value;
    const badge = isScore ? "score" : "sacrifice";
    const bigValue = isScore
      ? `${value >= 0 ? "+" : ""}${formatNumber(value, 2)}`
      : `${formatNumber(evaluation.percent, 1)}%`;
    const sub = isScore
      ? `White expected points ${formatNumber(evaluation.expected_points * 100, 1)}%`
      : evaluation.threshold !== null && evaluation.threshold !== undefined
        ? `${evaluation.above_threshold ? "above" : "below"} threshold ${formatNumber(
            evaluation.threshold,
            2,
          )}`
        : "natural-prevalence probability";
    const barPercent = isScore
      ? ((value + 1) / 2) * 100
      : Math.max(0, Math.min(100, evaluation.percent));
    const markerPercent = isScore ? 50 : (evaluation.threshold || 0) * 100;
    const metrics = isScore
      ? [
          metric("R²", evaluation.metrics?.r2),
          metric("AUC", evaluation.metrics?.auroc_decisive),
          metric("MAE", evaluation.metrics?.mae),
        ].join("")
      : [
          metric("PR-AUC", evaluation.metrics?.pr_auc),
          metric("AUROC", evaluation.metrics?.auroc),
          metric("F1", evaluation.metrics?.f1),
        ].join("");

    card.innerHTML = `
      <div class="card-top">
        <div>
          <div class="model-name">${escapeHtml(evaluation.label)}</div>
          <div class="eval-sub">${escapeHtml(evaluation.model || "")}</div>
        </div>
        <span class="task-badge ${badge}">${badge}</span>
      </div>
      <div class="eval-value">${bigValue}</div>
      <div class="eval-sub">${sub}</div>
      <div class="bar">
        <div class="fill" style="width:${barPercent}%"></div>
        ${
          isScore
            ? `<div class="marker" style="left:${markerPercent}%"></div>`
            : evaluation.threshold !== null && evaluation.threshold !== undefined
              ? `<div class="marker" style="left:${markerPercent}%"></div>`
              : ""
        }
      </div>
      <div class="metric-row">${metrics}</div>
    `;
    elements.evalCards.appendChild(card);
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setEvalStatus(text) {
  elements.evalStatus.textContent = text || "";
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.detail || response.statusText);
  }
  return payload;
}

function applyServerState(payload) {
  state.position = payload.position;
  state.evaluations = payload.evaluations || [];
  state.startFen = payload.position.start_fen;
  state.moves = payload.position.moves;
  state.selectedSquare = null;
  renderBoard();
  renderStatus();
  renderMoves();
  renderEvaluations();
  if (elements.sfAuto.checked) {
    runStockfish();
  } else {
    state.stockfish.controller?.stop();
    state.stockfish.info = null;
    state.stockfish.bestmove = null;
    state.stockfish.thinking = false;
    renderStockfish();
  }
}

async function refreshPosition() {
  const requestId = ++boardRequestId;
  setEvalStatus("evaluating…");
  try {
    const payload = await api("/api/position", {
      start_fen: state.startFen,
      moves: state.moves,
      model_ids: [...state.selected],
    });
    if (requestId !== boardRequestId) return;
    applyServerState(payload);
    setEvalStatus("");
  } catch (error) {
    if (requestId !== boardRequestId) return;
    showToast(error.message);
    setEvalStatus("error");
  }
}

async function submitMove(source, target) {
  if (state.busy) return;
  const candidates = state.position
    ? state.position.legal_moves.filter((move) =>
        move.startsWith(source + target),
      )
    : [];
  if (!candidates.length) return;
  const promotion = elements.promotion.value;
  const move = candidates.find((candidate) => candidate.length === 5)
    ? source + target + promotion
    : source + target;
  const requestId = ++boardRequestId;
  state.busy = true;
  setEvalStatus("evaluating…");
  try {
    const payload = await api("/api/move", {
      start_fen: state.startFen,
      moves: state.moves,
      source: move.slice(0, 2),
      target: move.slice(2, 4),
      promotion: move.length === 5 ? move.slice(4) : null,
      model_ids: [...state.selected],
    });
    if (requestId !== boardRequestId) return;
    applyServerState(payload);
    clearSampleSelection();
    setEvalStatus("");
  } catch (error) {
    if (requestId !== boardRequestId) return;
    showToast(error.message);
    setEvalStatus("error");
  } finally {
    state.busy = false;
  }
}

function onSquareClick(square) {
  const board = parseFen(state.position ? state.position.fen : state.startFen);
  const turn = state.position ? state.position.turn : fenTurn(state.startFen);
  const piece = board[square];
  const own = isOwnPiece(piece, turn);
  if (state.selectedSquare) {
    if (square === state.selectedSquare) {
      state.selectedSquare = null;
      renderBoard();
      return;
    }
    if (legalTargetsFrom(state.selectedSquare).has(square)) {
      submitMove(state.selectedSquare, square);
      return;
    }
    if (own) {
      state.selectedSquare = square;
      renderBoard();
      return;
    }
    state.selectedSquare = null;
    renderBoard();
    return;
  }
  if (own) {
    state.selectedSquare = square;
    renderBoard();
  }
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 5200);
}

function bestForTask(task) {
  const candidates = state.models.filter((model) => model.task === task);
  return (
    candidates.find(
      (model) =>
        model.label.includes("20M") &&
        (model.label.includes("full FT mean") ||
          model.label.includes("all parameters")),
    ) ||
    candidates.find((model) => model.label.includes("20M")) ||
    candidates[0] ||
    null
  );
}

function selectBest(task) {
  const best = bestForTask(task);
  if (!best) return;
  for (const modelId of [...state.selected]) {
    const selected = state.models.find((model) => model.model_id === modelId);
    if (selected?.task === task) state.selected.delete(modelId);
  }
  state.selected.add(best.model_id);
  renderModelList();
  refreshPosition();
}

function renderModelList() {
  elements.modelList.innerHTML = "";
  elements.modelCount.textContent = `${state.models.length} available`;
  const groups = [
    ["sacrifice", "Sacrifice classifiers"],
    ["score", "Score regressors"],
  ];
  for (const [task, title] of groups) {
    const models = state.models.filter((model) => model.task === task);
    if (!models.length) continue;
    const group = document.createElement("div");
    group.className = "model-group";
    group.innerHTML = `<h3>${title}</h3>`;
    for (const [index, model] of models.entries()) {
      const row = document.createElement("div");
      row.className = "model-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `model-${task}-${index}`;
      checkbox.checked = state.selected.has(model.model_id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) state.selected.add(model.model_id);
        else state.selected.delete(model.model_id);
        refreshPosition();
      });
      const label = document.createElement("label");
      label.htmlFor = checkbox.id;
      label.textContent = model.label;
      row.appendChild(checkbox);
      row.appendChild(label);
      group.appendChild(row);
    }
    elements.modelList.appendChild(group);
  }
}

async function loadModels() {
  const payload = await fetch("/api/models").then((response) => response.json());
  state.models = payload.models || [];
  elements.deviceLine.textContent = `${state.models.length} checkpoints · device ${payload.device || "?"}`;
  const score = bestForTask("score");
  const sacrifice = bestForTask("sacrifice");
  if (score) state.selected.add(score.model_id);
  if (sacrifice) state.selected.add(sacrifice.model_id);
  renderModelList();
  await refreshPosition();
}

function renderSamples() {
  const select = elements.sampleSelect;
  select.innerHTML = "";
  state.sampleMap.clear();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = state.samples.length
    ? "Choose a puzzle position…"
    : "No puzzle samples found";
  select.appendChild(placeholder);

  const groups = [
    { label: "Sacrifice samples", sacrifice: 1 },
    { label: "Non-sacrifice samples", sacrifice: 0 },
  ];
  for (const group of groups) {
    const samples = state.samples.filter(
      (sample) => sample.sacrifice === group.sacrifice,
    );
    if (!samples.length) continue;
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label;
    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample.sample_id;
      option.textContent = sample.label;
      optgroup.appendChild(option);
      state.sampleMap.set(sample.sample_id, sample);
    }
    select.appendChild(optgroup);
  }
}

async function loadSamples() {
  try {
    const response = await fetch("/api/samples");
    if (!response.ok) throw new Error(`samples failed (${response.status})`);
    const payload = await response.json();
    state.samples = payload.samples || [];
    renderSamples();
  } catch (error) {
    elements.sampleSelect.innerHTML =
      '<option value="">Could not load puzzle samples</option>';
    showToast(error.message);
  }
}

function loadSample(sample) {
  state.startFen = sample.fen;
  state.moves = [];
  state.selectedSquare = null;
  elements.sampleSelect.value = sample.sample_id;
  elements.fenInput.value = sample.fen;
  refreshPosition();
}

function clearSampleSelection() {
  elements.sampleSelect.value = "";
}

function formatCompact(value) {
  if (!value) return "0";
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(Math.round(value));
}

function formatMegabytes(bytes) {
  return (bytes / (1024 * 1024)).toFixed(1);
}

function formatEngineScore(info) {
  if (info.scoreType === "mate") {
    return `#${info.scoreValue}`;
  }
  const pawns = info.scoreValue / 100;
  return `${pawns >= 0 ? "+" : ""}${pawns.toFixed(2)}`;
}

function renderStockfish() {
  const sf = state.stockfish;
  const progress = sf.progress;
  if (sf.error) {
    elements.sfDot.className = "engine-dot error";
    elements.sfEval.textContent = "unavailable";
    elements.sfMeta.textContent = sf.error;
    elements.sfPv.textContent = "";
    return;
  }
  if (!sf.ready) {
    elements.sfDot.className = "engine-dot loading";
    elements.sfEval.textContent = "Loading engine…";
    if (progress && progress.total > 0) {
      const percent = Math.min(100, (progress.received / progress.total) * 100);
      elements.sfMeta.textContent = `Downloading NNUE ${percent.toFixed(0)}% (${formatMegabytes(
        progress.received,
      )}/${formatMegabytes(progress.total)} MB)`;
    } else if (progress) {
      elements.sfMeta.textContent = `Downloading NNUE ${formatMegabytes(
        progress.received,
      )} MB`;
    } else {
      elements.sfMeta.textContent = "Starting Stockfish 19 WebAssembly…";
    }
    elements.sfPv.textContent = "";
    return;
  }
  if (!sf.info) {
    elements.sfDot.className = sf.thinking ? "engine-dot thinking" : "engine-dot ready";
    elements.sfEval.textContent = sf.thinking ? "…" : "—";
    elements.sfMeta.textContent = sf.thinking ? "thinking…" : "ready";
    elements.sfPv.textContent = "";
    return;
  }
  elements.sfDot.className = `engine-dot ${sf.bestmove ? "ready" : "thinking"}`;
  elements.sfEval.textContent = formatEngineScore(sf.info);
  const depth = sf.info.seldepth
    ? `${sf.info.depth}/${sf.info.seldepth}`
    : `${sf.info.depth}`;
  const best = sf.bestmove ? ` · best ${sf.bestmove}` : "";
  elements.sfMeta.textContent = `depth ${depth} · ${formatCompact(
    sf.info.nodes,
  )} nodes · ${formatCompact(sf.info.nps)} nps${best}`;
  elements.sfPv.textContent = sf.info.pv ? `PV (UCI): ${sf.info.pv}` : "";
}

function handleNnueProgress(progress) {
  state.stockfish.progress = progress;
  if (!state.stockfish.ready) renderStockfish();
}

function handleEngineLine(line) {
  const info = parseStockfishInfo(line, state.position?.turn || "white");
  if (info) {
    state.stockfish.info = info;
    state.stockfish.bestmove = null;
    renderStockfish();
  }
  if (line.startsWith("bestmove ")) {
    state.stockfish.bestmove = line.split(" ")[1] || null;
    state.stockfish.thinking = false;
    renderStockfish();
  }
}

function runStockfish(force = false) {
  const sf = state.stockfish;
  if (!sf.controller || !state.position) return;
  if (!force && !elements.sfAuto.checked) return;
  sf.info = null;
  sf.bestmove = null;
  sf.thinking = true;
  renderStockfish();
  sf.controller.evaluate(state.position.fen, Number(elements.sfDepth.value));
}

function initStockfish() {
  if (!window.crossOriginIsolated) {
    state.stockfish.error =
      "Stockfish needs cross-origin isolation. Open http://127.0.0.1:8000 or serve over HTTPS.";
    renderStockfish();
    return;
  }
  try {
    state.stockfish.controller = new StockfishController({
      onReady: () => {
        if (state.stockfish.ready) return;
        state.stockfish.ready = true;
        renderStockfish();
        if (state.position) runStockfish(true);
      },
      onLine: handleEngineLine,
      onError: (message) => {
        state.stockfish.error = message;
        renderStockfish();
      },
      onProgress: handleNnueProgress,
    });
    renderStockfish();
  } catch (error) {
    state.stockfish.error = error?.message || String(error);
    renderStockfish();
  }
}

function wireEvents() {
  document.getElementById("flip-btn").addEventListener("click", () => {
    state.orientation = state.orientation === "white" ? "black" : "white";
    renderBoard();
  });
  document.getElementById("reset-btn").addEventListener("click", () => {
    state.startFen = START_FEN;
    state.moves = [];
    state.selectedSquare = null;
    clearSampleSelection();
    refreshPosition();
  });
  document.getElementById("load-fen-btn").addEventListener("click", async () => {
    const fen = elements.fenInput.value.trim();
    if (!fen) return;
    state.startFen = fen;
    state.moves = [];
    state.selectedSquare = null;
    clearSampleSelection();
    await refreshPosition();
  });
  elements.sampleSelect.addEventListener("change", () => {
    const sample = state.sampleMap.get(elements.sampleSelect.value);
    if (sample) loadSample(sample);
  });
  elements.fenInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      document.getElementById("load-fen-btn").click();
    }
  });
  document.getElementById("copy-fen-btn").addEventListener("click", async () => {
    const fen = state.position ? state.position.fen : elements.fenInput.value;
    try {
      await navigator.clipboard.writeText(fen);
      showToast("FEN copied");
    } catch (error) {
      showToast("Could not copy FEN");
    }
  });
  document
    .getElementById("select-score-btn")
    .addEventListener("click", () => selectBest("score"));
  document
    .getElementById("select-sacrifice-btn")
    .addEventListener("click", () => selectBest("sacrifice"));
  document.getElementById("clear-models-btn").addEventListener("click", () => {
    state.selected.clear();
    renderModelList();
    refreshPosition();
  });
  elements.sfRun.addEventListener("click", () => runStockfish(true));
  elements.sfDepth.addEventListener("change", () => {
    if (elements.sfAuto.checked) runStockfish(true);
  });
  elements.sfAuto.addEventListener("change", () => {
    if (elements.sfAuto.checked) {
      runStockfish(true);
    } else {
      state.stockfish.controller?.stop();
      state.stockfish.info = null;
      state.stockfish.bestmove = null;
      state.stockfish.thinking = false;
      renderStockfish();
    }
  });
}

wireEvents();
renderBoard();
initStockfish();
loadModels().catch((error) => {
  elements.deviceLine.textContent = `Could not load models: ${error.message}`;
  showToast(error.message);
});
loadSamples().catch((error) => showToast(error.message));
