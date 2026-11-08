from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       background: #14161a; color: #e6e6e6; }
header { padding: 22px 24px 16px; border-bottom: 1px solid #2a2e35; }
h1 { margin: 0 0 6px; font-size: 21px; }
h2 { margin: 20px 0 12px; font-size: 18px; }
h3 { margin: 16px 0 8px; font-size: 15px; color: #c9d1d9; }
main { padding: 0 24px 40px; }
.sub { color: #9aa4b2; font-size: 13px; }
.tabs { display: flex; gap: 8px; padding: 14px 24px 0; }
.tab-button { background: #1c1f25; color: #c9d1d9; border: 1px solid #2a2e35;
              border-radius: 8px; padding: 8px 14px; cursor: pointer; font-size: 14px; }
.tab-button.active { background: #59a14f; color: #fff; border-color: #59a14f; }
.tab { display: none; }
.tab.active { display: block; }
table { border-collapse: collapse; margin: 8px 0 18px; font-size: 13px; }
th, td { border: 1px solid #2a2e35; padding: 5px 9px; text-align: left;
         vertical-align: top; }
th { background: #1c1f25; color: #c9d1d9; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 12px; }
.card { background: #1c1f25; border: 1px solid #2a2e35; border-radius: 12px;
        padding: 12px; margin: 10px 0; }
.board { display: grid; grid-template-columns: repeat(8, 1fr); width: 100%;
         max-width: 240px; margin: 0 auto; aspect-ratio: 1; overflow: hidden;
         border-radius: 8px; box-shadow: 0 3px 10px #0006; }
.sq { display: flex; align-items: center; justify-content: center; }
.light { background: #f0d9b5; }
.dark { background: #b58863; }
.piece { font-size: clamp(13px, 2.4vw, 24px); line-height: 1; user-select: none; }
.piece.white { color: #fff; text-shadow: 0 1px 2px #000, 0 0 1px #000; }
.piece.black { color: #111; text-shadow: 0 1px 1px #fff4; }
.candidate { border: 2px solid #2a2e35; border-radius: 10px; padding: 6px; }
.candidate.positive { border-color: #59a14f; }
.meta { color: #9aa4b2; font-size: 12px; margin-top: 4px; }
.rank { font-weight: 700; color: #f0c674; }
.chip { display: inline-block; background: #2a2f38; border-radius: 999px;
        padding: 1px 7px; margin: 1px; font-size: 11px; }
.note { color: #f0c674; font-size: 13px; }
details { margin: 8px 0; }
summary { cursor: pointer; color: #c9d1d9; font-weight: 600; }
details details { margin-left: 14px; }
"""

_JS = r"""
const PIECES = {p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚',
                P:'♙',R:'♖',N:'♘',B:'♗',Q:'♕',K:'♔'};
function renderBoard(element) {
  const placement = (element.dataset.fen || '').split(' ')[0];
  const ranks = placement.split('/');
  let html = '';
  for (let rank = 0; rank < 8; rank += 1) {
    let file = 0;
    for (const symbol of (ranks[rank] || '')) {
      if (/\d/.test(symbol)) {
        for (let i = 0; i < Number(symbol); i += 1) {
          html += square(file, rank, '');
          file += 1;
        }
      } else {
        html += square(file, rank, symbol);
        file += 1;
      }
    }
  }
  element.innerHTML = html;
}
function square(file, rank, piece) {
  const color = (file + rank) % 2 === 1 ? 'dark' : 'light';
  const glyph = piece ? PIECES[piece] : '';
  const shade = piece && piece === piece.toUpperCase() ? 'white' : 'black';
  const inner = glyph ? `<span class="piece ${shade}">${glyph}</span>` : '';
  return `<div class="sq ${color}">${inner}</div>`;
}
document.querySelectorAll('.board[data-fen]').forEach(renderBoard);
document.querySelectorAll('.tab-button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tab-button').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach((s) => s.classList.remove('active'));
    button.classList.add('active');
    document.getElementById('tab-' + button.dataset.tab).classList.add('active');
  });
});
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return _escape(value)


def board_html(fen: str) -> str:
    return f"<div class='board' data-fen='{_escape(fen)}'></div>"


def _chips(themes: Sequence[str]) -> str:
    return "".join(f"<span class='chip'>{_escape(theme)}</span>" for theme in themes)


def _ply_text(ply: object) -> str:
    return f" &middot; ply {_escape(ply)}" if ply is not None else ""


def _example_card(example: dict) -> str:
    candidates = []
    for candidate in example.get("candidates", []):
        classes = "candidate positive" if candidate.get("is_positive") else "candidate"
        label = "positive" if candidate.get("is_positive") else "negative"
        candidates.append(
            f"<div class='{classes}'>"
            f"{board_html(candidate['fen'])}"
            f"<div class='meta'>{label} #{candidate['sample_id']}"
            f"{_ply_text(candidate.get('ply'))} "
            f"&middot; score {float(candidate['score']):.4f} "
            f"&middot; rank <span class='rank'>{float(candidate['rank']):.1f}</span>"
            f"</div><div>{_chips(candidate.get('themes', []))}</div></div>"
        )
    group = example.get("group")
    group_text = f" &middot; {_escape(group)}" if group else ""
    return (
        "<div class='card'>"
        f"<div class='meta'>query #{example['query_sample_id']}"
        f"{_ply_text(example.get('query_ply'))}{group_text} "
        f"&middot; positive rank <span class='rank'>"
        f"{float(example['positive_rank']):.1f}</span>"
        f" &middot; top1={bool(example.get('top1'))}</div>"
        "<div class='grid'>"
        f"<div class='candidate'>{board_html(example['query_fen'])}"
        "<div class='meta'>query</div></div>" + "".join(candidates) + "</div></div>"
    )


def _examples_block(examples: Sequence[dict]) -> str:
    if not examples:
        return "<p class='note'>No examples available.</p>"
    return "".join(_example_card(example) for example in examples)


def _ranking_table_rows(metric: dict, label: str) -> str:
    return (
        f"<tr><th>{_escape(label)}</th>"
        f"<td>{_format(metric.get('n_queries'))}</td>"
        f"<td>{_format(metric.get('top1'))}</td>"
        f"<td>{_format(metric.get('mrr'))}</td>"
        f"<td>{_format(metric.get('mean_rank'))}</td>"
        f"<td>{_format(metric.get('chance_top1'))}</td>"
        f"<td>{_format(metric.get('chance_mrr'))}</td></tr>"
    )


def _ranking_table(rows: Sequence[tuple[str, dict]]) -> str:
    header = (
        "<table><thead><tr><th>condition</th><th>n</th><th>top1</th>"
        "<th>MRR</th><th>mean rank</th><th>chance top1</th>"
        "<th>chance MRR</th></tr></thead><tbody>"
    )
    return (
        header
        + "".join(_ranking_table_rows(row, label) for label, row in rows)
        + ("</tbody></table>")
    )


_BUCKET_LABELS = {
    "best": "Best 3",
    "random": "Random 3",
    "worst": "Worst 3",
}


def _bucket_block(buckets: dict | None) -> str:
    buckets = buckets or {}
    parts = []
    for bucket in ("best", "random", "worst"):
        rows = buckets.get(bucket, [])
        parts.append(
            f"<details><summary>{_BUCKET_LABELS[bucket]} "
            f"({len(rows)})</summary>{_examples_block(rows)}</details>"
        )
    return "".join(parts)


def _test_section(test: dict, modes: Sequence[str]) -> str:
    metrics = test.get("metrics")
    if not metrics:
        return "<p class='note'>No test samples were loaded.</p>"
    rows = [
        ("in-game vs out-of-game (random)", metrics["random"]),
        ("in-game vs out-of-game (matched)", metrics["matched"]),
    ]
    html_parts = [
        "<p class='meta'>Positive: a temporal neighbour from the same game. "
        "Negatives: positions from other games. Matched negatives also have "
        "similar material, pawn structure, phase, open files, mobility, and "
        "king safety.</p>",
        "<p class='meta'>Best = positive ranked 1st; worst = positive ranked "
        "last; random = sampled from the middle.</p>",
        _ranking_table(rows),
    ]
    for mode in modes:
        buckets = test.get("examples", {}).get(mode, {})
        html_parts.append(
            f"<details><summary>{_escape(mode)} negatives</summary>"
            f"{_bucket_block(buckets)}</details>"
        )
    return "".join(html_parts)


def _motif_section(motifs: dict, modes: Sequence[str]) -> str:
    metrics = motifs.get("metrics")
    if not metrics:
        return "<p class='note'>No puzzle samples or themes were loaded.</p>"
    rows = []
    for theme in motifs.get("themes", []):
        per_theme = metrics["per_theme"].get(theme)
        if not per_theme:
            continue
        rows.append((f"{theme} (random)", per_theme["random"]))
        rows.append((f"{theme} (matched)", per_theme["matched"]))
    html_parts = [
        "<p class='meta'>Positive: same Lichess theme, different puzzle. "
        "Negatives: different themes. Matched negatives share material, pawn "
        "structure, phase, open files, mobility, and king safety with the "
        "query.</p>",
        "<p class='meta'>Best = positive ranked 1st; worst = positive ranked "
        "last; random = sampled from the middle.</p>",
        _ranking_table(rows),
    ]
    examples = motifs.get("examples", {})
    for theme in motifs.get("themes", []):
        theme_parts = [
            f"<p class='meta'>Condition: <b>{_escape(theme)}</b> positions. "
            "Positive is another puzzle with the same theme; negatives are "
            "puzzles without it.</p>"
        ]
        for mode in modes:
            buckets = examples.get(mode, {}).get(theme, {})
            theme_parts.append(
                f"<details><summary>{_escape(mode)} negatives</summary>"
                f"{_bucket_block(buckets)}</details>"
            )
        html_parts.append(
            f"<details><summary>{_escape(theme)}</summary>"
            + "".join(theme_parts)
            + "</details>"
        )
    return "".join(html_parts)


def _class_definition(task: dict, class_label: str) -> str:
    for name, definition in task.get("class_definitions", []):
        if name == class_label:
            return definition
    return ""


def _surrogate_section(surrogates: dict, modes: Sequence[str]) -> str:
    metrics = surrogates.get("metrics")
    if not metrics:
        return "<p class='note'>No surrogate ranking metrics were computed.</p>"
    rows = []
    for task_name, task in metrics.get("per_task", {}).items():
        for class_label, row in task.get("per_class", {}).items():
            rows.append((f"{task['label']} / {class_label} (random)", row["random"]))
            rows.append((f"{task['label']} / {class_label} (matched)", row["matched"]))
    html_parts = [
        "<p class='meta'>Positive: same surrogate class, different game/puzzle. "
        "Negatives: a different class. Matched negatives control for all other "
        "surrogates but leave the target surrogate free "
        "(leave-one-surrogate-out).</p>",
        "<p class='meta'>Best = positive ranked 1st; worst = positive ranked "
        "last; random = sampled from the middle.</p>",
        _ranking_table(rows),
    ]
    examples = surrogates.get("examples", {})
    tasks = {task["name"]: task for task in surrogates.get("tasks", [])}
    for task_name, classes in examples.items():
        task = tasks.get(task_name, {})
        task_label = task.get("label", task_name)
        class_parts = [
            f"<p class='meta'><b>{_escape(task_label)}</b>: "
            f"{_escape(task.get('description', ''))}</p>",
            "<table><thead><tr><th>class</th><th>definition</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>{_escape(name)}</td><td>{_escape(definition)}</td></tr>"
                for name, definition in task.get("class_definitions", [])
            )
            + "</tbody></table>",
        ]
        for class_label, modes_data in classes.items():
            definition = _class_definition(task, class_label)
            mode_parts = [
                f"<p class='meta'>Condition: <b>{_escape(class_label)}</b> "
                f"({_escape(definition)}).</p>"
            ]
            for mode in modes:
                buckets = modes_data.get(mode, {})
                mode_parts.append(
                    f"<details><summary>{_escape(mode)} negatives</summary>"
                    f"{_bucket_block(buckets)}</details>"
                )
            class_parts.append(
                f"<details><summary>{_escape(class_label)}</summary>"
                + "".join(mode_parts)
                + "</details>"
            )
        html_parts.append(
            f"<details><summary>{_escape(task_label)}</summary>"
            + "".join(class_parts)
            + "</details>"
        )
    return "".join(html_parts)


def render_report(
    path: str | Path,
    *,
    report: dict,
    modes: Sequence[str] = ("matched",),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = report.get("info", {})
    header = (
        "<header><h1>Position similarity ranking</h1>"
        f"<div class='sub'>{_escape(info.get('path', 'checkpoint'))} "
        f"&middot; step {_format(info.get('step'))} "
        f"&middot; val_loss {_format(info.get('val_loss'))} "
        f"&middot; {report.get('n_samples', 0)} positions</div></header>"
    )
    tabs = (
        "<div class='tabs'>"
        "<button class='tab-button active' data-tab='test'>Test</button>"
        "<button class='tab-button' data-tab='motifs'>Motifs</button>"
        "<button class='tab-button' data-tab='surrogates'>Surrogates</button>"
        "</div>"
    )
    sections = (
        "<section class='tab active' id='tab-test'><main>"
        f"<h2>Test: in-game vs out-of-game</h2>"
        f"{_test_section(report.get('test', {}), modes)}</main></section>"
        "<section class='tab' id='tab-motifs'><main>"
        f"<h2>Motif labels</h2>"
        f"{_motif_section(report.get('motifs', {}), modes)}</main></section>"
        "<section class='tab' id='tab-surrogates'><main>"
        f"<h2>Surrogate classes</h2>"
        f"{_surrogate_section(report.get('surrogates', {}), modes)}</main></section>"
    )
    path.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Chesspos position similarity ranking</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{header}{tabs}{sections}<script>{_JS}</script></body></html>"
    )
    return path
