"""Plot training progression curves from a train_masked_transformer log.

Parses `step=... loss=... lr=... tokens/s=...` training lines and
`step=... val_loss=... val_acc=...` eval lines, then writes a 2x2 figure:
train loss (log scale), learning rate, throughput, and validation metrics.

Usage:
    python scripts/analysis/plot_training_curves.py \
        --log /tmp/opencode/logs_d256l3_20261020/train.log \
        --out data/eval/d256l3_20261020/training_curves.png \
        [--restart-step 5000] [--window 20]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRAIN_RE = re.compile(r"^step=(\d+) loss=([\d.]+) lr=([\d.e+-]+) tokens/s=(\d+)")
VAL_RE = re.compile(r"^step=(\d+) val_loss=([\d.]+) val_acc=([\d.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="train.log from a training run")
    parser.add_argument("--out", required=True, help="output png path")
    parser.add_argument(
        "--restart-step",
        type=int,
        default=None,
        help="optional vertical marker at a restart step",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="rolling mean window for the train loss curve",
    )
    parser.add_argument("--title", default="training progression")
    return parser.parse_args()


def parse_log(path: str) -> tuple[np.ndarray, ...]:
    train_step, train_loss, train_lr, toks = [], [], [], []
    val_step, val_loss, val_acc = [], [], []
    for line in open(path):
        if m := TRAIN_RE.match(line):
            step_s, loss_s, lr_s, toks_s = m.groups()
            train_step.append(int(step_s))
            train_loss.append(float(loss_s))
            train_lr.append(float(lr_s))
            toks.append(int(toks_s) / 1e6)
        elif m := VAL_RE.match(line):
            s, vl, va = m.groups()
            val_step.append(int(s))
            val_loss.append(float(vl))
            val_acc.append(float(va))
    return (
        np.array(train_step),
        np.array(train_loss),
        np.array(train_lr),
        np.array(toks),
        np.array(val_step),
        np.array(val_loss),
        np.array(val_acc),
    )


def plot(args: argparse.Namespace, arrays: tuple[np.ndarray, ...]) -> None:
    (train_step, train_loss, train_lr, toks, val_step, val_loss, val_acc) = arrays

    smooth = np.convolve(train_loss, np.ones(args.window) / args.window, mode="valid")
    smooth_x = train_step[args.window - 1 :]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(args.title, fontsize=14)

    ax = axes[0, 0]
    ax.plot(
        train_step,
        train_loss,
        alpha=0.25,
        lw=0.6,
        color="tab:blue",
        label=f"raw (every {int(np.median(np.diff(train_step)))} steps)",
    )
    ax.plot(
        smooth_x,
        smooth,
        lw=1.8,
        color="tab:blue",
        label=f"rolling mean ({args.window})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (weighted)")
    ax.set_title("training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(train_step, train_lr, color="tab:orange")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.set_title("learning rate")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(train_step, toks, color="tab:green", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("M tokens/s")
    ax.set_title(f"throughput (mean {toks.mean():.2f} M tok/s)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(val_step, val_loss, "o-", color="tab:red", label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("val loss", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(val_step, val_acc, "s-", color="tab:purple", label="val acc")
    ax2.set_ylabel("val acc", color="tab:purple")
    if args.restart_step is not None:
        ax.axvline(args.restart_step, color="gray", ls="--", lw=1)
        ax.annotate(
            "restart",
            xy=(args.restart_step, ax.get_ylim()[1]),
            xytext=(args.restart_step + 200, ax.get_ylim()[1] * 0.95),
            fontsize=8,
            color="gray",
        )
    ax.set_title("validation (per eval interval)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print("val table:")
    for s, vl, va in zip(val_step, val_loss, val_acc):
        print(f"  step={s:<7} val_loss={vl:.4f} val_acc={va:.4f}")
    return


def main() -> None:
    args = parse_args()
    plot(args, parse_log(args.log))
    return


if __name__ == "__main__":
    main()
