import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"COMPLETED epoch\s+(\d+)\s+of\s+\d+")
METRIC_LINE_RE = re.compile(r"\[main\]\s+(TRAIN|VAL)\s+metrics:\s*(.+)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)")


def parse_log(text):
    epochs = {}
    current_epoch = None

    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            current_epoch = int(m.group(1))
            epochs.setdefault(current_epoch, {})
            continue

        m = METRIC_LINE_RE.search(line)
        if m and current_epoch is not None:
            split = m.group(1)  # TRAIN or VAL
            kvs = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
            epochs[current_epoch][split] = kvs

    return epochs


def to_series(epochs, split, metric):
    xs, ys = [], []
    for ep in sorted(epochs):
        vals = epochs[ep].get(split, {})
        if metric in vals:
            xs.append(ep)
            ys.append(vals[metric])
    return xs, ys


def main():
    ap = argparse.ArgumentParser(description="Plot a learning curve from a training log.")
    ap.add_argument("logfile", nargs="?", help="Path to log file (reads stdin if omitted).")
    ap.add_argument("-o", "--output", default="learning_curve.png", help="Output PNG path.")
    ap.add_argument("-m", "--metrics", nargs="+", default=["loss"],
                    help="Metrics to plot, e.g. loss mse r2 acc f1 roc_auc.")
    ap.add_argument("--title", default="Learning Curve", help="Figure title.")
    ap.add_argument("--dpi", type=int, default=150, help="Output resolution.")
    args = ap.parse_args()

    text = open(args.logfile).read() if args.logfile else sys.stdin.read()

    epochs = parse_log(text)
    if not epochs:
        sys.exit("No epoch metrics found. Expected '[main] TRAIN/VAL metrics:' lines.")

    metrics = args.metrics
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        tx, ty = to_series(epochs, "TRAIN", metric)
        vx, vy = to_series(epochs, "VAL", metric)

        if not tx and not vx:
            ax.set_title(f"{metric} (not found)")
            ax.axis("off")
            continue

        if tx:
            ax.plot(tx, ty, marker="o", markersize=3, label="train", color="#1f77b4")
        if vx:
            ax.plot(vx, vy, marker="s", markersize=3, label="val", color="#d62728")

        # Log scale reads better for loss/mse which span orders of magnitude.
        if metric in ("loss", "mse"):
            ax.set_yscale("log")

        # Mark best val epoch (min for loss/mse, max otherwise).
        if vx:
            if metric in ("loss", "mse"):
                bi = min(range(len(vy)), key=lambda i: vy[i])
            else:
                bi = max(range(len(vy)), key=lambda i: vy[i])
            ax.axvline(vx[bi], color="gray", ls="--", lw=0.8, alpha=0.7)
            ax.annotate(f"best val\nep {vx[bi]}: {vy[bi]:.4g}",
                        xy=(vx[bi], vy[bi]),
                        xytext=(0.98, 0.95), textcoords="axes fraction",
                        ha="right", va="top", fontsize=8,
                        color="gray")

        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.output} ({len(epochs)} epochs, metrics: {', '.join(metrics)})")


if __name__ == "__main__":
    main()
