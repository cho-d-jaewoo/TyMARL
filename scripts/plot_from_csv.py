"""
Plot a learning curve for one (or several) scenario CSVs produced by
extract_metrics.py.

CSV format expected (wide):
    env_steps, <algo1>, <algo2>, ...

Usage
-----
    # one scenario
    python scripts/plot_from_csv.py --scenario protoss_5_vs_5
    python scripts/plot_from_csv.py --scenario spread

    # all scenarios in the metrics dir
    python scripts/plot_from_csv.py --all

    # filter by env (so --all only opens smacv2 panels, etc.)
    python scripts/plot_from_csv.py --all --env smacv2
    python scripts/plot_from_csv.py --all --env mpe

Each scenario gets its own matplotlib window, with:
  - faint raw curve per algorithm (alpha 0.25)
  - bold smoothed curve per algorithm (legend entry)

Stdout summary:
  - SMACv2: env_steps where smoothed curve first crosses 0.5
  - MPE:    final-window mean, max, final smoothed value
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# How to tell smacv2 vs mpe apart from a CSV alone:
#   - smacv2 win_rate values are bounded in [0, 1]
#   - mpe ep_return_mean values are typically negative (or at least < 0 nearby)
# We use a robust heuristic: look at the max of the data values across all
# algorithm columns. > 1 ish or many negatives -> mpe.
# ---------------------------------------------------------------------------
def infer_env(df: pd.DataFrame) -> tuple[str, str]:
    """Return (env_label, metric_label) inferred from the CSV's data range."""
    value_cols = [c for c in df.columns if c != "env_steps"]
    flat = df[value_cols].to_numpy(dtype=float).ravel()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return "unknown", "value"
    if flat.max() <= 1.0 and flat.min() >= 0.0:
        return "smacv2", "win_rate"
    return "mpe", "ep_return_mean"


# ---------------------------------------------------------------------------
# Smoothing: simple moving average. NaN-aware (interpolates over short gaps,
# then smooths). Output preserves the input's leading/trailing NaNs.
# ---------------------------------------------------------------------------
def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 2:
        return y.copy()
    # Mark where the original was NaN; we'll restore those positions after.
    mask_nan = np.isnan(y)
    if mask_nan.all():
        return y.copy()
    # Linearly interpolate internal NaNs so the moving average doesn't get holes.
    idx = np.arange(len(y))
    valid = ~mask_nan
    y_filled = np.interp(idx, idx[valid], y[valid])
    pad = window // 2
    yp = np.pad(y_filled, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(yp, kernel, mode="valid")[: len(y)]
    # Restore NaNs only where there was a *long* leading/trailing NaN block,
    # i.e. NaNs at the very start/end of the series.
    out = smoothed.copy()
    if mask_nan[0]:
        # find first non-NaN
        first_valid = np.argmax(valid)
        out[:first_valid] = np.nan
    if mask_nan[-1]:
        # find last non-NaN (from end)
        last_valid = len(y) - 1 - np.argmax(valid[::-1])
        out[last_valid + 1:] = np.nan
    return out


def step_to_threshold(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    """First x where y crosses thr (linear interp). NaN if never."""
    valid = ~np.isnan(y)
    if not valid.any():
        return float("nan")
    crossed = np.where(valid & (y >= thr))[0]
    if len(crossed) == 0:
        return float("nan")
    i = crossed[0]
    if i == 0:
        return float(x[0])
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (thr - y0) / (y1 - y0) * (x1 - x0))


def plot_one(csv_path: Path, smooth_window: int, threshold: float,
             final_window_frac: float):
    """Open one matplotlib window for one CSV; print summary table."""
    df = pd.read_csv(csv_path)
    if df.empty or "env_steps" not in df.columns:
        print(f"[skip] {csv_path.name}: empty or missing env_steps")
        return None

    env, metric = infer_env(df)
    scenario = csv_path.stem  # filename without .csv
    algos = [c for c in df.columns if c != "env_steps"]
    if not algos:
        print(f"[skip] {csv_path.name}: no algorithm columns")
        return None

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), num=f"{env} / {scenario}")

    cmap = plt.get_cmap("tab10")
    color_for = {a: cmap(i) for i, a in enumerate(algos)}

    x = df["env_steps"].to_numpy(dtype=float)
    summary_rows = []

    for algo in algos:
        y = df[algo].to_numpy(dtype=float)
        if np.isnan(y).all():
            continue
        color = color_for[algo]

        # Faint raw curve
        ax.plot(x, y, color=color, alpha=0.25, linewidth=1.0)

        # Bold smoothed curve
        ys = smooth(y, smooth_window)
        ax.plot(x, ys, color=color, linewidth=2.4, label=algo)

        # Summary metrics
        valid = ~np.isnan(ys)
        x_v, y_v = x[valid], ys[valid]
        if len(x_v) == 0:
            continue
        if env == "smacv2":
            s = step_to_threshold(x_v, y_v, threshold)
            summary_rows.append({
                "algo": algo,
                "final_smoothed": float(y_v[-1]),
                "max_smoothed":   float(y_v.max()),
                f"step_to_{threshold}": s,
            })
        else:
            cut = int((1 - final_window_frac) * len(y_v))
            fw_mean = float(np.mean(y_v[cut:])) if cut < len(y_v) else float(y_v[-1])
            summary_rows.append({
                "algo": algo,
                "final_smoothed": float(y_v[-1]),
                f"mean_last_{int(final_window_frac*100)}pct": fw_mean,
                "max_smoothed":   float(y_v.max()),
            })

    ax.set_xlabel("env_steps")
    ax.set_ylabel(metric)
    ax.set_title(f"{env} / {scenario}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()

    # Print summary
    print(f"\n--- {env} / {scenario} ---")
    if summary_rows:
        tbl = pd.DataFrame(summary_rows).sort_values("algo")
        with pd.option_context(
                "display.float_format",
                lambda v: f"{v:.4f}" if (np.isfinite(v) and abs(v) < 1e3) else f"{v:.3e}",
        ):
            print(tbl.to_string(index=False))

    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", default="results/metrics", type=Path)
    ap.add_argument("--scenario", default=None,
                    help="scenario name (matches CSV filename without extension); "
                         "ignored if --all is set")
    ap.add_argument("--all", action="store_true",
                    help="plot every CSV in the metrics dir")
    ap.add_argument("--env", default=None, choices=["smacv2", "mpe"],
                    help="filter to env (only with --all); inferred from data range")
    ap.add_argument("--smooth", type=int, default=11,
                    help="moving-average window (in raw datapoints)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="win-rate threshold for sample-efficiency print (smacv2)")
    ap.add_argument("--final-window", type=float, default=0.25,
                    help="fraction of training used for 'mean_last_X%%' (mpe)")
    args = ap.parse_args()

    if not args.metrics_dir.exists():
        print(f"[err] metrics dir not found: {args.metrics_dir}")
        print("      run extract_metrics.py first.")
        return 1

    if not args.all and not args.scenario:
        # List what's available and exit
        avail = sorted(p.stem for p in args.metrics_dir.glob("*.csv"))
        print("Available scenarios in", args.metrics_dir, ":")
        for s in avail:
            print(f"  {s}")
        print("\nPick one with --scenario <name> or use --all.")
        return 0

    if args.all:
        csv_paths = sorted(args.metrics_dir.glob("*.csv"))
    else:
        target = args.metrics_dir / f"{args.scenario}.csv"
        if not target.exists():
            avail = sorted(p.stem for p in args.metrics_dir.glob("*.csv"))
            print(f"[err] scenario CSV not found: {target}")
            print("      available:", ", ".join(avail) if avail else "(none)")
            return 1
        csv_paths = [target]

    figures = []
    for p in csv_paths:
        # If the user requested a specific env in --all mode, peek and filter.
        if args.all and args.env is not None:
            df_peek = pd.read_csv(p)
            env, _ = infer_env(df_peek)
            if env != args.env:
                continue
        fig = plot_one(p, args.smooth, args.threshold, args.final_window)
        if fig is not None:
            figures.append(fig)

    if not figures:
        print("\n[plot] nothing to show")
        return 0

    import matplotlib.pyplot as plt
    print(f"\n[plot] showing {len(figures)} window(s) -- close them to exit")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
