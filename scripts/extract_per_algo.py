"""
Split log.jsonl files into one CSV per (env, scenario, algo) combination.

Reads:
    results/runs/<algo>_<env>_<scenario>_seed<i>/log.jsonl

Writes (one file per benchmark/scenario/algorithm triplet):
    results/metrics_per_algo/smacv2_protoss_5_vs_5_typpo.csv
    results/metrics_per_algo/smacv2_protoss_5_vs_5_happo.csv
    results/metrics_per_algo/smacv2_protoss_5_vs_5_mappo.csv
    results/metrics_per_algo/smacv2_terran_5_vs_5_typpo.csv
    ...
    results/metrics_per_algo/mpe_spread_typpo.csv
    results/metrics_per_algo/mpe_speaker_listener_tysac.csv
    ...

Each CSV has two columns: env_steps and the metric value.
The metric is win_rate for smacv2 and ep_return_mean for mpe.
If multiple seeds exist for the same triplet, values are averaged at each
env_steps (with outer-merge for non-aligned step grids).

Usage
-----
    python scripts/extract_per_algo.py
    python scripts/extract_per_algo.py --runs-dir results/runs --out-dir results/metrics_per_algo
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(r"^(?P<algo>[a-zA-Z0-9]+)_(?P<env>[a-zA-Z0-9]+)_(?P<scenario>.+)_seed(?P<seed>\d+)$")

# What metric to extract per benchmark family.
ENV_METRIC = {
    "smacv2": "win_rate",
    "smac":   "win_rate",
    "mpe":    "ep_return_mean",
}


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_run_dir(name: str):
    m = RUN_RE.match(name)
    return m.groupdict() if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="results/runs", type=Path)
    ap.add_argument("--out-dir", default="results/metrics_per_algo", type=Path)
    ap.add_argument("--xkey", default="env_steps")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        print("[err] pandas required: pip install pandas")
        return 1

    # Group runs by (env, scenario, algo); inside each group, list of seed dataframes.
    triplet_seeds: dict[tuple[str, str, str], list[pd.DataFrame]] = defaultdict(list)

    for run in sorted(args.runs_dir.glob("*")):
        if not run.is_dir():
            continue
        meta = parse_run_dir(run.name)
        if not meta:
            print(f"[skip] cannot parse: {run.name}")
            continue
        log = run / "log.jsonl"
        if not log.exists():
            print(f"[skip] no log.jsonl: {run.name}")
            continue
        env, scenario, algo = meta["env"], meta["scenario"], meta["algo"]
        ykey = ENV_METRIC.get(env)
        if ykey is None:
            print(f"[skip] unknown env: {env}")
            continue

        rows = []
        for rec in iter_jsonl(log):
            x = rec.get(args.xkey, rec.get("step"))
            y = rec.get(ykey)
            if x is None or y is None:
                continue
            rows.append((int(x), float(y)))
        if not rows:
            print(f"[skip] empty: {run.name}")
            continue

        df = pd.DataFrame(rows, columns=["env_steps", ykey])
        # Deduplicate any same-step entries (rare).
        df = df.groupby("env_steps", as_index=False).mean()
        triplet_seeds[(env, scenario, algo)].append(df)
        print(f"[ok]   {run.name}: {len(df)} steps")

    if not triplet_seeds:
        print("\nNo data extracted.")
        return 1

    print()
    for (env, scenario, algo), dfs in sorted(triplet_seeds.items()):
        ykey = ENV_METRIC[env]
        if len(dfs) == 1:
            merged = dfs[0]
        else:
            # Outer-merge across seeds, then average across seed columns.
            m = dfs[0].rename(columns={ykey: "v0"})
            for i, d in enumerate(dfs[1:], 1):
                m = m.merge(d.rename(columns={ykey: f"v{i}"}),
                            on="env_steps", how="outer")
            m = m.sort_values("env_steps").reset_index(drop=True)
            value_cols = [c for c in m.columns if c.startswith("v")]
            m[ykey] = m[value_cols].mean(axis=1, skipna=True)
            merged = m[["env_steps", ykey]]

        out_path = args.out_dir / f"{env}_{scenario}_{algo}.csv"
        merged.to_csv(out_path, index=False)
        print(f"[csv] {out_path.name}  rows={len(merged)}, n_seeds={len(dfs)}")

    print(f"\n[done] wrote {len(triplet_seeds)} files to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
