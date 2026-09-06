"""
Extract per-step metrics from log.jsonl files into wide-format CSVs --
one file per scenario, columns = algorithms.

Reads:
    results/runs/<algo>_<env>_<scenario>_seed<i>/log.jsonl

Writes (one CSV per scenario, with `env_steps` as the index column):
    results/metrics/protoss_5_vs_5.csv          cols: env_steps, happo, mappo, typpo
    results/metrics/terran_5_vs_5.csv           cols: env_steps, happo, mappo, typpo
    results/metrics/zerg_5_vs_5.csv             cols: env_steps, happo, mappo, typpo
    results/metrics/spread.csv                  cols: env_steps, happo, mappo, typpo, hasac, tysac
    results/metrics/speaker_listener.csv        cols: env_steps, happo, mappo, typpo, hasac, tysac

The metric written into each algorithm's column is determined by the env:
  smacv2 -> win_rate
  mpe    -> ep_return_mean

If multiple seeds exist for the same (algo, scenario), they are averaged at
each env_steps (post merge). This keeps the file simple; the seed-level raw
trace is left in log.jsonl for anyone who wants it.

Usage
-----
    python scripts/extract_metrics.py
    python scripts/extract_metrics.py --runs-dir results/runs --out-dir results/metrics
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(r"^(?P<algo>[a-zA-Z0-9]+)_(?P<env>[a-zA-Z0-9]+)_(?P<scenario>.+)_seed(?P<seed>\d+)$")

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
    ap.add_argument("--out-dir", default="results/metrics", type=Path)
    ap.add_argument("--xkey", default="env_steps")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError:
        print("[err] pandas required: pip install pandas")
        return 1

    # First pass: read every run into a list of (scenario, algo, env, seed, df)
    # where df has columns [env_steps, value].
    runs_by_scenario: dict[str, dict[str, list[pd.DataFrame]]] = defaultdict(
        lambda: defaultdict(list))
    scenario_env: dict[str, str] = {}  # scenario -> env (to know which metric)

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

        df = pd.DataFrame(rows, columns=["env_steps", "value"])
        # If duplicates at the same env_steps (rare), average.
        df = df.groupby("env_steps", as_index=False).mean()
        runs_by_scenario[scenario][algo].append(df)
        scenario_env[scenario] = env
        print(f"[ok]   {run.name}: {len(df)} unique steps")

    if not runs_by_scenario:
        print("\nNo data extracted.")
        return 1

    # Second pass: per scenario, outer-merge all algorithms on env_steps.
    print()
    for scenario, algo_dfs in runs_by_scenario.items():
        env = scenario_env[scenario]
        ykey = ENV_METRIC[env]

        # For each algo, average across seeds first.
        per_algo: dict[str, pd.DataFrame] = {}
        for algo, dfs in algo_dfs.items():
            if len(dfs) == 1:
                merged = dfs[0]
            else:
                # outer-merge across seeds, then average across seed columns
                m = dfs[0].rename(columns={"value": "v0"})
                for i, d in enumerate(dfs[1:], 1):
                    m = m.merge(d.rename(columns={"value": f"v{i}"}),
                                on="env_steps", how="outer")
                m = m.sort_values("env_steps").reset_index(drop=True)
                value_cols = [c for c in m.columns if c.startswith("v")]
                m["value"] = m[value_cols].mean(axis=1, skipna=True)
                merged = m[["env_steps", "value"]]
            per_algo[algo] = merged.rename(columns={"value": algo})

        # Outer-merge all algos on env_steps.
        algos = list(per_algo.keys())
        out = per_algo[algos[0]]
        for algo in algos[1:]:
            out = out.merge(per_algo[algo], on="env_steps", how="outer")
        out = out.sort_values("env_steps").reset_index(drop=True)

        # Stable column order: env_steps + algos in a sensible order
        # (PPO family first, then SAC family; alphabetical inside each)
        ppo_family = sorted(a for a in algos if a in ("happo", "mappo", "typpo"))
        sac_family = sorted(a for a in algos if a in ("hasac", "tysac"))
        other      = sorted(a for a in algos if a not in ppo_family + sac_family)
        ordered = ppo_family + sac_family + other
        out = out[["env_steps"] + ordered]

        out_path = args.out_dir / f"{scenario}.csv"
        out.to_csv(out_path, index=False)
        print(f"[csv] {out_path}  rows={len(out)}, cols=env_steps + {ordered}  "
              f"(env={env}, metric={ykey})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
