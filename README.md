# TyMARL: Type-routed Multi-Agent Reinforcement Learning

Cooperative MARL that maps policy networks to **unit types** rather than agent
slots. TyPPO and TySAC use MAPPO-style parameter sharing within a type and
HAPPO-style sequential trust-region updates across types, tied together by a
per-(time, batch) cumulative importance ratio `M`.

A SMACv2 team of 3 Stalkers + 2 Zealots is handled by exactly two networks.
SMACv2 resamples the team every episode, so the wrapper reads the spawned unit
types after each reset, forwards a fresh `GroupAssignment` to the algorithm,
and types that did not spawn skip their update that iteration.

## What's here

- **TyPPO** (on-policy), **TySAC** (off-policy), and `GroupQCritic` — a twin-Q
  critic with one head per group.
- A **dynamic-group SMACv2 wrapper**.
- Baselines (MAPPO, HAPPO, HASAC, QMIX). Independent implementations with their
  own training loops, not TyPPO/TySAC reduced to boundary partitions — they
  share only the per-batch infrastructure in `_ppo_common.py` so the comparison
  is apples to apples.
- Configs, training scripts, reproduction recipes.

Built on [HARL](https://github.com/PKU-MARL/HARL) (Zhong et al., JMLR 2024).

## Install

```bash
git clone https://github.com/cho-d-jaewoo/TyMARL.git
cd TyMARL
conda create -n env_hetgrl python=3.10 -y
conda activate env_hetgrl
pip install -e ".[dev,viz]"

bash scripts/install_harl.sh   # HARL upstream -> third_party/HARL/
```

Per-environment setup (StarCraft II, SMACv2, MAMuJoCo, Bi-DexHands) is in
`docs/`. Exact versions and timings for the reported runs:
[`docs/compute_resources.md`](docs/compute_resources.md).

## Run

```bash
bash scripts/train_typpo.sh protoss_5_vs_5
bash scripts/train_tysac.sh mamujoco Walker2d-v2
bash scripts/reproduce_main_results.sh
```

```cmd
REM Windows
python -m harl.train --algo typpo --env smacv2 --scenario protoss_5_vs_5 ^
    --algo-config harl\configs\algos_cfgs\typpo.yaml ^
    --env-config  harl\configs\envs_cfgs\smacv2.yaml ^
    --seed 1 --exp-name typpo_protoss_seed1 --device cuda --save-replay

REM Overnight grids (edit `set SEEDS=` at the top)
scripts\windows\run_typpo.bat
```

Runs land in `results/runs/<exp_name>/` as `log.jsonl` plus checkpoints.

```bash
python scripts/extract_metrics.py                    # log.jsonl -> CSV
python scripts/plot_from_csv.py --all --env smacv2   # CSV -> figures
```

## Algorithms and benchmarks

| | |
|---|---|
| Ours | TyPPO, TySAC |
| Baselines | MAPPO, HAPPO, HASAC, QMIX |

| Benchmark | Composition | Role |
|---|---|---|
| **SMACv2** (`protoss/terran/zerg_5_vs_5`) | stochastic per episode | **primary** — the setting the group update targets |
| SMAC (`MMM2`) | fixed per map | established mixed-unit benchmark |
| MAMuJoCo (`Walker2d-v2`, `HalfCheetah-2x3`) | body-part groups | continuous action, TySAC vs HASAC |
| MPE (`simple_spread`) | mostly homogeneous | where within-type sharing should win |

Full matrix in [`docs/algorithm_compatibility.md`](docs/algorithm_compatibility.md),
hyperparameters in [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Layout

```
harl/
  algorithms/actors/    typpo.py, tysac.py ⭐ + baselines
  algorithms/critics/   group_critic.py ⭐
  envs/smacv2/          ⭐ dynamic-group wrapper
  runners/              ⭐ {on,off}_policy_group_runner.py
  utils/group.py        ⭐ GroupAssignment
  configs/              one YAML per algorithm and benchmark
  train.py              entry point
scripts/                install, train, extract, plot (+ windows/)
tests/                  smoke tests — no SC2 or MuJoCo needed
docs/
```

⭐ = added on top of HARL.

`third_party/HARL/` (cloned by `install_harl.sh`) and `results/` (run output)
are not tracked.

## Test

```bash
pip install -e ".[dev]"
pytest tests/
```

`tests/conftest.py` sets `TYMARL_ALLOW_DUMMY_SMACV2=1` so the suite runs against
a synthetic SMACv2 backend. Real training refuses that fallback and raises
`RuntimeError`, so training on synthetic rewards cannot happen by accident.

## License

MIT — [`LICENSE`](LICENSE), [`NOTICE`](NOTICE). Inherited from HARL.
