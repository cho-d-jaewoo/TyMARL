# Reproducibility checklist

## Hardware footprint (paper)

- 1× NVIDIA RTX A6000 (48 GB) per training run.
- 16-core CPU, 64 GB RAM minimum (StarCraft II's memory pressure dominates).

For mixed_smacv2 each (algo, scenario, seed) takes ~6–10 hours wall-clock at
``num_env_steps: 10_000_000``. The full reproduction grid in
``scripts/reproduce_main_results.sh`` is roughly:

- 5 ours-algos × 5 scenarios × 5 seeds  → 125 runs (~10–14 days × 1 GPU)
- 6 baseline algos × 3 main scenarios × 1 seed → 18 runs (~6 days × 1 GPU)

If you have a small GPU budget, prioritise:

1. TyPPO + HAPPO + MAPPO on ``mixed_terran_3m2m1d`` (3 algos × 5 seeds) --
   demonstrates the main numerical claim.
2. TySAC + HASAC on MAMuJoCo Walker-2x3 (2 algos × 5 seeds) -- compares
   the off-policy hybrid against per-agent independent networks on a
   continuous-action, body-part-grouped task.

## Software footprint

- Python 3.10–3.12, PyTorch ≥ 2.0, CUDA 11.8+.
- StarCraft II 4.10+ with the SMAC/SMACv2 maps installed (see
  ``smac.env`` README for paths).
- Optional: MAMuJoCo, MPE, Bi-DexHands (see ``docs/install_bidexhands.md``).

## Random seeds

Each run uses an explicit seed passed via ``--seed``. All sources of
randomness are seeded inside ``harl.train.set_seed``:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

We do **not** set ``torch.backends.cudnn.deterministic = True`` — the
overhead doubles single-GPU wall-clock and the seed range we report (5
seeds) already gives reasonable noise estimates without it.

The paper reports mean ± stderr over seeds 1, 2, 3, 4, 5.

## Logging contract

Every run writes a single JSONL file at:

    results/runs/<algo>__<env>__<scenario>__seed<i>/log.jsonl

with at minimum these fields:

- ``env_steps`` (or ``step`` for off-policy)
- ``ep_return_mean``
- ``win_rate``  (only on SMAC family)
- per-group losses under ``g<id>/{policy_loss,value_loss,entropy,...}``

``scripts/extract_metrics.py`` turns this into one CSV per scenario under
``results/metrics/``; ``scripts/plot_from_csv.py`` plots them.

## Sanity-check pipeline

```bash
# 1) Verify the algorithmic core compiles + construction tests pass.
pip install -e ".[dev]"
pytest tests/                                         # ~30 s, no SC2 needed

# 2) Single short real run to verify SC2 is wired correctly.
SEED=1 bash scripts/train_typpo.sh mixed_terran_3m2m1d
```

If step 1 passes, you are ready to dispatch ``reproduce_main_results.sh``.
