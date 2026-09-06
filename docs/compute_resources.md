# Compute resources

The environment behind the numbers reported in the paper. Reproduction steps:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Hardware

GPU 1 × NVIDIA RTX-class (24 GB) · CPU x86-64 ≥ 8 cores · RAM 32 GB ·
Windows 11 · CUDA 12.8

## Software

Python 3.11.15 · PyTorch 2.11.0+cu128 · NumPy 1.26.4 · PettingZoo 1.23.1 ·
SuperSuit 3.10.0 · SMACv2 1.0.0 · PySC2 4.0.0 · Gymnasium 1.3.0 · PyYAML 6.0.3 ·
SciPy 1.17.1 · HARL 1.0.0

Fully pinned set: [`requirements-paper.txt`](../requirements-paper.txt).

## Memory

Peak 6–8 GB per `(algorithm, scenario, seed)` process.

| | |
|---|---|
| Model parameters (2-layer MLP, hidden 64) | < 100 MB |
| On-policy rollout buffer (length 200, 4 threads, 5 agents) | < 1 GB |
| Off-policy replay buffer (1e6 transitions) | 2–4 GB |
| StarCraft II client (SMACv2 only) | 1.5–2 GB |

## Wall-clock

| | |
|---|---|
| SMACv2 5_vs_5, 10M steps, 4 threads — TyPPO / MAPPO / HAPPO | 14–20 h |
| MPE 2M steps, 4 threads — PPO family | 3–4 h |
| MPE 2M steps, 4 threads — SAC family | 5–6 h |

On SMACv2 the bottleneck is the StarCraft II simulator, not the learner.

## Reported runs

SMACv2 (Figure 2): 3 races × 3 algorithms × 5 seeds = 45.
MPE (Figure 3): 2 scenarios × 5 algorithms × 5 seeds = 50 — PPO family on the
discrete variant, SAC family on the continuous one. 95 runs total.
