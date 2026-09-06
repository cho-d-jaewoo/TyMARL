# Algorithm–Environment Compatibility Matrix

## Legend

- ✅ Supported and tested.
- 🟡 Supported in principle but not in our paper.
- ❌ Not supported (typically because of action-space mismatch).

## Compatibility matrix

| Algorithm           | Type                       | SMAC | **SMACv2 (primary)** | MAMuJoCo | MPE | Bi-DexHands |
|---------------------|----------------------------|:----:|:--------------------:|:--------:|:---:|:-----------:|
| **MAPPO**           | on-policy stochastic       | ✅   | ✅                   | 🟡       | ✅  | 🟡          |
| **MADDPG**          | off-policy deterministic   | ❌   | ❌                   | ✅       | ✅  | 🟡          |
| **MATD3**           | off-policy deterministic   | ❌   | ❌                   | ✅       | ✅  | 🟡          |
| **QMIX**            | value-based                | ✅   | ✅                   | ❌       | 🟡  | ❌          |
| **HAPPO**           | on-policy stochastic       | ✅   | ✅                   | ✅       | ✅  | ✅          |
| **HATRPO**          | on-policy stochastic       | ✅   | ✅                   | ✅       | ✅  | 🟡          |
| **HAA2C**           | on-policy stochastic       | 🟡   | 🟡                   | 🟡       | 🟡  | 🟡          |
| **HASAC**           | off-policy stochastic      | ❌   | ❌                   | ✅       | ✅  | ✅          |
| **HADDPG**          | off-policy deterministic   | ❌   | ❌                   | 🟡       | 🟡  | 🟡          |
| **HATD3**           | off-policy deterministic   | ❌   | ❌                   | 🟡       | 🟡  | 🟡          |
| **TyPPO** (ours)   | on-policy stochastic       | ✅   | ✅                   | ✅       | ✅  | 🟡          |
| **TySAC** (ours)  | off-policy stochastic      | ❌   | ❌                   | ✅       | ✅  | 🟡          |

Notes:
- Discrete-action environments (SMAC family) cannot run deterministic-policy
  algorithms (MADDPG, MATD3, HADDPG, HATD3) without a Gumbel-softmax relaxation.
- HASAC/TySAC require continuous actions; their HARL-family implementations
  on SMAC have not been validated and are marked ❌.
- QMIX is value-based with a centralised mixer; it does not natively handle
  continuous action spaces.
- ``MAPPO`` here is an independent standalone implementation
  (``harl/algorithms/actors/mappo.py``) -- not a reduction of TyPPO at any
  particular partition. The upstream HARL MAPPO is a drop-in replacement
  when fidelity to the 2022 paper matters.

## Group-structure handling per env

| Env | Group structure | How TyPPO consumes it |
|---|---|---|
| **SMACv2** | **Stochastic per episode** (sampled from race pool) | Wrapper rebuilds GroupAssignment after each reset; runner forwards via ``algo.set_groups``. Algorithm pre-allocates one actor per *canonical* unit type and routes agents accordingly; unspawned types skip update. |
| SMAC | Static per map | One GroupAssignment built once at boot from unit types; never changes. |
| MAMuJoCo | Static (body parts as groups) | One GroupAssignment per env config; TySAC's primary use case. |
| MPE | Homogeneous (one group) | Within-type sharing applies trivially; results comparable to standalone MAPPO. |
| Bi-DexHands | Two groups (one per hand) | Static; clean K=2 case. |

## Primary metrics

- **SMAC, SMACv2**: episode win rate (primary), episode return.
- **MAMuJoCo**: episode return.
- **MPE**: episode return.
- **Bi-DexHands**: episode return.

## Recommended scenario coverage (paper)

- **SMACv2** (PRIMARY, Section 7.1): protoss_5_vs_5, terran_5_vs_5, zerg_5_vs_5
  -- three races with stochastic team composition.
- **SMAC** (historic baseline, Section 7.2): MMM2.
- **MAMuJoCo** (continuous-action benchmark, Section 7.3): Walker-2x3, HalfCheetah-2x3.
- **MPE** (homogeneous setting, Section 7.4): simple_spread.
