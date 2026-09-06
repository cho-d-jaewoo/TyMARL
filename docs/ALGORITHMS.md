# Algorithm reference

Hyperparameter notes for our two algorithms. Defaults live in
``harl/configs/algos_cfgs/{typpo,tysac}.yaml``.

## TyPPO (on-policy)

| Block | Field | Default | Notes |
|-------|-------|---------|-------|
| ppo | clip_param        | 0.2   | PPO clip ε. We do not anneal it; HARL also doesn't. |
| ppo | ppo_epoch         | 5     | Per-group epoch count. Per-group early stop on KL kicks in earlier when needed. |
| ppo | mini_batch        | 256   | Mini-batch *across all agents in the group*; the buffer flattens (T, B, group-size). |
| ppo | entropy_coef      | 0.01  | Conservative; SMAC family already explores via available_actions. |
| ppo | value_loss_coef   | 1.0   | V critic gets equal weight. |
| ppo | gamma             | 0.99  | |
| ppo | gae_lambda        | 0.95  | |
| ppo | target_kl         | 0.02  | Per-group early-stop threshold (1.5 × target_kl). Comes from p1; absent in the original p2 scaffold. |
| sequential_update | random_group_permutation | true | Shuffle the group update order each iteration. Section 7.5 ablates this. |

### Partition-shape extremes

The TyPPO partition is a free choice. Two extreme shapes are useful as
references:

- ``num_groups == 1`` -- all agents share one actor. Topologically
  identical to MAPPO's single shared network, but TyPPO's training loop
  still runs through its sequential-update machinery (with M held at 1
  trivially since there are no prior groups). This is *not* MAPPO --
  the standalone MAPPO implementation in
  ``harl.algorithms.actors.mappo`` is the reference baseline.
- ``num_groups == n_agents`` -- each agent has its own actor.
  Topologically identical to HAPPO. Again, this is *not* HAPPO -- the
  standalone HAPPO implementation in
  ``harl.algorithms.actors.happo`` is the reference baseline.

The intended TyPPO use case sits strictly between these extremes:
``num_groups`` equals the number of distinct unit types in the team,
which under SMACv2 is typically 2 or 3.

## TySAC (off-policy)

| Block | Field | Default | Notes |
|-------|-------|---------|-------|
| sac | gamma             | 0.99   | |
| sac | tau               | 0.005  | Polyak. |
| sac | target_entropy    | auto   | ``-action_dim`` per group, summed over the group's agents (matches HASAC's per-agent recipe). |
| sac | alpha             | 0.2    | Initial temperature; per-group log alpha is auto-tuned. |
| sac | auto_alpha        | true   | |
| sac | buffer_size       | 1e6    | |
| sac | batch_size        | 1000   | |
| sac | warmup_steps      | 1e4    | Pure exploration before any update. |

### Architectural choices specific to TySAC

- **Per-group Q heads** (``GroupQCritic``). Each group's actor sees gradient
  through its own head only; the trunk is shared. Section 7.4 ablates this.
- **Target actors**. A Polyak-averaged copy of every actor stabilises the
  bootstrap; HASAC uses the same recipe per agent.
- **Sequential log-ratio multiplier**. After group k moves, ``log_M[:, k]``
  accumulates the log-ratio between new-actor and target-actor samples; the
  next group's actor loss is weighted by ``exp(log_M[:, group_id])``. This is
  the off-policy analogue of TyPPO's cumulative ratio.

## Construction tests

``tests/test_algorithms.py::test_typpo_boundary_cases_construct`` (the
historical name is preserved for git-blame stability) verifies that
TyPPO can be *constructed* under singleton-per-agent and all-in-one
partitions without errors. It does NOT verify any behavioural identity
between TyPPO at those partitions and the standalone MAPPO/HAPPO
implementations -- those algorithms differ in update order and
methodology, so behavioural identity does not hold.
