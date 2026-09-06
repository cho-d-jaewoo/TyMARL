"""
Heterogeneous group assignment.

A *group* is a set of agents that (a) share a policy network and
(b) are updated together as one block in our sequential trust-region
schedule. Groups are the central abstraction of TyPPO / TySAC.

This module is responsible for one job: turn a list of N agents
(possibly annotated with type metadata coming from the environment)
into a ``GroupAssignment`` that the runner and the algorithm consume.

Two extreme partitions are useful as test fixtures and as conceptual
references:

- One group per agent  -- the HAPPO/HASAC topology.
- One group total      -- the MAPPO topology.

These are exposed via ``boundary_case_happo`` and ``boundary_case_mappo``
strictly as topology constructors used by ``tests/test_groups.py``.
TyPPO/TySAC are *not* presented as a unifying framework that recovers
MAPPO/HAPPO under specific partitions: those two algorithms differ from
TyPPO in update order and methodology, not just in partition shape, and
their implementations live in standalone classes (``mappo.py``, ``happo.py``).
TyPPO instead combines MAPPO-style parameter sharing within each type
with HAPPO-style sequential update across types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------

@dataclass
class GroupAssignment:
    """Partition of N agents into K disjoint groups.

    Attributes
    ----------
    n_agents : int
        Total number of agents (= sum of group sizes).
    group_of_agent : np.ndarray, shape (N,)
        ``group_of_agent[i]`` is the group index of agent ``i`` in [0, K).
    agents_of_group : list[np.ndarray]
        ``agents_of_group[k]`` is the array of agent indices in group k.
    group_labels : list[str]
        Human-readable label per group (e.g. ``"stalker"``). Length K.
    """

    n_agents: int
    group_of_agent: np.ndarray
    agents_of_group: List[np.ndarray]
    group_labels: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_groups(self) -> int:
        return len(self.agents_of_group)

    @property
    def group_sizes(self) -> List[int]:
        return [len(g) for g in self.agents_of_group]

    def is_one_per_agent(self) -> bool:
        """True iff every agent is its own group (HAPPO / HASAC limit)."""
        return self.n_groups == self.n_agents

    def is_single_group(self) -> bool:
        """True iff all agents share one group (MAPPO limit)."""
        return self.n_groups == 1

    def label_of(self, agent_idx: int) -> str:
        if not self.group_labels:
            return f"group_{self.group_of_agent[agent_idx]}"
        return self.group_labels[self.group_of_agent[agent_idx]]

    def as_list_of_lists(self) -> List[List[int]]:
        """Return the partition in the ``[[0,1,2],[3,4],[5]]`` form.

        Provided for compatibility with code paths that prefer the plain
        list-of-lists representation (notably the original HARL idioms).
        """
        return [list(map(int, g)) for g in self.agents_of_group]

    def __repr__(self) -> str:  # pragma: no cover
        labels = self.group_labels or [f"g{i}" for i in range(self.n_groups)]
        sizes = ", ".join(f"{lbl}:{sz}" for lbl, sz in zip(labels, self.group_sizes))
        return f"GroupAssignment(N={self.n_agents}, K={self.n_groups}, [{sizes}])"


# -----------------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------------

def assign_groups_by_type(
    agent_types: Sequence[str],
    canonical_order: Sequence[str] | None = None,
) -> GroupAssignment:
    """Group agents by exact type-string equality.

    This is the default we use for ``mixed_smacv2``, where the env knows
    each agent is e.g. ``"marine"``, ``"marauder"``, ``"medivac"``.

    Parameters
    ----------
    agent_types
        Per-agent type label, length ``n_agents``.
    canonical_order
        Optional ordered list of *all* possible types in the environment.
        When provided, group IDs follow this order regardless of which
        types actually appear in the current episode. This is essential
        for TyMARL on environments with stochastic team composition: in
        SMACv2 the same actor must always represent the same unit type,
        even across episodes where the type counts vary (or where some
        types do not appear at all).

        When ``None`` (legacy behaviour), group IDs follow the order of
        first appearance in ``agent_types``. This is unsafe for
        stochastic-team training: ``g0`` would mean ``stalker`` in one
        episode and ``colossus`` in the next, so a single actor would
        receive gradients from inconsistent types.
    """
    n = len(agent_types)
    if n == 0:
        raise ValueError("agent_types is empty")

    if canonical_order is not None:
        # Sanity-check: every observed type must be in the canonical list.
        unknown = sorted(set(agent_types) - set(canonical_order))
        if unknown:
            raise ValueError(
                f"agent_types contain types not in canonical_order: "
                f"{unknown}. canonical_order={list(canonical_order)}"
            )
        # Stable group IDs from canonical_order. We instantiate a group
        # for *every* canonical type, even those absent this episode, so
        # `actors[k]` keeps a consistent type meaning across episodes.
        label_to_id: Dict[str, int] = {
            t: i for i, t in enumerate(canonical_order)
        }
        labels = list(canonical_order)
    else:
        # Legacy: order groups by first appearance (UNSAFE for stochastic
        # team composition; preserved only for backward compatibility).
        label_to_id = {}
        for t in agent_types:
            if t not in label_to_id:
                label_to_id[t] = len(label_to_id)
        labels = [
            lbl for lbl, _ in sorted(label_to_id.items(), key=lambda kv: kv[1])
        ]

    group_of_agent = np.empty(n, dtype=np.int64)
    for i, t in enumerate(agent_types):
        group_of_agent[i] = label_to_id[t]

    k = len(labels)
    agents_of_group = [np.where(group_of_agent == g)[0] for g in range(k)]
    return GroupAssignment(
        n_agents=n,
        group_of_agent=group_of_agent,
        agents_of_group=agents_of_group,
        group_labels=labels,
    )


def assign_groups_uniform(n_agents: int, n_groups: int) -> GroupAssignment:
    """Split N agents into K contiguous, near-equal-size groups.

    Used when the environment exposes no type information but the user
    wants to study the behaviour of TyM* with an a-priori K.
    """
    if n_groups <= 0 or n_groups > n_agents:
        raise ValueError(f"n_groups must be in [1, {n_agents}], got {n_groups}")

    splits = np.array_split(np.arange(n_agents), n_groups)
    group_of_agent = np.empty(n_agents, dtype=np.int64)
    for g, idxs in enumerate(splits):
        group_of_agent[idxs] = g
    labels = [f"group_{g}" for g in range(n_groups)]
    return GroupAssignment(
        n_agents=n_agents,
        group_of_agent=group_of_agent,
        agents_of_group=[np.asarray(s) for s in splits],
        group_labels=labels,
    )


def assign_groups_from_list(groups: Sequence[Sequence[int]]) -> GroupAssignment:
    """Build a GroupAssignment from the plain ``[[0,1],[2,3]]`` form.

    Used when an env (or a config) hands the partition in HARL idiom.
    """
    flat = [a for g in groups for a in g]
    n = len(flat)
    if sorted(flat) != list(range(n)):
        raise ValueError(
            f"groups must be a partition of [0, n_agents) — got {groups}"
        )
    group_of_agent = np.empty(n, dtype=np.int64)
    agents_of_group: List[np.ndarray] = []
    for k, g in enumerate(groups):
        idxs = np.asarray(list(g), dtype=np.int64)
        agents_of_group.append(idxs)
        group_of_agent[idxs] = k
    return GroupAssignment(
        n_agents=n,
        group_of_agent=group_of_agent,
        agents_of_group=agents_of_group,
        group_labels=[f"group_{k}" for k in range(len(groups))],
    )


# -----------------------------------------------------------------------------
# Topology-extreme shortcuts (used by tests and by the experimental grid)
# -----------------------------------------------------------------------------

def boundary_case_happo(n_agents: int) -> GroupAssignment:
    """Each agent in its own group -- the HAPPO/HASAC partition topology.

    Note: this only constructs the singleton-group partition. It does NOT
    turn TyPPO into HAPPO -- the two algorithms differ in update order
    and other internals, and the standalone HAPPO implementation lives in
    ``harl.algorithms.actors.happo``.
    """
    return assign_groups_uniform(n_agents, n_agents)


def boundary_case_mappo(n_agents: int) -> GroupAssignment:
    """All agents in a single group -- the MAPPO partition topology.

    Note: this only constructs the all-in-one partition. It does NOT turn
    TyPPO into MAPPO -- the two algorithms differ in methodology, and the
    standalone MAPPO implementation lives in ``harl.algorithms.actors.mappo``.
    """
    return assign_groups_uniform(n_agents, 1)


# -----------------------------------------------------------------------------
# Helpers used by the algorithm core
# -----------------------------------------------------------------------------

def random_group_permutation(
    rng: np.random.Generator,
    assignment: GroupAssignment,
) -> np.ndarray:
    """Random order of group indices for the sequential update.

    HAML-style algorithms shuffle the agent update order each iteration.
    For TyM* we shuffle the *group* update order — within a group, the
    update is a single shared-parameter SGD step, so per-agent order is
    irrelevant.
    """
    return rng.permutation(assignment.n_groups)


def coerce_group_assignment(
    spec, n_agents: int | None = None,
    canonical_order: Sequence[str] | None = None,
) -> GroupAssignment:
    """Best-effort coercion of a user-supplied group spec into a GroupAssignment.

    Accepts:

    - a ``GroupAssignment`` (returned unchanged),
    - a list of lists like ``[[0,1,2],[3,4],[5]]``,
    - a list of type strings like ``["marine","marine","marauder"]``,
    - ``None`` together with ``n_agents``: defaults to one group per agent
      (HAPPO topology).

    ``canonical_order`` is forwarded to ``assign_groups_by_type`` when the
    spec is a list of type strings; ignored otherwise.
    """
    if isinstance(spec, GroupAssignment):
        return spec
    if spec is None:
        if n_agents is None:
            raise ValueError("n_agents must be provided when spec is None")
        return boundary_case_happo(n_agents)
    if not spec:
        raise ValueError("group spec is empty")

    first = spec[0]
    if isinstance(first, str):
        return assign_groups_by_type(spec, canonical_order=canonical_order)
    if isinstance(first, (list, tuple, np.ndarray)):
        return assign_groups_from_list(spec)
    raise TypeError(
        f"Cannot interpret group spec of type {type(spec).__name__}: {spec!r}"
    )
