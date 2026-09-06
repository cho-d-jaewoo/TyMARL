"""Tests for ``harl.utils.group``.

These tests are pure NumPy — they don't require torch, StarCraft, or MuJoCo,
which is by design: the group abstraction is so central that a regression
here breaks every algorithm in the repo, so it gets the lightest test
dependency footprint.
"""

from __future__ import annotations

import numpy as np
import pytest

from harl.utils.group import (
    GroupAssignment,
    assign_groups_by_type,
    assign_groups_from_list,
    assign_groups_uniform,
    boundary_case_happo,
    boundary_case_mappo,
    coerce_group_assignment,
    random_group_permutation,
)


def test_assign_groups_by_type_simple():
    ga = assign_groups_by_type(["marine", "marine", "marine", "marauder", "marauder", "medivac"])
    assert ga.n_agents == 6
    assert ga.n_groups == 3
    assert ga.group_labels == ["marine", "marauder", "medivac"]
    assert list(ga.agents_of_group[0]) == [0, 1, 2]
    assert list(ga.agents_of_group[1]) == [3, 4]
    assert list(ga.agents_of_group[2]) == [5]
    assert ga.label_of(4) == "marauder"


def test_assign_groups_by_type_preserves_order():
    """The first appearance of each type fixes its group id."""
    ga = assign_groups_by_type(["medivac", "marine", "marine", "medivac"])
    assert ga.group_labels == ["medivac", "marine"]
    assert list(ga.group_of_agent) == [0, 1, 1, 0]


def test_boundary_case_happo():
    ga = boundary_case_happo(5)
    assert ga.is_one_per_agent()
    assert ga.n_groups == 5
    for k, agents in enumerate(ga.agents_of_group):
        assert list(agents) == [k]


def test_boundary_case_mappo():
    ga = boundary_case_mappo(5)
    assert ga.is_single_group()
    assert ga.n_groups == 1
    assert list(ga.agents_of_group[0]) == [0, 1, 2, 3, 4]


def test_assign_groups_uniform_valid():
    ga = assign_groups_uniform(7, 3)
    assert ga.n_agents == 7
    assert ga.n_groups == 3
    # 7 / 3 → sizes [3, 2, 2]
    assert ga.group_sizes == [3, 2, 2]


def test_assign_groups_uniform_rejects_bad_k():
    with pytest.raises(ValueError):
        assign_groups_uniform(5, 0)
    with pytest.raises(ValueError):
        assign_groups_uniform(5, 6)


def test_assign_groups_from_list_round_trip():
    raw = [[0, 1, 2], [3, 4], [5]]
    ga = assign_groups_from_list(raw)
    assert ga.as_list_of_lists() == raw


def test_assign_groups_from_list_rejects_non_partition():
    with pytest.raises(ValueError):
        assign_groups_from_list([[0, 1, 1], [2]])  # 1 appears twice


def test_random_group_permutation_is_permutation():
    ga = assign_groups_uniform(10, 4)
    rng = np.random.default_rng(0)
    perm = random_group_permutation(rng, ga)
    assert sorted(perm.tolist()) == list(range(4))


def test_coerce_accepts_all_inputs():
    ga = boundary_case_happo(3)

    # Already a GroupAssignment
    assert coerce_group_assignment(ga) is ga

    # List of lists
    out = coerce_group_assignment([[0, 1], [2]])
    assert out.n_agents == 3 and out.n_groups == 2

    # List of type strings
    out = coerce_group_assignment(["a", "a", "b"])
    assert out.group_labels == ["a", "b"]

    # None + n_agents → HAPPO topology
    out = coerce_group_assignment(None, n_agents=4)
    assert out.is_one_per_agent()


def test_coerce_rejects_garbage():
    with pytest.raises(TypeError):
        coerce_group_assignment(42)
    with pytest.raises(ValueError):
        coerce_group_assignment([])
    with pytest.raises(ValueError):
        coerce_group_assignment(None)        # n_agents missing
