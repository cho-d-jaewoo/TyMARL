"""
on_policy_ha_runner.py — inherited from HARL upstream.

This is one of the per-agent runners HARL ships with. The TyMARL fork uses
these unchanged for all baseline algorithms (HAPPO, HATRPO, HASAC, MAPPO,
MADDPG, etc.). TyPPO and TySAC use ``on_policy_group_runner.py`` and
``off_policy_group_runner.py`` instead.

When using this fork, run ``bash scripts/install_harl.sh``; the upstream
file at ``third_party/HARL/harl/runners/on_policy_ha_runner.py`` will be on
``sys.path`` and resolves the import below.
"""

try:
    from harl.runners.on_policy_ha_runner import OnPolicyHARunner  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OnPolicyHARunner is provided by the HARL upstream. "
        "Run `bash scripts/install_harl.sh` first."
    ) from e

__all__ = ["OnPolicyHARunner"]
