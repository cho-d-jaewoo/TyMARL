"""
off_policy_ha_runner.py — inherited from HARL upstream.

See ``on_policy_ha_runner.py`` for context.
"""

try:
    from harl.runners.off_policy_ha_runner import OffPolicyHARunner  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OffPolicyHARunner is provided by the HARL upstream. "
        "Run `bash scripts/install_harl.sh` first."
    ) from e

__all__ = ["OffPolicyHARunner"]
