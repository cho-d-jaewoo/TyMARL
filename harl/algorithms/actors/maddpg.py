"""
MADDPG -- Multi-Agent DDPG (Lowe et al., NeurIPS 2017).

This repository does not ship a standalone MADDPG implementation. The
algorithm has a well-known reference implementation in
`HARL <https://github.com/PKU-MARL/HARL>`_; if you need MADDPG numbers for
a comparison, use that codebase directly rather than going through the
TyMARL entry point.

The MADDPG class is intentionally not a reduction of TySAC at any
particular partition / temperature setting: TySAC's per-group log-ratio
multiplier and per-group alpha tuning differ from MADDPG's deterministic
target-network updates in ways that affect both gradient estimation and
exploration, so calling one a "boundary case" of the other would be
misleading. We therefore raise a clear error instead of silently
substituting a TySAC look-alike that would inflate paper baselines.

Compatible benchmarks (when running upstream HARL)
--------------------------------------------------
MAMuJoCo, MPE, Bi-DexHands -- continuous-action only.
"""

from __future__ import annotations


class MADDPG:                                          # noqa: N801
    """Stub. See module docstring -- use HARL upstream for MADDPG."""

    def __init__(self, *args, **kwargs):               # noqa: D401
        raise NotImplementedError(
            "MADDPG is not implemented standalone in this repository. "
            "Install HARL (https://github.com/PKU-MARL/HARL) and use its "
            "MADDPG implementation directly. We deliberately do not "
            "approximate it via TySAC with low alpha -- that would not be "
            "a faithful baseline."
        )


__all__ = ["MADDPG"]
