"""
MATD3 -- Multi-Agent TD3 (Ackermann et al., 2019).

This repository does not ship a standalone MATD3 implementation. As with
MADDPG, the algorithm has a reference implementation in
`HARL <https://github.com/PKU-MARL/HARL>`_; use that codebase directly if
you need MATD3 numbers.

The MATD3 class is intentionally not a reduction of TySAC: the twin-Q
clipped target update and policy delay are mechanisms TySAC does not
have, so substituting TySAC with low alpha would not be a faithful
baseline. We raise a clear error instead of papering over it.

Compatible benchmarks (when running upstream HARL)
--------------------------------------------------
MAMuJoCo, MPE, Bi-DexHands -- continuous-action only.
"""

from __future__ import annotations


class MATD3:                                           # noqa: N801
    """Stub. See module docstring -- use HARL upstream for MATD3."""

    def __init__(self, *args, **kwargs):               # noqa: D401
        raise NotImplementedError(
            "MATD3 is not implemented standalone in this repository. "
            "Install HARL (https://github.com/PKU-MARL/HARL) and use its "
            "MATD3 implementation directly."
        )


__all__ = ["MATD3"]
