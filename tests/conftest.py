"""
pytest configuration.

The SMACv2 wrapper now refuses to silently fall back to a synthetic dummy
backend in production runs, because that fallback used to produce
plausible-looking learning curves that had nothing to do with the trained
policy. The dummy backend is gated behind the ``TYMARL_ALLOW_DUMMY_SMACV2``
env var.

The unit tests in ``tests/test_envs.py`` and ``tests/test_algorithms.py``
explicitly want the dummy backend (no StarCraft binary installed in CI),
so we set the gating env var here once, before any test module is imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("TYMARL_ALLOW_DUMMY_SMACV2", "1")
