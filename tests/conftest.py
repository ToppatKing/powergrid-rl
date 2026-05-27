"""Shared pytest configuration for powergrid-rl.

Inserts the project root onto sys.path so that the ``env`` and ``ppo``
packages are importable regardless of how pytest is invoked (e.g. from
the project root, from the tests/ subdirectory, or via CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path.
# This is necessary because env/ and ppo/ live at the root level, not
# inside a src/ layout, so pytest cannot find them without this.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
