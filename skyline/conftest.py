"""Pytest/import reuse shim for the skyline bench.

`hsreloc` is a top-level module that *reuses* the evaluator (`naveval`, under `evaluation/`) and the
skyline profile package (`skyline`, under `skyline/skyline/`) across a documented boundary without
being nested in either. This shim puts those source roots on `sys.path` so
`from naveval.frames import ...` and `from skyline.extraction import ...` resolve natively when
pytest's rootdir is this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REUSE_ROOTS = [
    _REPO_ROOT / "evaluation",                 # -> naveval
    Path(__file__).resolve().parent,           # -> hsreloc and skyline
]

for _root in _REUSE_ROOTS:
    _s = str(_root)
    if _root.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
