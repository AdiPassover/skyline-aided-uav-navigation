"""hsreloc — skyline place-evidence bench (research-grade).

Simulator ingest (``simret``), skyline extraction methods (``extraction``), profile matchers
(``matchers``, ``retrieval``) and the study code behind the skyline results.

Reuse boundary: importing this package puts the reused source roots on ``sys.path`` so ``naveval``
(evaluator, ``evaluation/``) and ``skyline`` (profile/descriptor package, ``skyline/skyline/``)
resolve without nesting. Works for the CLI and for pytest alike (conftest.py mirrors it for the
test rootdir).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_BENCH_ROOT = _Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
for _root in (_REPO_ROOT / "evaluation", _BENCH_ROOT):
    _s = str(_root)
    if _root.is_dir() and _s not in _sys.path:
        _sys.path.insert(0, _s)
