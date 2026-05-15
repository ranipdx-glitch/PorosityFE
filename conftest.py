"""Repo-root conftest: makes the top-level modules importable to tests.

Each test file used to do its own `sys.path.insert(0, ...)` to find
`porosity_fe_analysis` / `app` / `validate_porosity_cli` / the
`validation` package. Pytest auto-imports this conftest before collecting
any test, so the boilerplate now lives in exactly one place (#29).

CI installs deps but not the package itself, so the path adjustment is
still needed; once `pip install -e .` becomes part of CI this whole file
can go away.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
