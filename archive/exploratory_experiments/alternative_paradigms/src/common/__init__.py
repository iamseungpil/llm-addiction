"""Redirect for archived runners: the `common` package now lives in experiments/shared/common/.

The investment-choice and mystery-wheel runners (paper code) moved to experiments/ and took
this shared harness with them. The archived paradigms here (blackjack, coin flip, card flip,
dice rolling, ...) still `import common` from this folder, so this file points the package
at the real copy instead of duplicating it. See PATH_MAP.md at the repository root.
"""
from pathlib import Path as _Path

# archive/exploratory_experiments/alternative_paradigms/src/common/__init__.py -> repo root
_REAL = _Path(__file__).resolve().parents[5] / "experiments" / "shared" / "common"
__path__ = [str(_REAL)]  # submodules (common.utils, common.model_loader, ...) load from there
exec(compile((_REAL / "__init__.py").read_text(), str(_REAL / "__init__.py"), "exec"))
