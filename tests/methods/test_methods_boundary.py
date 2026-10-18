"""Static boundary test: ``experimental/methods`` imports only stdlib, third-party and the core.

PROTOCOL.md: no cross-module imports except downward onto contracts; methods-core is registered with
``deps = []``, so its only imports are stdlib, third-party and the ai_functions core. Parsed with
``ast`` so the guard fires on an import that is never executed (a ``TYPE_CHECKING`` block, a lazy
import inside a function).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import ai_functions.experimental.methods as methods_pkg

PACKAGE = Path(methods_pkg.__file__).parent
OWN = "ai_functions.experimental.methods"
ALLOWED_THIRD_PARTY = frozenset({"strands", "tstr"})


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative: resolve against the package so `from .core import x` counts as our own.
                base = OWN.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                found.append(".".join([*base, node.module] if node.module else base))
            else:
                found.append(node.module or "")
    return found


def test_methods_imports_nothing_outside_the_allowed_set() -> None:
    """Every module under ``experimental/methods`` imports only stdlib, third-party, the ai_functions core, itself."""
    files = sorted(PACKAGE.glob("*.py"))
    assert files, f"no modules found under {PACKAGE}"
    offenders: list[str] = []
    for path in files:
        for name in _imports(path):
            root = name.split(".")[0]
            if name.startswith("ai_functions.experimental") and not (name == OWN or name.startswith(OWN + ".")):
                offenders.append(f"{path.name}: {name} (a sibling experimental package)")
            elif root != "ai_functions" and root not in sys.stdlib_module_names and root not in ALLOWED_THIRD_PARTY:
                offenders.append(f"{path.name}: {name} (outside stdlib, {sorted(ALLOWED_THIRD_PARTY)} and the core)")
    assert offenders == [], offenders
