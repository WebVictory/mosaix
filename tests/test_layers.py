"""The layers import inward only: domain knows nothing else, services know the domain and their own ports,
adapters (repositories, infrastructure) know the domain, and the app wires everything together."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

ALLOWED = {
    "domain": {"domain"},
    "services": {"domain", "services"},
    "repositories": {"domain", "repositories"},
    "infrastructure": {"domain", "infrastructure"},
}
LAYERS = {*ALLOWED, "app"}


def _layers_imported(path: Path) -> set[str]:
    layers = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        layers |= {name.split(".")[0] for name in names} & LAYERS
    return layers


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_a_layer_imports_inward_only(layer: str) -> None:
    leaks = {
        str(path.relative_to(SRC)): sorted(imported - ALLOWED[layer])
        for path in (SRC / layer).rglob("*.py")
        if (imported := _layers_imported(path)) - ALLOWED[layer]
    }
    assert leaks == {}
