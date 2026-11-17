"""Guards the layering described in ARCHITECTURE.md: dependencies point downward only."""

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "rag_eval"
SERVICES = ("rag_eval.pipeline", "rag_eval.evaluate", "rag_eval.assistant", "rag_eval.wiring")

# module group -> rag_eval modules it must never import
FORBIDDEN = {
    "core": ("rag_eval.api", "rag_eval.warehouse", "rag_eval.config", "rag_eval.__main__", *SERVICES),
    "warehouse": ("rag_eval.api", "rag_eval.config", "rag_eval.__main__", *SERVICES),
    "services": ("rag_eval.api", "rag_eval.__main__"),
}
GROUPS = {
    "core": sorted((PKG / "core").glob("*.py")),
    "warehouse": sorted((PKG / "warehouse").glob("*.py")),
    "services": [PKG / f"{m.split('.')[-1]}.py" for m in SERVICES] + [PKG / "config.py"],
}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("group,path", [(g, p) for g, paths in GROUPS.items() for p in paths],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_layer_does_not_import_upward(group, path):
    bad = [m for m in _imports(path) for f in FORBIDDEN[group] if m == f or m.startswith(f + ".")]
    assert not bad, f"{path.relative_to(PKG)} ({group}) must not import {bad}"


def test_core_reads_no_environment_except_model_defaults():
    """Only factories and OpenAI model-name defaults may read os.environ in core; the warehouse never does."""
    for path in GROUPS["core"] + GROUPS["warehouse"]:
        if "os.environ" in path.read_text(encoding="utf-8"):
            assert path.name in {"embeddings.py", "generate.py", "router.py", "text_to_sql.py"}, path.name
