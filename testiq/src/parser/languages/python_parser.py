"""
Python parser — Tree-sitter based implementation of BaseParser.

Extracts functions, methods, imports, docstrings, parameters, and return
type annotations from Python source files.
"""

from __future__ import annotations

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from src.config import load_config
from src.parser.base_parser import BaseParser, FunctionChunk

PY_LANGUAGE = Language(tspython.language())


class PythonParser(BaseParser):
    """Parse Python source files using Tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)
        cfg = load_config()
        self._skip_dunder = cfg.parser.skip_dunder_methods
        self._skip_private = cfg.parser.skip_private_methods
        self._min_lines = cfg.parser.min_function_lines

    # ── BaseParser interface ─────────────────────────────────────────────

    def get_language(self) -> str:
        return "python"

    def parse_file(self, filepath: str) -> list[FunctionChunk]:
        source = Path(filepath).read_bytes()
        tree = self._parser.parse(source)
        source_text = source.decode("utf-8", errors="replace")
        source_lines = source_text.splitlines()

        chunks: list[FunctionChunk] = []
        self._walk(tree.root_node, source_lines, filepath, chunks)
        return chunks

    def extract_imports(self, filepath: str) -> list[str]:
        source = Path(filepath).read_bytes()
        tree = self._parser.parse(source)
        source_text = source.decode("utf-8", errors="replace")
        source_lines = source_text.splitlines()

        imports: list[str] = []
        for node in tree.root_node.children:
            if node.type in ("import_statement", "import_from_statement"):
                start = node.start_point[0]
                end = node.end_point[0]
                text = "\n".join(source_lines[start : end + 1]).strip()
                imports.append(text)
        return imports

    # ── Internal helpers ─────────────────────────────────────────────────

    def _walk(
        self,
        node,
        source_lines: list[str],
        filepath: str,
        out: list[FunctionChunk],
    ) -> None:
        """Recursively walk the AST and collect function/method definitions."""
        if node.type in ("function_definition", "decorated_definition"):
            func_node = node
            # For decorated definitions, dig into the actual function
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type == "function_definition":
                        func_node = child
                        break
                else:
                    # No function_definition child — skip
                    return

            chunk = self._extract_function(func_node, source_lines, filepath)
            if chunk is not None:
                out.append(chunk)
            # Don't recurse inside functions — nested funcs are skipped
            return

        for child in node.children:
            self._walk(child, source_lines, filepath, out)

    def _extract_function(
        self,
        node,
        source_lines: list[str],
        filepath: str,
    ) -> FunctionChunk | None:
        """Extract a FunctionChunk from a function_definition node."""
        name = ""
        parameters: list[str] = []
        return_type: str | None = None
        docstring: str | None = None

        for child in node.children:
            if child.type == "identifier":
                name = child.text.decode()
            elif child.type == "parameters":
                parameters = self._extract_parameters(child)
            elif child.type == "type":
                return_type = child.text.decode()
            elif child.type == "block":
                docstring = self._extract_docstring(child)

        # ── Filtering ────────────────────────────────────────────────────
        if not name:
            return None
        if self._skip_dunder and name.startswith("__") and name.endswith("__"):
            return None
        if self._skip_private and name.startswith("_") and not name.startswith("__"):
            return None

        line_start = node.start_point[0] + 1  # 1-indexed
        line_end = node.end_point[0] + 1
        func_line_count = line_end - line_start + 1
        if func_line_count < self._min_lines:
            return None

        body = "\n".join(source_lines[node.start_point[0] : node.end_point[0] + 1])

        return FunctionChunk(
            name=name,
            parameters=parameters,
            return_type=return_type,
            body=body,
            docstring=docstring,
            imports=[],
            line_start=line_start,
            line_end=line_end,
            language="python",
            filepath=filepath,
        )

    @staticmethod
    def _extract_parameters(params_node) -> list[str]:
        """Extract parameter names from a parameters node."""
        result: list[str] = []
        for child in params_node.children:
            if child.type == "identifier":
                result.append(child.text.decode())
            elif child.type in (
                "default_parameter",
                "typed_parameter",
                "typed_default_parameter",
            ):
                # First identifier child is the param name
                for sub in child.children:
                    if sub.type == "identifier":
                        result.append(sub.text.decode())
                        break
            elif child.type == "list_splat_pattern":
                for sub in child.children:
                    if sub.type == "identifier":
                        result.append(f"*{sub.text.decode()}")
                        break
            elif child.type == "dictionary_splat_pattern":
                for sub in child.children:
                    if sub.type == "identifier":
                        result.append(f"**{sub.text.decode()}")
                        break
        return result

    @staticmethod
    def _extract_docstring(block_node) -> str | None:
        """Extract the docstring (first expression_statement → string) from a block."""
        if not block_node.children:
            return None
        first = block_node.children[0]
        if first.type == "expression_statement":
            for child in first.children:
                if child.type == "string":
                    raw = child.text.decode()
                    # Strip triple-quotes
                    for quote in ('"""', "'''"):
                        if raw.startswith(quote) and raw.endswith(quote):
                            return raw[3:-3].strip()
                    # Single-quoted string used as docstring (unusual but valid)
                    return raw.strip("\"'").strip()
        return None


# ── Auto-register with the language registry ─────────────────────────────────

def _register() -> None:
    from src.parser.language_registry import registry
    registry.register("python", PythonParser)


_register()
