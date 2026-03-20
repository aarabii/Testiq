"""
Tests for the Python parser, config loader, and language registry.
"""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_PYTHON = textwrap.dedent('''\
    import os
    from pathlib import Path

    GLOBAL_VAR = 42


    def hello(name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"


    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b


    def _private_helper(x):
        result = x * 2
        return result


    def __dunder__():
        pass


    def tiny():
        pass


    class Calculator:
        def multiply(self, a, b):
            """Multiply two numbers."""
            return a * b

        def _internal(self):
            return None
''')


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Write a sample Python file and return its path."""
    p = tmp_path / "sample.py"
    p.write_text(SAMPLE_PYTHON, encoding="utf-8")
    return p


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal testiq.config.toml and return its path."""
    cfg = tmp_path / "testiq.config.toml"
    cfg.write_text(textwrap.dedent("""\
        [parser]
        language = "python"
        skip_dunder_methods = true
        skip_private_methods = false
        min_function_lines = 3

        [languages]
        python = "pytest"
    """), encoding="utf-8")
    return cfg


# ── Config tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_defaults(self):
        """Loading without a file returns sensible defaults."""
        from src.config import load_config

        cfg = load_config(path="nonexistent.toml")
        assert cfg.parser.language == "python"
        assert cfg.llm.provider == "ollama"
        assert cfg.languages["python"] == "pytest"

    def test_load_from_file(self, sample_config: Path):
        from src.config import load_config

        cfg = load_config(path=str(sample_config))
        assert cfg.parser.language == "python"
        assert cfg.parser.skip_dunder_methods is True
        assert cfg.parser.min_function_lines == 3

    def test_get_test_framework(self):
        from src.config import load_config

        cfg = load_config(path="nonexistent.toml")
        assert cfg.get_test_framework("python") == "pytest"
        assert cfg.get_test_framework("javascript") == "jest"


# ── BaseParser / FunctionChunk tests ─────────────────────────────────────────

class TestFunctionChunk:
    def test_defaults(self):
        from src.parser.base_parser import FunctionChunk

        chunk = FunctionChunk(name="foo")
        assert chunk.name == "foo"
        assert chunk.parameters == []
        assert chunk.return_type is None
        assert chunk.body == ""
        assert chunk.language == ""


# ── Language registry tests ──────────────────────────────────────────────────

class TestLanguageRegistry:
    def test_python_registered(self):
        from src.parser.language_registry import registry

        assert "python" in registry.registered_languages

    def test_get_parser(self):
        from src.parser.language_registry import registry
        from src.parser.languages.python_parser import PythonParser

        parser = registry.get_parser("python")
        assert isinstance(parser, PythonParser)

    def test_get_parser_for_file(self):
        from src.parser.language_registry import registry
        from src.parser.languages.python_parser import PythonParser

        parser = registry.get_parser_for_file("app.py")
        assert isinstance(parser, PythonParser)

    def test_unknown_language_raises(self):
        from src.parser.language_registry import registry

        with pytest.raises(ValueError, match="No parser registered"):
            registry.get_parser("brainfuck")

    def test_unknown_extension_raises(self):
        from src.parser.language_registry import registry

        with pytest.raises(ValueError, match="Unrecognised file extension"):
            registry.get_parser_for_file("main.bf")


# ── Python parser tests ─────────────────────────────────────────────────────

class TestPythonParser:
    def test_get_language(self):
        from src.parser.languages.python_parser import PythonParser

        p = PythonParser()
        assert p.get_language() == "python"

    def test_parse_file_extracts_functions(self, sample_file: Path):
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        chunks = parser.parse_file(str(sample_file))
        names = [c.name for c in chunks]

        # Public functions with >= 3 lines should be present
        assert "hello" in names
        assert "add" in names

        # Dunder should be skipped (skip_dunder_methods=True by default)
        assert "__dunder__" not in names

        # tiny() is only 2 lines → below min_function_lines = 3
        assert "tiny" not in names

    def test_parse_file_extracts_methods(self, sample_file: Path):
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        chunks = parser.parse_file(str(sample_file))
        names = [c.name for c in chunks]

        assert "multiply" in names

    def test_function_chunk_fields(self, sample_file: Path):
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        chunks = parser.parse_file(str(sample_file))
        hello = next(c for c in chunks if c.name == "hello")

        assert hello.language == "python"
        assert hello.filepath == str(sample_file)
        assert hello.docstring == "Say hello."
        assert "name" in hello.parameters
        assert hello.return_type is not None
        assert hello.line_start > 0
        assert hello.line_end >= hello.line_start

    def test_parameters_extracted(self, sample_file: Path):
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        chunks = parser.parse_file(str(sample_file))
        add_chunk = next(c for c in chunks if c.name == "add")

        assert "a" in add_chunk.parameters
        assert "b" in add_chunk.parameters

    def test_extract_imports(self, sample_file: Path):
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        imports = parser.extract_imports(str(sample_file))

        assert any("import os" in i for i in imports)
        assert any("from pathlib import Path" in i for i in imports)

    def test_private_methods_included_by_default(self, sample_file: Path):
        """skip_private_methods defaults to False, so _private_helper should appear."""
        from src.parser.languages.python_parser import PythonParser

        parser = PythonParser()
        chunks = parser.parse_file(str(sample_file))
        names = [c.name for c in chunks]

        assert "_private_helper" in names
