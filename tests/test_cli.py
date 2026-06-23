"""
Integration tests for the TestIQ CLI.

Uses Typer's CliRunner with monkeypatched backends — no real Ollama or ChromaDB.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.rag.indexer import IndexResult

runner = CliRunner()


# ── Sample source file ──────────────────────────────────────────────────────

SAMPLE_PY = textwrap.dedent("""\
    import os

    def hello(name: str) -> str:
        \"\"\"Say hello.\"\"\"
        return f"Hello, {name}!"

    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b
""")

VALID_TEST_CODE = textwrap.dedent("""\
    import pytest
    from sample import hello

    def test_hello():
        assert hello("world") == "Hello, world!"
""")


# ── Version command ──────────────────────────────────────────────────────────

class TestVersionCommand:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "testiq v" in result.output


# ── Index command ────────────────────────────────────────────────────────────

class TestIndexCommand:
    @patch("src.cli._check_ollama")
    @patch("src.cli.Indexer")
    def test_index_success(self, mock_indexer_cls, mock_ollama, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "sample.py").write_text(SAMPLE_PY)

        mock_indexer = MagicMock()
        mock_indexer.index_directory.return_value = IndexResult(
            files_processed=1, chunks_indexed=2
        )
        mock_indexer_cls.return_value = mock_indexer

        result = runner.invoke(app, ["index", str(src_dir)])
        assert result.exit_code == 0
        assert "2" in result.output  # chunks_indexed
        assert "1" in result.output  # files_processed

    def test_index_missing_dir(self):
        result = runner.invoke(app, ["index", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "not a directory" in result.output.lower() or "error" in result.output.lower()


# ── Generate command ─────────────────────────────────────────────────────────

class TestGenerateCommand:
    @patch("src.cli._check_ollama")
    @patch("src.cli.Retriever", side_effect=Exception("no db"))
    @patch("src.cli.generate_functions_tests")
    @patch("src.cli.generate_class_tests")
    @patch("src.cli.generate_tests")
    def test_generate_writes_file(
        self, mock_gen, mock_gen_class, mock_gen_funcs, mock_retriever, mock_ollama, tmp_path
    ):
        from src.workflows.generate import GenerationResult

        src_file = tmp_path / "sample.py"
        src_file.write_text(SAMPLE_PY)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        gen_result = GenerationResult(
            code=VALID_TEST_CODE, is_valid=True, attempts=1
        )
        mock_gen.return_value = gen_result
        mock_gen_class.return_value = gen_result
        mock_gen_funcs.return_value = gen_result

        with patch("src.cli.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.logging.show_spinner = False
            cfg.generation.dry_run = False
            cfg.generation.output_dir = str(tests_dir)
            cfg.parser.language = "python"
            cfg.get_test_framework.return_value = "pytest"
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["generate", str(src_file)])

        assert result.exit_code == 0
        assert "Done!" in result.output

    @patch("src.cli._check_ollama")
    @patch("src.cli.Retriever", side_effect=Exception("no db"))
    @patch("src.cli.generate_functions_tests")
    @patch("src.cli.generate_class_tests")
    @patch("src.cli.generate_tests")
    def test_generate_dry_run(
        self, mock_gen, mock_gen_class, mock_gen_funcs, mock_retriever, mock_ollama, tmp_path
    ):
        from src.workflows.generate import GenerationResult

        src_file = tmp_path / "sample.py"
        src_file.write_text(SAMPLE_PY)

        gen_result = GenerationResult(
            code=VALID_TEST_CODE, is_valid=True, attempts=1
        )
        mock_gen.return_value = gen_result
        mock_gen_class.return_value = gen_result
        mock_gen_funcs.return_value = gen_result

        with patch("src.cli.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.logging.show_spinner = False
            cfg.generation.dry_run = False
            cfg.generation.output_dir = str(tmp_path / "tests")
            cfg.parser.language = "python"
            cfg.get_test_framework.return_value = "pytest"
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["generate", str(src_file), "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output
        # No test file should have been written
        assert not (tmp_path / "tests" / "test_sample.py").exists()

    @patch("src.cli._check_ollama")
    @patch("src.cli.Retriever", side_effect=Exception("no db"))
    @patch("src.cli.generate_functions_tests")
    @patch("src.cli.generate_class_tests")
    @patch("src.cli.generate_tests")
    def test_generate_writes_multiple_files_for_classes(
        self, mock_gen, mock_gen_class, mock_gen_funcs, mock_retriever, mock_ollama, tmp_path
    ):
        from src.workflows.generate import GenerationResult

        src_content = textwrap.dedent("""\
            class Calculator:
                def add(self, a, b):
                    \"\"\"Add.\"\"\"
                    return a + b
                def sub(self, a, b):
                    \"\"\"Sub.\"\"\"
                    return a - b

            def hello():
                \"\"\"Hello.\"\"\"
                return "hello"
        """)
        src_file = tmp_path / "sample_classes.py"
        src_file.write_text(src_content, encoding="utf-8")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(exist_ok=True)

        gen_result_class = GenerationResult(
            code="class CalculatorTests", is_valid=True, attempts=1
        )
        gen_result_funcs = GenerationResult(
            code="def test_hello()", is_valid=True, attempts=1
        )
        mock_gen_class.return_value = gen_result_class
        mock_gen_funcs.return_value = gen_result_funcs

        with patch("src.cli.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.logging.show_spinner = False
            cfg.generation.dry_run = False
            cfg.generation.output_dir = str(tests_dir)
            cfg.parser.language = "python"
            cfg.get_test_framework.return_value = "pytest"
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["generate", str(src_file)])

        assert result.exit_code == 0
        assert "Done!" in result.output

        assert (tests_dir / "test_calculator.py").exists()
        assert (tests_dir / "test_sample_classes_functions.py").exists()

        assert (tests_dir / "test_calculator.py").read_text() == "class CalculatorTests"
        assert (tests_dir / "test_sample_classes_functions.py").read_text() == "def test_hello()"

    def test_generate_missing_file(self):
        result = runner.invoke(app, ["generate", "/nonexistent/file.py"])
        assert result.exit_code == 1
        assert "error" in result.output.lower() or "not found" in result.output.lower()


# ── Explain command ──────────────────────────────────────────────────────────

class TestExplainCommand:
    @patch("src.cli._check_ollama")
    @patch("src.cli.explain_failure")
    @patch("subprocess.run")
    def test_explain_prints_analysis(
        self, mock_subprocess, mock_explain, mock_ollama, tmp_path
    ):
        from src.workflows.explain import ExplanationResult

        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def test_fail(): assert False")

        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="FAILED test_sample.py::test_fail - AssertionError",
            stderr="",
        )

        mock_explain.return_value = ExplanationResult(
            error_summary="The assertion is wrong.",
            bug_location="test_sample.py, line 1.",
            suggested_fix="Change assert False to assert True.",
        )

        with patch("src.cli.load_config") as mock_cfg:
            cfg = MagicMock()
            cfg.logging.show_spinner = False
            cfg.llm.base_url = "http://localhost:11434"
            cfg.llm.model = "test"
            cfg.embeddings.model = "test"
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["explain", str(test_file)])

        assert result.exit_code == 0
        assert "Error Summary" in result.output
        assert "Bug Location" in result.output
        assert "Suggested Fix" in result.output


# ── Scan command ─────────────────────────────────────────────────────────────

class TestScanCommand:
    def test_scan_table_output(self, tmp_path):
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "app.py").write_text(SAMPLE_PY)

        result = runner.invoke(app, ["scan", str(src_dir)])
        assert result.exit_code == 0
        assert "Coverage" in result.output or "coverage" in result.output

    def test_scan_json_output(self, tmp_path):
        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "app.py").write_text(SAMPLE_PY)

        result = runner.invoke(app, ["scan", str(src_dir), "--output", "json"])
        assert result.exit_code == 0

        # Parse the JSON from output
        output_lines = result.output.strip()
        data = json.loads(output_lines)
        assert "total_functions" in data
        assert "untested" in data

    def test_scan_missing_dir(self):
        result = runner.invoke(app, ["scan", "/nonexistent/path"])
        assert result.exit_code == 1
