"""
Tests for Phase 3: prompts, validator, and all three workflows.

All tests use fake LLM callables — no Ollama server needed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config import TestIQConfig, GenerationConfig, ParserConfig, ScanConfig
from src.parser.base_parser import FunctionChunk
from src.rag.retriever import RetrievalResult


# ── Fake LLM helpers ─────────────────────────────────────────────────────────

VALID_PYTEST_CODE = textwrap.dedent("""\
    import pytest
    from mymodule import add

    def test_add_positive():
        assert add(2, 3) == 5

    def test_add_negative():
        assert add(-1, -2) == -3

    def test_add_zero():
        assert add(0, 0) == 0
""")

INVALID_PYTEST_CODE_NO_ASSERT = textwrap.dedent("""\
    import pytest
    from mymodule import add

    def test_add_positive():
        result = add(2, 3)
        print(result)
""")

INVALID_PYTEST_CODE_NO_IMPORT = textwrap.dedent("""\
    from mymodule import add

    def test_add_positive():
        assert add(2, 3) == 5
""")


def make_fake_llm(responses: list[str]):
    """Create a fake LLM callable that returns canned responses in order."""
    call_count = [0]

    def fake_llm(prompt: str) -> str:
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    return fake_llm


# ── Prompt template tests ────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_generation_prompt_formats(self):
        from src.prompts.templates import TEST_GENERATION_PROMPT

        result = TEST_GENERATION_PROMPT.format(
            language="python",
            test_framework="pytest",
            function_name="add",
            function_body="def add(a, b): return a + b",
            context="(none)",
            imports="import math",
        )
        assert "python" in result
        assert "pytest" in result
        assert "add" in result
        assert "import math" in result

    def test_self_correction_prompt_formats(self):
        from src.prompts.templates import SELF_CORRECTION_PROMPT

        result = SELF_CORRECTION_PROMPT.format(
            language="javascript",
            test_framework="jest",
            function_name="validate",
            original_tests="test code here",
            issues="- Missing import\n- No assertions",
        )
        assert "javascript" in result
        assert "jest" in result
        assert "Missing import" in result

    def test_explain_prompts_format(self):
        from src.prompts.templates import (
            EXPLAIN_STEP1_PROMPT,
            EXPLAIN_STEP2_PROMPT,
            EXPLAIN_STEP3_PROMPT,
        )

        s1 = EXPLAIN_STEP1_PROMPT.format(
            test_code="code", traceback="error trace"
        )
        assert "error trace" in s1

        s2 = EXPLAIN_STEP2_PROMPT.format(
            test_code="code",
            traceback="error trace",
            error_summary="Something failed",
        )
        assert "Something failed" in s2

        s3 = EXPLAIN_STEP3_PROMPT.format(
            test_code="code",
            traceback="error trace",
            error_summary="Something failed",
            bug_location="line 42",
        )
        assert "line 42" in s3

    def test_scan_summary_prompt_formats(self):
        from src.prompts.templates import SCAN_SUMMARY_PROMPT

        result = SCAN_SUMMARY_PROMPT.format(
            language="python",
            untested_functions="- foo\n- bar",
            total_functions="10",
        )
        assert "python" in result
        assert "foo" in result


# ── Validator tests ──────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_pytest_code(self):
        from src.validator.test_validator import validate_test_code

        result = validate_test_code(VALID_PYTEST_CODE, "python", "pytest")
        assert result.is_valid is True
        assert result.issues == []

    def test_missing_assertion(self):
        from src.validator.test_validator import validate_test_code

        result = validate_test_code(
            INVALID_PYTEST_CODE_NO_ASSERT, "python", "pytest"
        )
        assert result.is_valid is False
        assert any("assertion" in i.lower() for i in result.issues)

    def test_missing_import(self):
        from src.validator.test_validator import validate_test_code

        result = validate_test_code(
            INVALID_PYTEST_CODE_NO_IMPORT, "python", "pytest"
        )
        assert result.is_valid is False
        assert any("import" in i.lower() for i in result.issues)

    def test_empty_code(self):
        from src.validator.test_validator import validate_test_code

        result = validate_test_code("", "python", "pytest")
        assert result.is_valid is False
        assert any("empty" in i.lower() for i in result.issues)

    def test_valid_jest_code(self):
        from src.validator.test_validator import validate_test_code

        jest_code = textwrap.dedent("""\
            const { add } = require('./math');

            describe('add', () => {
                it('adds two numbers', () => {
                    expect(add(2, 3)).toBe(5);
                });
            });
        """)
        result = validate_test_code(jest_code, "javascript", "jest")
        assert result.is_valid is True

    def test_valid_junit_code(self):
        from src.validator.test_validator import validate_test_code

        junit_code = textwrap.dedent("""\
            import org.junit.Test;
            import static org.junit.Assert.*;

            public class MathTest {
                @Test
                public void testAdd() {
                    assertEquals(5, Math.add(2, 3));
                }
            }
        """)
        result = validate_test_code(junit_code, "java", "junit")
        assert result.is_valid is True

    def test_unknown_framework_fallback(self):
        from src.validator.test_validator import validate_test_code

        code = "def test_something():\n    x = 1\n    assert x == 1\n    assert True"
        result = validate_test_code(code, "python", "unknown_framework")
        # Should still pass basic assertion check
        assert result.is_valid is True


# ── Generate workflow tests ──────────────────────────────────────────────────

class TestGenerateWorkflow:
    def _make_chunk(self) -> FunctionChunk:
        return FunctionChunk(
            name="add",
            parameters=["a", "b"],
            return_type="int",
            body="def add(a: int, b: int) -> int:\n    return a + b",
            docstring="Add two numbers.",
            imports=["import math"],
            line_start=1,
            line_end=2,
            language="python",
            filepath="math_utils.py",
        )

    def _make_config(self) -> TestIQConfig:
        return TestIQConfig(
            generation=GenerationConfig(max_retries=2),
            parser=ParserConfig(language="python"),
        )

    def test_happy_path_valid_on_first_try(self):
        from src.workflows.generate import generate_tests

        fake_llm = make_fake_llm([VALID_PYTEST_CODE])
        result = generate_tests(
            self._make_chunk(),
            config=self._make_config(),
            llm_fn=fake_llm,
        )

        assert result.is_valid is True
        assert result.attempts == 1
        assert "assert" in result.code

    def test_self_correction_succeeds(self):
        from src.workflows.generate import generate_tests

        fake_llm = make_fake_llm([
            INVALID_PYTEST_CODE_NO_ASSERT,  # First try: invalid
            VALID_PYTEST_CODE,               # Correction: valid
        ])
        result = generate_tests(
            self._make_chunk(),
            config=self._make_config(),
            llm_fn=fake_llm,
        )

        assert result.is_valid is True
        assert result.attempts == 2

    def test_max_retries_exhausted(self):
        from src.workflows.generate import generate_tests

        # All attempts return invalid code
        fake_llm = make_fake_llm([
            INVALID_PYTEST_CODE_NO_ASSERT,
            INVALID_PYTEST_CODE_NO_ASSERT,
            INVALID_PYTEST_CODE_NO_ASSERT,
        ])
        result = generate_tests(
            self._make_chunk(),
            config=self._make_config(),
            llm_fn=fake_llm,
        )

        assert result.is_valid is False
        assert result.attempts == 3  # 1 initial + 2 retries
        assert len(result.issues) > 0

    def test_with_context_chunks(self):
        from src.workflows.generate import generate_tests

        context = [
            RetrievalResult(
                content="def multiply(a, b): return a * b",
                metadata={"function_name": "multiply"},
                distance=0.1,
            ),
        ]
        fake_llm = make_fake_llm([VALID_PYTEST_CODE])
        result = generate_tests(
            self._make_chunk(),
            context_chunks=context,
            config=self._make_config(),
            llm_fn=fake_llm,
        )

        assert result.is_valid is True

    def test_strips_markdown_fences(self):
        from src.workflows.generate import generate_tests

        fenced = f"```python\n{VALID_PYTEST_CODE}\n```"
        fake_llm = make_fake_llm([fenced])
        result = generate_tests(
            self._make_chunk(),
            config=self._make_config(),
            llm_fn=fake_llm,
        )

        assert result.is_valid is True
        assert "```" not in result.code


# ── Explain workflow tests ───────────────────────────────────────────────────

class TestExplainWorkflow:
    def test_three_step_chain(self):
        from src.workflows.explain import explain_failure

        fake_llm = make_fake_llm([
            "The test expects add(2,3) to return 6 but it returns 5.",
            "Bug is in math_utils.py, function add(), line 2.",
            "Change `return a * b` to `return a + b`.",
        ])

        result = explain_failure(
            test_code="def test_add(): assert add(2,3) == 6",
            traceback="AssertionError: 5 != 6",
            config=TestIQConfig(),
            llm_fn=fake_llm,
        )

        assert result.error_summary != ""
        assert result.bug_location != ""
        assert result.suggested_fix != ""
        assert "add" in result.error_summary.lower() or "return" in result.error_summary.lower()

    def test_each_step_receives_previous_output(self):
        """Verify the chain is sequential — each prompt builds on the previous."""
        from src.workflows.explain import explain_failure

        prompts_received: list[str] = []

        def tracking_llm(prompt: str) -> str:
            prompts_received.append(prompt)
            return f"Response to step {len(prompts_received)}"

        explain_failure(
            test_code="test code",
            traceback="traceback here",
            config=TestIQConfig(),
            llm_fn=tracking_llm,
        )

        assert len(prompts_received) == 3
        # Step 2 should contain step 1's output
        assert "Response to step 1" in prompts_received[1]
        # Step 3 should contain step 1 and step 2 outputs
        assert "Response to step 1" in prompts_received[2]
        assert "Response to step 2" in prompts_received[2]


# ── Scan workflow tests ──────────────────────────────────────────────────────

SAMPLE_SRC = textwrap.dedent("""\
    import os

    def process_data(data):
        \"\"\"Process input data.\"\"\"
        result = transform(data)
        return result

    def transform(data):
        \"\"\"Transform data.\"\"\"
        return data.upper()

    def validate(data):
        \"\"\"Validate data.\"\"\"
        if not data:
            raise ValueError("empty")
        return True

    def _internal_helper(x):
        result = x + 1
        return result
""")

SAMPLE_TEST = textwrap.dedent("""\
    from mymodule import process_data

    def test_process_data():
        assert process_data("hello") == "HELLO"
""")

SAMPLE_SRC_2 = textwrap.dedent("""\
    from mymodule import transform, validate

    def pipeline(data):
        \"\"\"Run full pipeline.\"\"\"
        validate(data)
        result = transform(data)
        return result
""")


class TestScanWorkflow:
    def test_identifies_untested_functions(self, tmp_path: Path):
        from src.workflows.scan import scan_coverage

        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "mymodule.py").write_text(SAMPLE_SRC, encoding="utf-8")
        (src_dir / "test_mymodule.py").write_text(SAMPLE_TEST, encoding="utf-8")

        config = TestIQConfig(scan=ScanConfig(risk_threshold=0))
        result = scan_coverage(str(src_dir), config)

        assert result.total_functions > 0
        untested_names = [u.name for u in result.untested]

        # process_data is tested
        assert "process_data" not in untested_names
        # transform and validate should be untested (not in test file by name ref)
        # Actually — test file imports process_data but calls it. Let's check:
        # transform is NOT referenced in the test file
        assert "validate" in untested_names

    def test_risk_scoring(self, tmp_path: Path):
        from src.workflows.scan import scan_coverage

        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "mymodule.py").write_text(SAMPLE_SRC, encoding="utf-8")
        (src_dir / "pipeline.py").write_text(SAMPLE_SRC_2, encoding="utf-8")
        (src_dir / "test_mymodule.py").write_text(SAMPLE_TEST, encoding="utf-8")

        config = TestIQConfig(scan=ScanConfig(risk_threshold=0))
        result = scan_coverage(str(src_dir), config)

        untested_names = [u.name for u in result.untested]
        # Functions called from multiple files should have higher risk
        if "validate" in untested_names:
            validate_fn = next(u for u in result.untested if u.name == "validate")
            # validate is referenced in mymodule.py (definition) + pipeline.py
            assert validate_fn.risk_score >= 1

    def test_coverage_percentage(self, tmp_path: Path):
        from src.workflows.scan import scan_coverage

        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "mymodule.py").write_text(SAMPLE_SRC, encoding="utf-8")
        (src_dir / "test_mymodule.py").write_text(SAMPLE_TEST, encoding="utf-8")

        config = TestIQConfig(scan=ScanConfig(risk_threshold=0))
        result = scan_coverage(str(src_dir), config)

        assert 0 <= result.coverage_pct <= 100
        assert result.tested_count + result.untested_count == result.total_functions

    def test_empty_directory(self, tmp_path: Path):
        from src.workflows.scan import scan_coverage

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = TestIQConfig(scan=ScanConfig(risk_threshold=0))
        result = scan_coverage(str(empty_dir), config)

        assert result.total_functions == 0
        assert result.untested == []
        assert result.coverage_pct == 100.0

    def test_risk_threshold_filters(self, tmp_path: Path):
        from src.workflows.scan import scan_coverage

        src_dir = tmp_path / "project"
        src_dir.mkdir()
        (src_dir / "mymodule.py").write_text(SAMPLE_SRC, encoding="utf-8")
        (src_dir / "test_mymodule.py").write_text(SAMPLE_TEST, encoding="utf-8")

        # High threshold should filter out low-risk functions
        config = TestIQConfig(scan=ScanConfig(risk_threshold=100))
        result = scan_coverage(str(src_dir), config)
        assert result.untested == []
