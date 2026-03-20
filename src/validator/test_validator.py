"""
Test validator — checks generated test code for structural correctness.

Validates that generated tests contain the right imports, assertions,
and framework-specific structure before accepting them as output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Outcome of validating generated test code."""

    is_valid: bool = True
    issues: list[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.is_valid = False


# ── Framework-specific patterns ──────────────────────────────────────────────

# Maps framework name → (import patterns, assertion patterns)
_FRAMEWORK_RULES: dict[str, dict] = {
    "pytest": {
        "import_patterns": [
            r"\bimport\s+pytest\b",
            r"\bfrom\s+pytest\b",
        ],
        "assertion_patterns": [
            r"\bassert\s+",
            r"\bpytest\.raises\b",
        ],
        "structure_patterns": [
            r"\bdef\s+test_",
        ],
        "structure_label": "function named 'test_*'",
    },
    "jest": {
        "import_patterns": [
            r"\brequire\s*\(\s*['\"]",
            r"\bimport\s+",
        ],
        "assertion_patterns": [
            r"\bexpect\s*\(",
            r"\bassert\b",
        ],
        "structure_patterns": [
            r"\b(?:describe|it|test)\s*\(",
        ],
        "structure_label": "describe/it/test block",
    },
    "junit": {
        "import_patterns": [
            r"\bimport\s+org\.junit\b",
            r"\bimport\s+static\s+org\.junit\b",
        ],
        "assertion_patterns": [
            r"\bassertEquals?\s*\(",
            r"\bassertTrue\s*\(",
            r"\bassertFalse\s*\(",
            r"\bassertThrows\s*\(",
            r"\bassertNotNull\s*\(",
        ],
        "structure_patterns": [
            r"@Test\b",
        ],
        "structure_label": "@Test annotation",
    },
    "gotest": {
        "import_patterns": [
            r'\bimport\s*\(',
            r'"testing"',
        ],
        "assertion_patterns": [
            r"\bt\.(?:Error|Fatal|Fail|Log)\b",
            r"\bassert\b",
        ],
        "structure_patterns": [
            r"\bfunc\s+Test[A-Z]",
        ],
        "structure_label": "func Test* function",
    },
    "cargo": {
        "import_patterns": [],  # Rust tests don't always need explicit imports
        "assertion_patterns": [
            r"\bassert!\s*\(",
            r"\bassert_eq!\s*\(",
            r"\bassert_ne!\s*\(",
        ],
        "structure_patterns": [
            r"#\[test\]",
        ],
        "structure_label": "#[test] attribute",
    },
}


def validate_test_code(
    code: str,
    language: str,
    test_framework: str,
) -> ValidationResult:
    """
    Validate generated test code for structural correctness.

    Parameters
    ----------
    code : str
        The generated test source code.
    language : str
        Programming language (e.g. ``"python"``).
    test_framework : str
        Test framework name (e.g. ``"pytest"``).

    Returns
    -------
    ValidationResult
        Whether the code is valid and any issues found.
    """
    result = ValidationResult()

    # ── Basic checks ─────────────────────────────────────────────────────
    stripped = code.strip()
    if not stripped:
        result.add_issue("Generated test code is empty.")
        return result

    if len(stripped.splitlines()) < 3:
        result.add_issue("Generated test code is suspiciously short (fewer than 3 lines).")

    # ── Framework-specific checks ────────────────────────────────────────
    rules = _FRAMEWORK_RULES.get(test_framework.lower())
    if rules is None:
        # Unknown framework — only do generic checks
        if not re.search(r"\bassert", code, re.IGNORECASE):
            result.add_issue("No assertion statements found in generated code.")
        return result

    # Check imports (skip if framework has no import requirements)
    if rules["import_patterns"]:
        has_import = any(
            re.search(pat, code) for pat in rules["import_patterns"]
        )
        if not has_import:
            result.add_issue(
                f"Missing {test_framework} import. Expected one of: "
                f"{', '.join(rules['import_patterns'])}"
            )

    # Check assertions
    has_assertion = any(
        re.search(pat, code) for pat in rules["assertion_patterns"]
    )
    if not has_assertion:
        result.add_issue(
            f"No {test_framework} assertion found. "
            f"Expected patterns like: {', '.join(rules['assertion_patterns'])}"
        )

    # Check test structure
    has_structure = any(
        re.search(pat, code) for pat in rules["structure_patterns"]
    )
    if not has_structure:
        result.add_issue(
            f"No {test_framework} test structure found. "
            f"Expected: {rules['structure_label']}"
        )

    return result
