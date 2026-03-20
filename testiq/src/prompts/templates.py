"""
Prompt templates for all TestIQ workflows.

Every prompt used by the system lives here. No prompt string is ever
hardcoded inside workflow files. Language and test framework are injected
dynamically from config.
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


# ── Test Generation ──────────────────────────────────────────────────────────

TEST_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "function_name",
        "function_body",
        "context",
        "imports",
    ],
    template="""\
You are a senior {language} engineer specialising in writing thorough, \
production-quality tests using {test_framework}.

## Target Function
**Name:** `{function_name}`
```{language}
{function_body}
```

## File Imports
```{language}
{imports}
```

## Related Context (helper functions, classes, constants)
{context}

## Instructions
Generate a complete {test_framework} test file for `{function_name}`.

Requirements:
1. Import the function under test and any necessary dependencies.
2. Include the correct {test_framework} imports and structure.
3. Cover: happy path, edge cases, boundary values, and error cases.
4. Each test should have a descriptive name explaining what it verifies.
5. Use assertions appropriate for {test_framework}.
6. Do NOT include any explanatory text — output ONLY valid {language} code.
""",
)


# ── Self-Correction ──────────────────────────────────────────────────────────

SELF_CORRECTION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "function_name",
        "original_tests",
        "issues",
    ],
    template="""\
You are a senior {language} engineer. The following {test_framework} tests \
for `{function_name}` have validation issues that must be fixed.

## Original Tests
```{language}
{original_tests}
```

## Validation Issues Found
{issues}

## Instructions
Fix ALL issues listed above. Output ONLY the corrected {language} test code — \
no explanations, no markdown fences, just valid {language} code. Ensure:
1. Correct {test_framework} imports are present.
2. Every test contains at least one assertion.
3. Test structure follows {test_framework} conventions.
""",
)


# ── Bug Explanation — Step 1: Summarise Error ────────────────────────────────

EXPLAIN_STEP1_PROMPT = PromptTemplate(
    input_variables=["test_code", "traceback"],
    template="""\
You are a debugging expert. A test is failing with the traceback below. \
Summarise the error in 2-3 plain English sentences that a junior developer \
would understand. Do NOT suggest a fix yet — just explain what went wrong.

## Failing Test Code
```
{test_code}
```

## Traceback
```
{traceback}
```

## Your Summary
""",
)


# ── Bug Explanation — Step 2: Locate Bug ─────────────────────────────────────

EXPLAIN_STEP2_PROMPT = PromptTemplate(
    input_variables=["test_code", "traceback", "error_summary"],
    template="""\
You are a debugging expert. Based on the error summary below, identify the \
exact location and root cause of the bug. Be specific — name the file, \
function, and line if possible.

## Error Summary
{error_summary}

## Test Code
```
{test_code}
```

## Traceback
```
{traceback}
```

## Bug Location & Root Cause
""",
)


# ── Bug Explanation — Step 3: Suggest Fix ────────────────────────────────────

EXPLAIN_STEP3_PROMPT = PromptTemplate(
    input_variables=["test_code", "traceback", "error_summary", "bug_location"],
    template="""\
You are a debugging expert. Based on the analysis below, suggest a concrete \
fix. Include a code patch if possible.

## Error Summary
{error_summary}

## Bug Location
{bug_location}

## Test Code
```
{test_code}
```

## Traceback
```
{traceback}
```

## Suggested Fix
""",
)


# ── Coverage Scan Summary ────────────────────────────────────────────────────

SCAN_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["language", "untested_functions", "total_functions"],
    template="""\
You are a testing strategist. A {language} codebase has {total_functions} \
total functions. The following functions have NO tests:

{untested_functions}

Summarise the testing gaps and prioritise which functions should be tested \
first, based on risk and importance. Keep your response concise — bullet \
points preferred.
""",
)
