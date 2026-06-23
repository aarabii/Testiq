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
# ── Class Test Generation ───────────────────────────────────────────────────

CLASS_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "class_name",
        "class_methods",
        "context",
        "imports",
    ],
    template="""\
You are a senior {language} engineer specialising in writing thorough, \
production-quality tests using {test_framework}.

## Target Class under Test
**Name:** `{class_name}`
**Methods and Implementations:**
```{language}
{class_methods}
```

## File Imports
```{language}
{imports}
```

## Related Context (helper functions, classes, constants)
{context}

## Instructions
Generate a complete {test_framework} test file for class `{class_name}`.

Requirements:
1. Import the class under test and any necessary dependencies.
2. Include the correct {test_framework} imports and structure (such as test classes, setup/teardown if appropriate).
3. Cover all methods: happy path, edge cases, boundary values, and error cases for each method.
4. Each test should have a descriptive name explaining what it verifies.
5. Use assertions appropriate for {test_framework}.
6. Do NOT include any explanatory text — output ONLY valid {language} code.
""",
)

CLASS_SELF_CORRECTION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "class_name",
        "original_tests",
        "issues",
    ],
    template="""\
You are a senior {language} engineer. The following {test_framework} tests \
for class `{class_name}` have validation issues that must be fixed.

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

# ── Standalone Functions Test Generation ──────────────────────────────────────

FUNCTIONS_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "file_name",
        "functions_body",
        "context",
        "imports",
    ],
    template="""\
You are a senior {language} engineer specialising in writing thorough, \
production-quality tests using {test_framework}.

## Target Functions in `{file_name}`
**Functions and Implementations:**
```{language}
{functions_body}
```

## File Imports
```{language}
{imports}
```

## Related Context (helper functions, classes, constants)
{context}

## Instructions
Generate a complete {test_framework} test file for these functions.

Requirements:
1. Import the functions under test and any necessary dependencies.
2. Include the correct {test_framework} imports and structure.
3. Cover: happy path, edge cases, boundary values, and error cases.
4. Each test should have a descriptive name explaining what it verifies.
5. Use assertions appropriate for {test_framework}.
6. Do NOT include any explanatory text — output ONLY valid {language} code.
""",
)

FUNCTIONS_SELF_CORRECTION_PROMPT = PromptTemplate(
    input_variables=[
        "language",
        "test_framework",
        "file_name",
        "original_tests",
        "issues",
    ],
    template="""\
You are a senior {language} engineer. The following {test_framework} tests \
for functions in `{file_name}` have validation issues that must be fixed.

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


# ── Run Instructions ──────────────────────────────────────────────────────────

RUN_INSTRUCTION_PROMPT = PromptTemplate(
    input_variables=["language", "test_framework", "test_files"],
    template="""\
You are a devops and testing engineer. A user has generated unit tests for a {language} codebase using {test_framework}.
The generated test files are:
{test_files}

Provide clear, step-by-step instructions on how the user can run these tests in their terminal.
Include the exact commands and any necessary setup steps.
Keep it concise and output using clean markdown format.
""",
)


# ── Assume Analyzer ───────────────────────────────────────────────────────────

ASSUME_PROMPT = PromptTemplate(
    input_variables=["file_or_dir", "code_structure"],
    template="""\
You are a senior QA engineer and testing analyst.
Analyze the following code structure and contents of {file_or_dir}:

{code_structure}

Predict the test scenarios and outcomes:
1. What will happen if it's going to pass (Happy Path scenario)?
2. What will happen if it's going to fail (e.g., typical bugs, bad input)?
3. Identify exactly where it is most likely to fail (Critical Failure Points).
4. What are the key edge cases we must verify?

Produce a visually understandable, structured markdown report with clear headings and bullet points.
""",
)
