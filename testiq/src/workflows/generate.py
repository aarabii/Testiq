"""
Generate workflow — test generation with self-correction loop.

Accepts a FunctionChunk + retrieved context, builds a prompt, calls the LLM,
validates the output, and retries up to max_retries if validation fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from langchain_ollama import ChatOllama

from src.config import TestIQConfig, load_config
from src.parser.base_parser import FunctionChunk
from src.prompts.templates import SELF_CORRECTION_PROMPT, TEST_GENERATION_PROMPT
from src.rag.retriever import RetrievalResult
from src.validator.test_validator import ValidationResult, validate_test_code

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Outcome of a test generation run."""

    code: str = ""
    is_valid: bool = False
    attempts: int = 0
    issues: list[str] = field(default_factory=list)


def _build_context_str(context_chunks: list[RetrievalResult]) -> str:
    """Format retrieved context chunks into a single string for the prompt."""
    if not context_chunks:
        return "(no additional context available)"
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        fn_name = chunk.metadata.get("function_name", "unknown")
        parts.append(f"### Context {i}: {fn_name}\n```\n{chunk.content}\n```")
    return "\n\n".join(parts)


def _default_llm_fn(config: TestIQConfig) -> Callable[[str], str]:
    """Create a default LLM callable using Ollama."""
    llm = ChatOllama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        temperature=config.llm.temperature,
        num_predict=config.llm.max_tokens,
    )

    def call(prompt: str) -> str:
        response = llm.invoke(prompt)
        return response.content  # type: ignore[return-value]

    return call


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if the LLM wraps output in them."""
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def generate_tests(
    chunk: FunctionChunk,
    context_chunks: list[RetrievalResult] | None = None,
    config: TestIQConfig | None = None,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> GenerationResult:
    """
    Generate test code for a function using LLM + self-correction loop.

    Parameters
    ----------
    chunk : FunctionChunk
        The function to generate tests for.
    context_chunks : list[RetrievalResult] | None
        Related code retrieved from ChromaDB.
    config : TestIQConfig | None
        Configuration. Loaded from file if not provided.
    llm_fn : Callable[[str], str] | None
        Optional LLM callable for testing. Defaults to Ollama.

    Returns
    -------
    GenerationResult
        Generated code, validity status, attempt count, and any issues.
    """
    cfg = config or load_config()
    call_llm = llm_fn or _default_llm_fn(cfg)
    language = chunk.language or cfg.parser.language
    test_framework = cfg.get_test_framework(language)
    max_retries = cfg.generation.max_retries

    context_str = _build_context_str(context_chunks or [])
    imports_str = "\n".join(chunk.imports) if chunk.imports else "(none)"

    # ── Initial generation ───────────────────────────────────────────────
    prompt = TEST_GENERATION_PROMPT.format(
        language=language,
        test_framework=test_framework,
        function_name=chunk.name,
        function_body=chunk.body,
        context=context_str,
        imports=imports_str,
    )

    result = GenerationResult()

    raw_code = call_llm(prompt)
    code = _strip_markdown_fences(raw_code)
    result.attempts = 1

    validation = validate_test_code(code, language, test_framework)

    if validation.is_valid:
        result.code = code
        result.is_valid = True
        return result

    # ── Self-correction loop ─────────────────────────────────────────────
    for retry in range(max_retries):
        logger.info(
            "Validation failed (attempt %d), retrying... Issues: %s",
            result.attempts,
            validation.issues,
        )

        correction_prompt = SELF_CORRECTION_PROMPT.format(
            language=language,
            test_framework=test_framework,
            function_name=chunk.name,
            original_tests=code,
            issues="\n".join(f"- {issue}" for issue in validation.issues),
        )

        raw_code = call_llm(correction_prompt)
        code = _strip_markdown_fences(raw_code)
        result.attempts += 1

        validation = validate_test_code(code, language, test_framework)

        if validation.is_valid:
            result.code = code
            result.is_valid = True
            return result

    # Max retries exhausted — return last attempt
    result.code = code
    result.is_valid = False
    result.issues = validation.issues
    return result
