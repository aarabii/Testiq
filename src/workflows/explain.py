"""
Explain workflow — 3-step sequential chain for bug explanation.

Accepts a failing test's source code and traceback, then runs three
LLM calls in sequence to produce a plain-English explanation and fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from langchain_ollama import ChatOllama

from src.config import TestIQConfig, load_config
from src.prompts.templates import (
    EXPLAIN_STEP1_PROMPT,
    EXPLAIN_STEP2_PROMPT,
    EXPLAIN_STEP3_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Outcome of the 3-step bug explanation chain."""

    error_summary: str = ""
    bug_location: str = ""
    suggested_fix: str = ""


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


def explain_failure(
    test_code: str,
    traceback: str,
    config: TestIQConfig | None = None,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> ExplanationResult:
    """
    Explain a failing test in plain English using a 3-step LLM chain.

    Parameters
    ----------
    test_code : str
        The source code of the failing test.
    traceback : str
        The full traceback / error output.
    config : TestIQConfig | None
        Configuration. Loaded from file if not provided.
    llm_fn : Callable[[str], str] | None
        Optional LLM callable for testing. Defaults to Ollama.

    Returns
    -------
    ExplanationResult
        Three-part explanation: summary, bug location, and suggested fix.
    """
    cfg = config or load_config()
    call_llm = llm_fn or _default_llm_fn(cfg)

    result = ExplanationResult()

    # ── Step 1: Summarise the error ──────────────────────────────────────
    prompt1 = EXPLAIN_STEP1_PROMPT.format(
        test_code=test_code,
        traceback=traceback,
    )
    result.error_summary = call_llm(prompt1).strip()
    logger.info("Step 1 (summarise) complete.")

    # ── Step 2: Locate the bug ───────────────────────────────────────────
    prompt2 = EXPLAIN_STEP2_PROMPT.format(
        test_code=test_code,
        traceback=traceback,
        error_summary=result.error_summary,
    )
    result.bug_location = call_llm(prompt2).strip()
    logger.info("Step 2 (locate) complete.")

    # ── Step 3: Suggest a fix ────────────────────────────────────────────
    prompt3 = EXPLAIN_STEP3_PROMPT.format(
        test_code=test_code,
        traceback=traceback,
        error_summary=result.error_summary,
        bug_location=result.bug_location,
    )
    result.suggested_fix = call_llm(prompt3).strip()
    logger.info("Step 3 (fix) complete.")

    return result
