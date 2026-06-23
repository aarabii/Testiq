"""
Assume workflow — predicts test scenarios and critical failure points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from langchain_ollama import ChatOllama
from rich.console import Console

from src.config import TestIQConfig, load_config
from src.parser.language_registry import registry
from src.prompts.templates import ASSUME_PROMPT

logger = logging.getLogger(__name__)
console = Console()

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

def run_assume(
    target_path: Path,
    config: TestIQConfig | None = None,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """
    Analyze code structure and predict pass/fail test scenarios.
    """
    cfg = config or load_config()
    call_llm = llm_fn or _default_llm_fn(cfg)

    # 1. Gather all files to analyze
    files_to_analyze: list[Path] = []
    if target_path.is_file():
        files_to_analyze.append(target_path)
    else:
        from src.parser.language_registry import EXTENSION_MAP
        supported_exts = set(EXTENSION_MAP.keys())
        for p in sorted(target_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in supported_exts:
                # Skip hidden folders
                if any(part.startswith(".") for part in p.relative_to(target_path).parts):
                    continue
                files_to_analyze.append(p)

    if not files_to_analyze:
        return "No codebase files found to analyze."

    # 2. Build code structure and logic summary
    summary_parts = []
    for file_path in files_to_analyze:
        try:
            parser = registry.get_parser_for_file(str(file_path))
            chunks = parser.parse_file(str(file_path))
            
            file_summary = f"File: {file_path.name}\n"
            if chunks:
                for chunk in chunks:
                    params_str = ", ".join(chunk.parameters)
                    file_summary += f"  - Function/Method: {chunk.name}({params_str})\n"
                    if chunk.docstring:
                        file_summary += f"    Docstring: {chunk.docstring}\n"
                    file_summary += f"    Implementation:\n    ```\n    {chunk.body}\n    ```\n"
            else:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                file_summary += f"  Code:\n  ```\n  {content}\n  ```\n"
                
            summary_parts.append(file_summary)
        except Exception as exc:
            summary_parts.append(f"File: {file_path.name} (failed to parse: {exc})")

    code_structure = "\n\n".join(summary_parts)

    # 3. Prompt LLM
    prompt = ASSUME_PROMPT.format(
        file_or_dir=target_path.name,
        code_structure=code_structure
    )

    class _noop_context:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    with console.status("[bold green]Analyzing scenarios…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
        response = call_llm(prompt)

    return response.strip()
