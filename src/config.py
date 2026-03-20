"""
TestIQ configuration loader.

Reads testiq.config.toml and exposes a validated Pydantic model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# tomllib is built-in from Python 3.11+; fall back to tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError(
            "Install 'tomli' for Python <3.11: pip install tomli"
        ) from exc


# ── Config Section Models ────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "deepseek-coder:1.3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 2048


class EmbeddingsConfig(BaseModel):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


class RAGConfig(BaseModel):
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    db_path: str = ".testiq/db"


class GenerationConfig(BaseModel):
    max_retries: int = 2
    output_dir: str = "tests"
    include_edge_cases: bool = True
    include_docstrings: bool = True
    dry_run: bool = False
    test_framework: Optional[str] = None


class ParserConfig(BaseModel):
    language: str = "python"
    skip_dunder_methods: bool = True
    skip_private_methods: bool = False
    min_function_lines: int = 3


class ScanConfig(BaseModel):
    risk_threshold: int = 2
    output_format: str = "table"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_file: str = ".testiq/testiq.log"
    show_spinner: bool = True


class ProjectConfig(BaseModel):
    name: str = "TestIQ"
    version: str = "0.1.0"


# ── Root Config ──────────────────────────────────────────────────────────────

class TestIQConfig(BaseModel):
    """Top-level configuration for TestIQ."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    languages: dict[str, str] = Field(default_factory=lambda: {
        "python": "pytest",
        "javascript": "jest",
        "typescript": "jest",
        "java": "junit",
        "go": "gotest",
        "rust": "cargo",
    })
    scan: ScanConfig = Field(default_factory=ScanConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)

    def get_test_framework(self, language: str | None = None) -> str:
        """Return the test framework for the given (or configured) language."""
        lang = language or self.parser.language
        # Explicit override in [generation] takes priority
        if self.generation.test_framework:
            return self.generation.test_framework
        return self.languages.get(lang, "pytest")


# ── Loader ───────────────────────────────────────────────────────────────────

_CONFIG_FILENAMES = ("testiq.config.toml",)


def _find_config_file(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a config file."""
    current = (start or Path.cwd()).resolve()
    for _ in range(50):  # safety cap
        for name in _CONFIG_FILENAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_config(path: str | Path | None = None) -> TestIQConfig:
    """
    Load and validate the TestIQ configuration.

    Parameters
    ----------
    path : str | Path | None
        Explicit path to a TOML config file. When *None*, the loader
        searches the current working directory and parent directories.

    Returns
    -------
    TestIQConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If no config file can be located.
    """
    if path is not None:
        config_path = Path(path)
    else:
        config_path = _find_config_file()

    if config_path is None or not config_path.is_file():
        # Return defaults when no file is found
        return TestIQConfig()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    return TestIQConfig(**raw)
