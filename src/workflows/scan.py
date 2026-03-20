"""
Scan workflow — coverage gap scanner.

Parses all supported source files in a directory, identifies public
functions, cross-references with existing test files to find untested
functions, and scores them by call frequency (risk).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import TestIQConfig, load_config
from src.parser.base_parser import FunctionChunk
from src.parser.language_registry import EXTENSION_MAP, registry

logger = logging.getLogger(__name__)


@dataclass
class UntestedFunction:
    """A function that has no corresponding test."""

    name: str
    filepath: str
    line_start: int
    language: str
    risk_score: int = 0


@dataclass
class ScanResult:
    """Outcome of a coverage gap scan."""

    untested: list[UntestedFunction] = field(default_factory=list)
    total_functions: int = 0
    tested_count: int = 0

    @property
    def untested_count(self) -> int:
        return len(self.untested)

    @property
    def coverage_pct(self) -> float:
        if self.total_functions == 0:
            return 100.0
        return (self.tested_count / self.total_functions) * 100.0


# ── Test file detection patterns ─────────────────────────────────────────────

_TEST_FILE_PATTERNS = [
    re.compile(r"^test_.*\.py$", re.IGNORECASE),
    re.compile(r"^.*_test\.py$", re.IGNORECASE),
    re.compile(r"^test_.*\.js$", re.IGNORECASE),
    re.compile(r"^.*\.test\.js$", re.IGNORECASE),
    re.compile(r"^.*\.spec\.js$", re.IGNORECASE),
    re.compile(r"^.*Test\.java$"),
    re.compile(r"^.*_test\.go$"),
    re.compile(r"^.*_test\.rs$"),
]


def _is_test_file(filename: str) -> bool:
    """Check if a filename matches common test file naming conventions."""
    return any(pat.match(filename) for pat in _TEST_FILE_PATTERNS)


def _extract_tested_names(test_file: Path) -> set[str]:
    """
    Extract function names referenced in a test file.

    Uses a simple heuristic: any identifier that appears in the test file
    and matches a known function name is considered "tested".
    """
    try:
        content = test_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    # Extract all word-like identifiers from the test file
    return set(re.findall(r"\b([a-zA-Z_]\w*)\b", content))


def _count_references(name: str, source_files: list[Path]) -> int:
    """Count how many times *name* appears across all source files."""
    count = 0
    for fpath in source_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Count occurrences as whole-word matches, excluding the definition itself
        count += len(re.findall(rf"\b{re.escape(name)}\b", content))
    # Subtract 1 for the definition itself (def name / func name etc.)
    return max(count - 1, 0)


def scan_coverage(
    directory: str,
    config: TestIQConfig | None = None,
) -> ScanResult:
    """
    Scan a directory for untested functions and score them by risk.

    Parameters
    ----------
    directory : str
        Root directory to scan.
    config : TestIQConfig | None
        Configuration. Loaded from file if not provided.

    Returns
    -------
    ScanResult
        Untested functions sorted by risk score (descending), plus counts.
    """
    cfg = config or load_config()
    dir_path = Path(directory).resolve()

    if not dir_path.is_dir():
        logger.error("Not a directory: %s", directory)
        return ScanResult()

    supported_exts = set(EXTENSION_MAP.keys())

    # ── Collect all source files and test files ──────────────────────────
    source_files: list[Path] = []
    test_files: list[Path] = []

    for fpath in sorted(dir_path.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in supported_exts:
            continue
        # Skip hidden directories
        try:
            rel = fpath.relative_to(dir_path)
            if any(part.startswith(".") for part in rel.parts):
                continue
        except ValueError:
            continue

        if _is_test_file(fpath.name):
            test_files.append(fpath)
        else:
            source_files.append(fpath)

    # ── Parse all source files → extract functions ───────────────────────
    all_chunks: list[FunctionChunk] = []
    for fpath in source_files:
        try:
            parser = registry.get_parser_for_file(str(fpath))
            chunks = parser.parse_file(str(fpath))
            all_chunks.extend(chunks)
        except (ValueError, OSError) as exc:
            logger.warning("Skipping %s: %s", fpath, exc)

    # ── Extract names referenced in test files ───────────────────────────
    tested_names: set[str] = set()
    for tfile in test_files:
        tested_names |= _extract_tested_names(tfile)

    # ── Cross-reference ──────────────────────────────────────────────────
    tested_chunks: list[FunctionChunk] = []
    untested_chunks: list[FunctionChunk] = []

    for chunk in all_chunks:
        if chunk.name in tested_names:
            tested_chunks.append(chunk)
        else:
            untested_chunks.append(chunk)

    # ── Score by call frequency ──────────────────────────────────────────
    untested: list[UntestedFunction] = []
    for chunk in untested_chunks:
        risk = _count_references(chunk.name, source_files)
        untested.append(
            UntestedFunction(
                name=chunk.name,
                filepath=chunk.filepath,
                line_start=chunk.line_start,
                language=chunk.language,
                risk_score=risk,
            )
        )

    # Sort by risk descending, then alphabetically
    untested.sort(key=lambda u: (-u.risk_score, u.name))

    # Filter by risk threshold from config
    if cfg.scan.risk_threshold > 0:
        untested = [u for u in untested if u.risk_score >= cfg.scan.risk_threshold]

    result = ScanResult(
        untested=untested,
        total_functions=len(all_chunks),
        tested_count=len(tested_chunks),
    )

    logger.info(
        "Scan complete: %d/%d functions tested (%.1f%%), %d untested above threshold",
        result.tested_count,
        result.total_functions,
        result.coverage_pct,
        result.untested_count,
    )

    return result
