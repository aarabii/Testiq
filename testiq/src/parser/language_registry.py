"""
Language registry — maps language names and file extensions to parser classes.

Usage::

    from src.parser.language_registry import registry

    parser = registry.get_parser("python")
    parser = registry.get_parser_for_file("app.py")
"""

from __future__ import annotations

from typing import Type

from src.parser.base_parser import BaseParser


# File extension → language name
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}


class LanguageRegistry:
    """Central registry that resolves language names to parser instances."""

    def __init__(self) -> None:
        self._parsers: dict[str, Type[BaseParser]] = {}

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, language: str, parser_class: Type[BaseParser]) -> None:
        """Register a parser class for the given language name."""
        self._parsers[language.lower()] = parser_class

    # ── Lookup ───────────────────────────────────────────────────────────

    def get_parser(self, language: str) -> BaseParser:
        """
        Instantiate and return a parser for *language*.

        Raises
        ------
        ValueError
            If no parser is registered for the requested language.
        """
        key = language.lower()
        if key not in self._parsers:
            available = ", ".join(sorted(self._parsers)) or "(none)"
            raise ValueError(
                f"No parser registered for language '{language}'. "
                f"Available: {available}"
            )
        return self._parsers[key]()

    def get_parser_for_file(self, filepath: str) -> BaseParser:
        """
        Determine the language from the file extension, then return a parser.

        Raises
        ------
        ValueError
            If the extension is not recognised or no parser is registered.
        """
        from pathlib import Path

        ext = Path(filepath).suffix.lower()
        if ext not in EXTENSION_MAP:
            supported = ", ".join(sorted(EXTENSION_MAP)) or "(none)"
            raise ValueError(
                f"Unrecognised file extension '{ext}'. "
                f"Supported: {supported}"
            )
        return self.get_parser(EXTENSION_MAP[ext])

    @property
    def registered_languages(self) -> list[str]:
        """Return a sorted list of registered language names."""
        return sorted(self._parsers)


# ── Singleton registry ───────────────────────────────────────────────────────

registry = LanguageRegistry()


def _auto_register() -> None:
    """Import built-in parsers so they self-register."""
    try:
        from src.parser.languages import python_parser  # noqa: F401
    except ImportError:
        pass  # grammar not installed — skip silently


_auto_register()
