"""
Abstract base parser interface.

Every language-specific parser implements BaseParser. The RAG pipeline and
workflows only interact through this interface and the FunctionChunk dataclass,
making the rest of the system fully language-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FunctionChunk:
    """Language-agnostic representation of a single function or method."""

    name: str
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    body: str = ""
    docstring: str | None = None
    imports: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    language: str = ""
    filepath: str = ""


class BaseParser(ABC):
    """Abstract interface that every language parser must implement."""

    @abstractmethod
    def parse_file(self, filepath: str) -> list[FunctionChunk]:
        """
        Parse a source file and return a list of extracted function chunks.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the source file.

        Returns
        -------
        list[FunctionChunk]
            One entry per function/method found in the file.
        """

    @abstractmethod
    def extract_imports(self, filepath: str) -> list[str]:
        """
        Extract all import statements from a source file.

        Parameters
        ----------
        filepath : str
            Path to the source file.

        Returns
        -------
        list[str]
            Raw import statement strings.
        """

    @abstractmethod
    def get_language(self) -> str:
        """Return the language identifier (e.g. ``'python'``, ``'java'``)."""
