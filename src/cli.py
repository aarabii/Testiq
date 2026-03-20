"""
TestIQ CLI — local AI-powered test generation and explanation tool.

Commands:
    testiq index <path>                      Index codebase into ChromaDB
    testiq generate <file> [--function NAME] Generate tests for a file/function
    testiq explain <test_file>               Explain a failing test
    testiq scan <path> [--output table|json] Find untested functions
    testiq version                           Print version
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.config import TestIQConfig, load_config
from src.parser.language_registry import registry
from src.rag.indexer import Indexer
from src.rag.retriever import Retriever
from src.workflows.generate import generate_tests
from src.workflows.explain import explain_failure
from src.workflows.scan import scan_coverage

app = typer.Typer(
    name="testiq",
    help="Local AI-powered test generation and explanation tool. "
         "Runs 100% locally via Ollama — no cloud, no paid APIs.",
    add_completion=False,
)

console = Console()

VERSION = "0.1.0"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_config_or_exit() -> TestIQConfig:
    """Load config, exiting with a friendly message if it fails."""
    try:
        cfg = load_config()
        return cfg
    except Exception as exc:
        console.print(
            "[bold red]Error:[/] Could not load testiq.config.toml\n"
            f"  {exc}\n\n"
            "[dim]Hint: Copy testiq.config.example.toml → testiq.config.toml "
            "and edit to your needs.[/]",
        )
        raise typer.Exit(code=1)


def _check_ollama(cfg: TestIQConfig) -> None:
    """Quick check that Ollama is reachable. Warn if not."""
    import urllib.request
    import urllib.error

    try:
        url = cfg.llm.base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3):
            pass
    except Exception:
        console.print(
            "[bold yellow]Warning:[/] Cannot reach Ollama at "
            f"[cyan]{cfg.llm.base_url}[/]\n"
            "  Make sure Ollama is running: [bold]ollama serve[/]\n"
            f"  Required models: [bold]{cfg.llm.model}[/], "
            f"[bold]{cfg.embeddings.model}[/]\n",
        )


# ── Commands ─────────────────────────────────────────────────────────────────


@app.command()
def version():
    """Print TestIQ version."""
    typer.echo(f"testiq v{VERSION}")


@app.command()
def index(
    path: str = typer.Argument(..., help="Directory to index."),
):
    """Index a codebase into ChromaDB for RAG retrieval."""
    cfg = _load_config_or_exit()
    _check_ollama(cfg)

    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {path}")
        raise typer.Exit(code=1)



    try:
        with console.status("[bold green]Indexing…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
            indexer = Indexer(cfg)
            result = indexer.index_directory(str(dir_path))
    except Exception as exc:
        console.print(f"[bold red]Error:[/] Indexing failed: {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold green]✓[/] Indexed [bold]{result.chunks_indexed}[/] functions "
        f"from [bold]{result.files_processed}[/] files.\n"
        f"  Database: [dim]{cfg.rag.db_path}[/]"
    )

    if result.errors:
        console.print(f"\n[yellow]Warnings ({len(result.errors)}):[/]")
        for err in result.errors[:5]:
            console.print(f"  • {err}")


@app.command()
def generate(
    file: str = typer.Argument(..., help="Source file to generate tests for."),
    function: Optional[str] = typer.Option(
        None, "--function", "-f", help="Generate tests for a specific function only."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print generated tests to stdout instead of writing a file."
    ),
):
    """Generate unit tests for a source file or specific function."""
    cfg = _load_config_or_exit()
    _check_ollama(cfg)

    file_path = Path(file).resolve()
    if not file_path.is_file():
        console.print(f"[bold red]Error:[/] File not found: {file}")
        raise typer.Exit(code=1)



    # Parse the file
    try:
        parser = registry.get_parser_for_file(str(file_path))
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1)

    chunks = parser.parse_file(str(file_path))
    if not chunks:
        console.print("[yellow]No functions found in the file.[/]")
        raise typer.Exit(code=0)

    # Filter to specific function if requested
    if function:
        chunks = [c for c in chunks if c.name == function]
        if not chunks:
            console.print(
                f"[bold red]Error:[/] Function '{function}' not found in {file_path.name}.\n"
                f"  Available: {', '.join(c.name for c in parser.parse_file(str(file_path)))}"
            )
            raise typer.Exit(code=1)

    # Enrich chunks with file imports
    try:
        file_imports = parser.extract_imports(str(file_path))
        for chunk in chunks:
            chunk.imports = file_imports
    except Exception:
        pass

    # Try to retrieve context from ChromaDB (non-fatal if DB doesn't exist)
    try:
        retriever = Retriever(cfg)
    except Exception:
        retriever = None

    # Override dry_run from CLI flag
    effective_dry_run = dry_run or cfg.generation.dry_run

    output_dir = Path(cfg.generation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    for chunk in chunks:
        # Retrieve context
        context_chunks = []
        if retriever:
            try:
                context_chunks = retriever.query_chunk(chunk)
            except Exception:
                pass

        console.print(f"\n[bold]Generating tests for[/] [cyan]{chunk.name}()[/]…")

        try:
            with console.status("[bold green]Thinking…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
                result = generate_tests(chunk, context_chunks, cfg)
        except Exception as exc:
            console.print(f"  [red]Failed: {exc}[/]")
            continue

        if not result.code:
            console.print(f"  [yellow]No code generated after {result.attempts} attempt(s).[/]")
            continue

        if effective_dry_run:
            console.print(f"\n[dim]── dry-run output for {chunk.name}() ──[/]\n")
            console.print(result.code)
            console.print(f"\n[dim]── end ({result.attempts} attempt(s)) ──[/]")
        else:
            # Write test file
            test_filename = f"test_{file_path.stem}.py"
            test_path = output_dir / test_filename

            # Append if file already exists (multiple functions)
            mode = "a" if test_path.exists() and total_generated > 0 else "w"
            with open(test_path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n")
                f.write(result.code)

            console.print(
                f"  [green]✓[/] Written to [bold]{test_path}[/] "
                f"({result.attempts} attempt(s))"
            )

        if not result.is_valid:
            console.print(f"  [yellow]⚠ Validation issues: {', '.join(result.issues)}[/]")

        total_generated += 1

    console.print(f"\n[bold green]Done![/] Generated tests for {total_generated} function(s).")


@app.command()
def explain(
    test_file: str = typer.Argument(..., help="Path to the failing test file."),
):
    """Explain a failing test in plain English and suggest a fix."""
    cfg = _load_config_or_exit()
    _check_ollama(cfg)

    test_path = Path(test_file).resolve()
    if not test_path.is_file():
        console.print(f"[bold red]Error:[/] File not found: {test_file}")
        raise typer.Exit(code=1)

    test_code = test_path.read_text(encoding="utf-8", errors="replace")

    # Run the test to capture traceback
    console.print(f"[bold]Running[/] [cyan]{test_path.name}[/] to capture errors…\n")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=long", "--no-header"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(test_path.parent),
        )
        traceback_str = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        traceback_str = "(test timed out after 60 seconds)"
    except FileNotFoundError:
        traceback_str = "(could not run pytest — is it installed?)"

    if proc.returncode == 0:
        console.print("[bold green]✓[/] All tests passed! Nothing to explain.")
        raise typer.Exit(code=0)


    with console.status("[bold green]Analysing failure…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
        result = explain_failure(test_code, traceback_str, cfg)

    console.print("\n[bold underline]Error Summary[/]")
    console.print(result.error_summary)

    console.print("\n[bold underline]Bug Location[/]")
    console.print(result.bug_location)

    console.print("\n[bold underline]Suggested Fix[/]")
    console.print(result.suggested_fix)


@app.command()
def scan(
    path: str = typer.Argument(..., help="Directory to scan for untested functions."),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: 'table' or 'json'."
    ),
):
    """Scan a codebase for untested functions, ranked by risk."""
    cfg = _load_config_or_exit()

    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {path}")
        raise typer.Exit(code=1)


    with console.status("[bold green]Scanning…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
        result = scan_coverage(str(dir_path), cfg)

    if output.lower() == "json":
        data = {
            "total_functions": result.total_functions,
            "tested_count": result.tested_count,
            "untested_count": result.untested_count,
            "coverage_pct": round(result.coverage_pct, 1),
            "untested": [
                {
                    "name": u.name,
                    "filepath": u.filepath,
                    "line_start": u.line_start,
                    "language": u.language,
                    "risk_score": u.risk_score,
                }
                for u in result.untested
            ],
        }
        typer.echo(json.dumps(data, indent=2))
    else:
        # Table output
        console.print(
            f"\n[bold]Coverage:[/] {result.tested_count}/{result.total_functions} "
            f"functions tested ([cyan]{result.coverage_pct:.1f}%[/])\n"
        )

        if not result.untested:
            console.print("[bold green]✓[/] No untested functions above risk threshold!")
        else:
            table = Table(title="Untested Functions", show_lines=True)
            table.add_column("Risk", justify="center", style="bold red", width=6)
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Line", justify="right")
            table.add_column("Language")

            for u in result.untested:
                table.add_row(
                    str(u.risk_score),
                    u.name,
                    str(Path(u.filepath).name),
                    str(u.line_start),
                    u.language,
                )

            console.print(table)


# ── Context manager for no-spinner mode ──────────────────────────────────────

class _noop_context:
    """Dummy context manager when spinner is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    app()
