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


def _show_custom_help(cfg: TestIQConfig) -> None:
    console.print("\n[bold cyan]TestIQ — Local AI-Powered Unit Test Assistant[/]")
    console.print("Runs 100% locally via Ollama — no cloud, no paid APIs.\n")
    
    # Configurations
    console.print("[bold underline]Current Configuration[/]")
    console.print(f"  • [bold]LLM Provider:[/]    {cfg.llm.provider}")
    console.print(f"  • [bold]LLM Model:[/]       {cfg.llm.model}")
    console.print(f"  • [bold]Embed Model:[/]     {cfg.embeddings.model}")
    console.print(f"  • [bold]Ollama URL:[/]      {cfg.llm.base_url}")
    console.print(f"  • [bold]Database:[/]        {cfg.rag.db_path}")
    console.print(f"  • [bold]Default Lang:[/]    {cfg.parser.language}")
    console.print(f"  • [bold]Test Framework:[/]  {cfg.get_test_framework()}\n")
    
    # Available Commands
    table = Table(title="Available Commands", show_header=True, header_style="bold green")
    table.add_column("Command", style="yellow", width=45)
    table.add_column("Description")
    
    table.add_row(
        "testiq [dim]<DIR_PATH>[/]",
        "Prompt to index a codebase directory."
    )
    table.add_row(
        "testiq index [dim]<DIR_PATH>[/]",
        "Directly index a directory into ChromaDB (no confirmation prompt)."
    )
    table.add_row(
        "testiq show",
        "Show all successfully indexed directories."
    )
    table.add_row(
        "testiq generate [dim]<dir_name>[/] | [dim]<dir_name>/<file_name>[/]",
        "Generate RAG-based unit tests saved in GENERATED_TEST_CASES/."
    )
    table.add_row(
        "testiq run [dim]<dir_name>[/]",
        "Show LLM-generated instructions on how to run tests in the directory."
    )
    table.add_row(
        "testiq assume [dim]<dir_name>[/] | [dim]<dir_name>/<file_name>[/]",
        "Analyze code structure to predict test scenarios and failure points."
    )
    table.add_row(
        "testiq explain [dim]<test_file>[/]",
        "Explain a failing test in plain English and suggest a fix."
    )
    table.add_row(
        "testiq scan [dim]<path>[/]",
        "Scan a codebase for untested functions ranked by risk."
    )
    table.add_row(
        "testiq version",
        "Print version."
    )
    
    console.print(table)
    console.print("\n[dim]To run a command, use: testiq <command> [args][/]\n")


def _run_indexing(dir_path: Path, cfg: TestIQConfig) -> None:
    _check_ollama(cfg)
    
    clean_name = dir_path.name.replace(" ", "_")
    output_dir = Path("GENERATED_TEST_CASES") / clean_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        f"  Database: [dim]{cfg.rag.db_path}[/]\n"
        f"  Output directory: [bold]{output_dir}[/]"
    )
    
    from src.state import register_directory
    register_directory(
        cleaned_name=clean_name,
        original_path=str(dir_path),
        file_count=result.files_processed,
        chunk_count=result.chunks_indexed
    )

    if result.errors:
        console.print(f"\n[yellow]Warnings ({len(result.errors)}):[/]")
        for err in result.errors[:5]:
            console.print(f"  • {err}")


# ── Commands ─────────────────────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Entry point for TestIQ CLI."""
    if ctx.invoked_subcommand is not None:
        return
    
    cfg = _load_config_or_exit()
    
    args = ctx.args
    if not args:
        _show_custom_help(cfg)
        return
    
    dir_path = args[0]
    path_obj = Path(dir_path).resolve()
    if not path_obj.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {dir_path}")
        raise typer.Exit(code=1)
    
    confirm = typer.confirm(
        f"Do you want to index all the files in '{dir_path}' for generating test cases?",
        default=True
    )
    if confirm:
        _run_indexing(path_obj, cfg)


@app.command()
def show():
    """Show all successfully indexed directories."""
    from datetime import datetime
    from src.state import get_all_directories
    
    dirs = get_all_directories()
    if not dirs:
        console.print("[yellow]No directories have been indexed yet.[/]")
        console.print("[dim]Hint: Run `testiq <DIR_PATH>` or `testiq index <DIR_PATH>` to index a codebase.[/]")
        return
    
    table = Table(title="Indexed Codebase Directories", show_header=True, header_style="bold green")
    table.add_column("Cleaned Name", style="yellow")
    table.add_column("Original Path", style="dim")
    table.add_column("Last Indexed", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    
    for name, info in sorted(dirs.items()):
        dt_str = info.get("last_indexed", "")
        try:
            dt = datetime.fromisoformat(dt_str)
            dt_formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            dt_formatted = dt_str
            
        table.add_row(
            name,
            info.get("path", ""),
            dt_formatted,
            str(info.get("file_count", 0)),
            str(info.get("chunk_count", 0))
        )
        
    console.print(table)


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
    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        console.print(f"[bold red]Error:[/] Not a directory: {path}")
        raise typer.Exit(code=1)
    
    _run_indexing(dir_path, cfg)


def _generate_tests_for_file(
    file_path: Path,
    output_dir: Path,
    cfg: TestIQConfig,
    function: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Helper to generate tests for a single file and write them to output_dir."""
    try:
        parser = registry.get_parser_for_file(str(file_path))
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] Unsupported file extension: {file_path.name}")
        return 0

    chunks = parser.parse_file(str(file_path))
    if not chunks:
        console.print(f"[yellow]No functions found in {file_path.name}.[/]")
        return 0

    if function:
        chunks = [c for c in chunks if c.name == function]
        if not chunks:
            console.print(
                f"[bold red]Error:[/] Function '{function}' not found in {file_path.name}.\n"
                f"  Available: {', '.join(c.name for c in parser.parse_file(str(file_path)))}"
            )
            return 0

    try:
        file_imports = parser.extract_imports(str(file_path))
        for chunk in chunks:
            chunk.imports = file_imports
    except Exception:
        pass

    try:
        retriever = Retriever(cfg)
    except Exception:
        retriever = None

    effective_dry_run = dry_run or cfg.generation.dry_run
    output_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    for chunk in chunks:
        context_chunks = []
        if retriever:
            try:
                context_chunks = retriever.query_chunk(chunk)
            except Exception:
                pass

        console.print(f"  • Generating tests for [cyan]{chunk.name}()[/]…")

        try:
            with console.status("[bold green]Thinking…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
                result = generate_tests(chunk, context_chunks, cfg)
        except Exception as exc:
            console.print(f"    [red]Failed: {exc}[/]")
            continue

        if not result.code:
            console.print(f"    [yellow]No code generated after {result.attempts} attempt(s).[/]")
            continue

        if effective_dry_run:
            console.print(f"\n[dim]── dry-run output for {chunk.name}() ──[/]\n")
            console.print(result.code)
            console.print(f"\n[dim]── end ({result.attempts} attempt(s)) ──[/]")
        else:
            lang = chunk.language or cfg.parser.language
            ext = ".py"
            if lang in ("javascript", "typescript"):
                ext = ".test.js" if lang == "javascript" else ".test.ts"
            elif lang == "java":
                ext = "Test.java"
            elif lang == "go":
                ext = "_test.go"
            elif lang == "rust":
                ext = "_test.rs"
                
            test_filename = f"test_{file_path.stem}{ext}"
            if lang == "java":
                test_filename = f"{file_path.stem.capitalize()}Test.java"
            elif lang == "go":
                test_filename = f"{file_path.stem}_test.go"
            elif lang == "rust":
                test_filename = f"{file_path.stem}_test.rs"

            test_path = output_dir / test_filename
            mode = "a" if test_path.exists() and total_generated > 0 else "w"
            with open(test_path, mode, encoding="utf-8") as f:
                if mode == "a":
                    f.write("\n\n")
                f.write(result.code)

            console.print(
                f"    [green]✓[/] Written to [bold]{test_path}[/] "
                f"({result.attempts} attempt(s))"
            )

        if not result.is_valid:
            console.print(f"    [yellow]⚠ Validation issues: {', '.join(result.issues)}[/]")

        total_generated += 1

    return total_generated


@app.command()
def generate(
    target: str = typer.Argument(..., help="Directory name, directory/file_name, or direct filepath to generate tests for."),
    function: Optional[str] = typer.Option(
        None, "--function", "-f", help="Generate tests for a specific function only (only applicable if target is a file)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print generated tests to stdout instead of writing a file."
    ),
):
    """Generate unit tests for an indexed directory/file, or a direct filepath."""
    cfg = _load_config_or_exit()
    _check_ollama(cfg)

    # 1. Backwards compatibility: check if target is an existing path on disk
    target_path = Path(target).resolve()
    if target_path.exists():
        if target_path.is_file():
            console.print(f"\n[bold]Generating tests for file directly:[/] [cyan]{target_path.name}[/]")
            output_dir = Path(cfg.generation.output_dir)
            count = _generate_tests_for_file(target_path, output_dir, cfg, function, dry_run)
            console.print(f"\n[bold green]✓ Done![/] Generated tests for {count} function(s).")
            return
        elif target_path.is_dir():
            console.print(f"\n[bold]Generating tests for directory directly:[/] [cyan]{target_path.name}[/]")
            from src.parser.language_registry import EXTENSION_MAP
            supported_exts = set(EXTENSION_MAP.keys())
            supported_files = []
            for p in sorted(target_path.rglob("*")):
                if p.is_file() and p.suffix.lower() in supported_exts:
                    if any(part.startswith(".") for part in p.relative_to(target_path).parts):
                        continue
                    supported_files.append(p)
            if not supported_files:
                console.print(f"[yellow]No supported source files found in '{target_path}'.[/]")
                raise typer.Exit(code=0)
            
            output_dir = Path(cfg.generation.output_dir)
            total_funcs = 0
            for file_path in supported_files:
                count = _generate_tests_for_file(file_path, output_dir, cfg, None, dry_run)
                total_funcs += count
            console.print(f"\n[bold green]✓ Done![/] Generated tests for {total_funcs} function(s) across {len(supported_files)} file(s).")
            return

    # 2. Otherwise, treat as RAG indexed dir: <dir_name> or <dir_name>/<file_name>
    normalized_target = target.replace("\\", "/")
    parts = normalized_target.split("/", 1)
    
    dir_name = parts[0]
    sub_path = parts[1] if len(parts) > 1 else None

    # Check if the directory name is indexed
    from src.state import get_directory_path, is_directory_indexed
    if not is_directory_indexed(dir_name):
        console.print(f"[bold red]Error:[/] Directory '{dir_name}' has not been indexed yet.")
        confirm = typer.confirm("Would you like to index a directory now?", default=True)
        if not confirm:
            raise typer.Exit(code=1)
        
        path_str = typer.prompt("Please enter the absolute path to the directory")
        path_obj = Path(path_str).resolve()
        if not path_obj.is_dir():
            console.print(f"[bold red]Error:[/] Not a directory: {path_str}")
            raise typer.Exit(code=1)
        
        # Verify the dir name matches the prompt's target directory
        prompt_dir_name = path_obj.name.replace(" ", "_")
        if prompt_dir_name != dir_name:
            console.print(f"[yellow]Warning:[/] The directory name '{prompt_dir_name}' does not match '{dir_name}'. We will index it anyway.")
            dir_name = prompt_dir_name
            
        _run_indexing(path_obj, cfg)
    
    # Get original directory path from state
    original_path_str = get_directory_path(dir_name)
    if not original_path_str:
        console.print(f"[bold red]Error:[/] Could not find original path for '{dir_name}' in state.")
        raise typer.Exit(code=1)
        
    original_dir = Path(original_path_str)
    output_dir = Path("GENERATED_TEST_CASES") / dir_name

    total_files_processed = 0
    total_funcs_generated = 0

    if sub_path:
        # Generate for specific file
        file_path = (original_dir / sub_path).resolve()
        if not file_path.is_file():
            console.print(f"[bold red]Error:[/] File not found: {file_path}")
            raise typer.Exit(code=1)
            
        console.print(f"\n[bold]Generating tests for file:[/] [cyan]{sub_path}[/]")
        count = _generate_tests_for_file(file_path, output_dir, cfg, function, dry_run)
        if count > 0:
            total_files_processed += 1
            total_funcs_generated += count
    else:
        # Generate for entire directory
        from src.parser.language_registry import EXTENSION_MAP
        supported_exts = set(EXTENSION_MAP.keys())
        
        supported_files = []
        for p in sorted(original_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in supported_exts:
                # Skip hidden directories
                if any(part.startswith(".") for part in p.relative_to(original_dir).parts):
                    continue
                supported_files.append(p)
                
        if not supported_files:
            console.print(f"[yellow]No supported source files found in '{original_dir}'.[/]")
            raise typer.Exit(code=0)
            
        console.print(f"\n[bold]Generating tests for all {len(supported_files)} files in directory:[/] [cyan]{dir_name}[/]")
        for file_path in supported_files:
            rel_file = file_path.relative_to(original_dir)
            console.print(f"\n[bold underline]File: {rel_file}[/]")
            count = _generate_tests_for_file(file_path, output_dir, cfg, None, dry_run)
            if count > 0:
                total_files_processed += 1
                total_funcs_generated += count

    console.print(
        f"\n[bold green]✓ Done![/] Generated tests for [bold]{total_funcs_generated}[/] function(s) "
        f"across [bold]{total_files_processed}[/] file(s)."
    )


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


@app.command()
def run(
    dir_name: str = typer.Argument(..., help="Cleaned directory name under GENERATED_TEST_CASES/ to run."),
):
    """Show LLM-generated instructions on how to run tests in the directory."""
    cfg = _load_config_or_exit()
    
    output_dir = Path("GENERATED_TEST_CASES") / dir_name
    if not output_dir.is_dir():
        console.print(f"[bold red]Error:[/] Directory '{output_dir}' does not exist.")
        console.print("[dim]Hint: Run `testiq generate <dir_name>` first.[/]")
        raise typer.Exit(code=1)
        
    test_files = [str(p.name) for p in sorted(output_dir.iterdir()) if p.is_file()]
    if not test_files:
        console.print(f"[yellow]No files found in '{output_dir}'.[/]")
        raise typer.Exit(code=0)
        
    first_file = output_dir / test_files[0]
    ext = first_file.suffix.lower()
    
    lang = "python"
    framework = "pytest"
    
    if ext == ".py":
        lang = "python"
        framework = "pytest"
    elif ext in (".js", ".jsx"):
        lang = "javascript"
        framework = "jest"
    elif ext in (".ts", ".tsx"):
        lang = "typescript"
        framework = "jest"
    elif ext == ".java":
        lang = "java"
        framework = "junit"
    elif ext == ".go":
        lang = "go"
        framework = "gotest"
    elif ext == ".rs":
        lang = "rust"
        framework = "cargo test"
        
    from src.prompts.templates import RUN_INSTRUCTION_PROMPT
    prompt = RUN_INSTRUCTION_PROMPT.format(
        language=lang,
        test_framework=framework,
        test_files="\n".join(f"- {f}" for f in test_files)
    )
    
    _check_ollama(cfg)
    
    from src.workflows.explain import _default_llm_fn
    call_llm = _default_llm_fn(cfg)
    
    console.print(f"\n[bold]Consulting local LLM for test run guidelines on {dir_name}…[/]")
    
    with console.status("[bold green]Thinking…[/]", spinner="dots") if cfg.logging.show_spinner else _noop_context():
        response = call_llm(prompt)
        
    console.print("\n[bold underline]Test Run Guidelines[/]")
    from rich.markdown import Markdown
    console.print(Markdown(response))


@app.command()
def assume(
    target: str = typer.Argument(..., help="Cleaned directory name (e.g. 'my_app') or directory/file_name (e.g. 'my_app/cli.py') to predict scenarios."),
):
    """Analyze codebase structure to predict pass/fail test scenarios."""
    cfg = _load_config_or_exit()
    
    normalized_target = target.replace("\\", "/")
    parts = normalized_target.split("/", 1)
    
    dir_name = parts[0]
    sub_path = parts[1] if len(parts) > 1 else None
    
    target_path = Path(target).resolve()
    if target_path.exists():
        resolved_path = target_path
    else:
        from src.state import get_directory_path, is_directory_indexed
        if not is_directory_indexed(dir_name):
            console.print(f"[bold red]Error:[/] Directory '{dir_name}' has not been indexed yet.")
            raise typer.Exit(code=1)
            
        original_path_str = get_directory_path(dir_name)
        if not original_path_str:
            console.print(f"[bold red]Error:[/] Could not find original path for '{dir_name}' in state.")
            raise typer.Exit(code=1)
            
        original_dir = Path(original_path_str)
        if sub_path:
            resolved_path = (original_dir / sub_path).resolve()
            if not resolved_path.exists():
                console.print(f"[bold red]Error:[/] File not found: {resolved_path}")
                raise typer.Exit(code=1)
        else:
            resolved_path = original_dir
            
    _check_ollama(cfg)
    
    from src.workflows.assume import run_assume
    console.print(f"\n[bold]Running scenario predictor on {target}…[/]")
    
    response = run_assume(resolved_path, cfg)
    
    console.print("\n[bold underline]Scenario Prediction Report[/]")
    from rich.markdown import Markdown
    console.print(Markdown(response))


# ── Context manager for no-spinner mode ──────────────────────────────────────

class _noop_context:
    """Dummy context manager when spinner is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    app()
