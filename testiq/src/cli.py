"""TestIQ CLI — placeholder for Phase 2+."""

import typer

app = typer.Typer(
    name="testiq",
    help="Local AI-powered test generation and explanation tool.",
    add_completion=False,
)


@app.command()
def version():
    """Print TestIQ version."""
    typer.echo("testiq v0.1.0")


if __name__ == "__main__":
    app()
