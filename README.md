# TestIQ

Local AI-powered CLI tool that auto-generates unit tests and explains failing tests. Uses Ollama for LLMs, Tree-sitter for code parsing, and ChromaDB for RAG — everything runs 100% locally. No cloud. No paid APIs.

## Requirements

- **Python** 3.10+
- **Ollama** running locally ([install guide](https://ollama.ai))
- ~4 GB RAM minimum (8 GB recommended)

## Setup

```bash
# Clone the repo
git clone https://github.com/your-org/testiq.git
cd testiq

# Copy and edit config
cp testiq.config.example.toml testiq.config.toml

# Pull required Ollama models
ollama pull deepseek-coder:1.3b
ollama pull nomic-embed-text

# Install TestIQ
pip install -e ".[dev]"
```

## CLI Commands

### Index a codebase

Parse and embed all functions into ChromaDB for context retrieval.

```bash
testiq index ./src
```

### Generate tests

Generate unit tests for a file or specific function.

```bash
# Generate tests for all functions in a file
testiq generate src/auth.py

# Generate tests for a specific function
testiq generate src/auth.py --function validate_token

# Preview without writing files
testiq generate src/auth.py --dry-run
```

### Explain a failing test

Run a failing test and get a plain-English explanation + fix.

```bash
testiq explain tests/test_auth.py
```

Output includes:
1. **Error Summary** — what went wrong
2. **Bug Location** — where the bug is
3. **Suggested Fix** — how to fix it

### Scan for untested functions

Find functions without tests, ranked by risk (call frequency).

```bash
# Table output (default)
testiq scan ./src

# JSON output
testiq scan ./src --output json
```

### Version

```bash
testiq version
```

## Configuration

All settings live in `testiq.config.toml`. Key fields:

| Section | Key | Default | Description |
|---|---|---|---|
| `[llm]` | `model` | `deepseek-coder:1.3b` | Ollama model for generation |
| `[llm]` | `temperature` | `0.2` | LLM temperature |
| `[llm]` | `max_tokens` | `2048` | Max tokens per response |
| `[embeddings]` | `model` | `nomic-embed-text` | Embedding model |
| `[rag]` | `top_k` | `5` | Number of context chunks to retrieve |
| `[rag]` | `db_path` | `.testiq/db` | ChromaDB storage path |
| `[generation]` | `max_retries` | `2` | Self-correction retry limit |
| `[generation]` | `output_dir` | `tests` | Where generated tests are written |
| `[generation]` | `dry_run` | `false` | Preview mode (no file writes) |
| `[parser]` | `language` | `python` | Default source language |
| `[parser]` | `skip_dunder_methods` | `true` | Skip `__init__`, `__str__`, etc. |
| `[parser]` | `min_function_lines` | `3` | Ignore tiny functions |
| `[scan]` | `risk_threshold` | `2` | Minimum risk score to report |
| `[scan]` | `output_format` | `table` | Default output format |
| `[logging]` | `level` | `INFO` | Log verbosity |
| `[logging]` | `show_spinner` | `true` | Show spinners during LLM calls |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## License

MIT
