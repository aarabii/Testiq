# TestIQ — System Architecture

## What This Project Is
A CLI tool that uses local LLMs (via Ollama) and RAG (via LangChain + ChromaDB) to auto-generate unit tests and explain failing tests for any codebase. Everything runs 100% locally. No cloud. No paid APIs. Language support is configurable via testiq.config.toml.

--- 

## High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                            │
│              (Typer — user entry point)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   [Index Engine]  [Generate Engine]  [Explain Engine]
         │               │               │
         └───────────────┼───────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    RAG Core Layer                           │
│                                                             │
│   Code Parser → Chunker → Embedder → ChromaDB              │
│   (Tree-sitter)            (nomic-embed-text via Ollama)    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  AI Workflow Layer                          │
│                                                             │
│   LangChain Chains → Prompt Templates → Ollama LLM         │
│         ↑                                                   │
│   [Reflection Loop] [Self-Correction] [Validation]         │
└─────────────────────────────────────────────────────────────┘
```

---

## Folder Structure
```
testiq/
├── .antigravity/
│   ├── AGENTS.md
│   ├── MEMORY.md
│   └── MCP.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── WORKFLOWS.md
│   └── TECH_STACK.md
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── parser/
│   │   ├── base_parser.py        ← abstract interface all languages implement
│   │   ├── language_registry.py  ← maps language name → parser class
│   │   └── languages/
│   │       ├── python_parser.py
│   │       ├── javascript_parser.py
│   │       ├── java_parser.py
│   │       └── go_parser.py
│   ├── rag/
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── workflows/
│   │   ├── generate.py
│   │   ├── explain.py
│   │   └── scan.py
│   ├── prompts/
│   │   └── templates.py
│   └── validator/
│       └── test_validator.py
├── tests/
├── testiq.config.example.toml
├── testiq.config.toml         ← gitignored, user's own copy
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Component 1 — Code Parser (Tree-sitter)

Tree-sitter is used for all languages. It provides a consistent API regardless of which language is being parsed.
```
Input: Source file path + language (from config)
         ↓
  language_registry.py
  reads config → loads correct grammar
         ↓
  Tree-sitter parses file into AST
         ↓
  language-specific parser extracts:
    - function/method name
    - parameters + types (if typed language)
    - return type
    - function body
    - imports/dependencies
    - line numbers
    - docstrings/comments
         ↓
Output: List of FunctionChunk objects (language-agnostic)
```

Every language parser implements the same base interface:
```python
class BaseParser:
    def parse_file(self, filepath: str) -> list[FunctionChunk]
    def extract_imports(self, filepath: str) -> list[str]
    def get_language(self) -> str
```

This means the RAG pipeline, generation workflow, and everything above the parser layer never needs to know what language it's dealing with.

---

## Component 2 — RAG Pipeline

### Indexing Phase (run once per project)
```
All source files in target directory
       ↓
  language_registry detects file extension → picks correct parser
       ↓
  Parser extracts FunctionChunks
       ↓
  Each chunk enriched with metadata:
    { filename, function_name, language, line_start, line_end }
       ↓
  nomic-embed-text (via Ollama) generates embeddings
       ↓
  ChromaDB persists to .testiq/db/
```

### Retrieval Phase (at generation time)
```
Target function
       ↓
  Query embedding generated
       ↓
  Top-K similar chunks fetched from ChromaDB
  (helper functions, imports, related classes)
       ↓
  Context window assembled for LLM
```

---

## Component 3 — AI Workflow Layer

### Workflow A — Test Generation with Self-Correction Loop
```
Target Function + Retrieved Context
             ↓
     Prompt LLM → Generate Tests (Draft 1)
             ↓
     Validation Agent
     checks: correct imports? assertions present?
     follows correct test framework structure?
             ↓
        Valid? ──YES──→ Write to output_dir
          │
          NO (max 2 retries)
          ↓
     Self-Correction Prompt
     LLM fixes based on validation errors
          ↓
     Write to output_dir
```

### Workflow B — Bug Explanation Chain
```
Failing test file + traceback
             ↓
     Chain Step 1: Summarize the error in plain English
             ↓
     Chain Step 2: Locate exact source of bug in code
             ↓
     Chain Step 3: Suggest a concrete fix
             ↓
     Output: explanation + suggested patch printed to terminal
```

### Workflow C — Coverage Gap Scanner
```
Target directory
             ↓
  Parse all files → extract all public functions
             ↓
  Scan existing test files → find which functions have tests
             ↓
  Cross-reference → find untested functions
             ↓
  Score by risk (call frequency across codebase)
             ↓
  Output: prioritized table of untested functions
```

---

## Component 4 — Prompt Templates

All prompts live in src/prompts/templates.py. No prompt string should ever be hardcoded inside workflow files.

Each prompt template is language-aware — the language name and its corresponding test framework are injected dynamically from config.

Example injection:
```
language = "java"
test_framework = "junit"

→ "Generate JUnit tests for the following Java function..."
```

Supported language → framework mappings:
- python   → pytest
- javascript → jest
- typescript → jest
- java     → junit
- go       → go test
- rust     → cargo test

---

## CLI Commands
```bash
testiq index <path>                        # index codebase into ChromaDB
testiq generate <file>                     # generate tests for whole file
testiq generate <file> --function <name>   # generate for one function only
testiq explain <test_file>                 # explain failing test in plain English
testiq scan <path>                         # find untested functions
testiq generate <file> --dry-run           # preview tests, don't write files
```

---

## Data Flow — End to End Example
```
User runs:
  testiq generate src/auth.js --function validateToken

Step 1 — Config loaded
  language = "javascript", test_framework = "jest"

Step 2 — Parser
  javascript_parser.py reads auth.js via Tree-sitter
  Extracts validateToken() — params, body, imports

Step 3 — Retrieval
  Embeds function → queries ChromaDB
  Retrieves: TokenError class, JWT_SECRET import,
  decodeToken() helper

Step 4 — Prompt Construction
  System: "You are a senior JavaScript engineer writing Jest tests."
  Context: [retrieved chunks]
  Task: "Generate Jest tests for validateToken() with edge cases."

Step 5 — Ollama LLM generates test draft

Step 6 — Validation Agent checks structure

Step 7 — Output
  Writes tests/auth.test.js
  Prints: "Generated 5 test cases for validateToken()"
```

---

## Key Architecture Rules

- Language is always read from testiq.config.toml — never hardcoded
- Parser layer is fully abstracted — pipeline above it is language-agnostic
- All prompts live in prompts/templates.py — never inline
- RAG retrieval always happens before any LLM call — no exceptions
- Self-correction loop is capped at max_retries from config — never infinite
- ChromaDB always persists to disk — never in-memory for production runs
- No external API calls — Ollama is the only LLM interface