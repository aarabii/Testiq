# 🧠 TestIQ

> **Local AI-Powered Unit Test Assistant** — Runs 100% locally on your machine via Ollama. No cloud dependencies, no paid API keys, and zero data leakage.

[![GitHub Repo](https://img.shields.io/badge/GitHub-aarabii%2Ftestiq-blue?logo=github)](https://github.com/aarabii/testiq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚡ How it Works (Visual Flow)

```
[ Codebase Directory ]
       │
       ▼  (testiq index <path>)
[ AST Parsing (Tree-sitter) ] ──► [ Local Vector DB (ChromaDB) ]
                                             │
                                             ▼  (testiq generate <path>)
                                  [ Retriever / Context Search ]
                                             │
                                             ▼
                                  [ Ollama (gemma4:e2b) ]
                                             │
                                             ▼
                                  [ Test Validator (Pytest) ] ──► [ Self-Correction Loop ]
                                             │
                                             ▼
                                  [ GENERATED_TEST_CASES/ ]
```

---

## 🚀 Setup Guide

### 1. Prerequisites
Make sure you have [Ollama](https://ollama.com/) running locally.

```bash
# Start Ollama service (if not running)
ollama serve

# Pull the LLM and Embedding models
ollama pull gemma4:e2b
ollama pull mxbai-embed-large
```

### 2. Installation
Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/aarabii/testiq.git
cd testiq
pip install -e .
```

### 3. Configuration
The tool reads from `testiq.config.toml` in your project root. Here is the default setup for local execution:

```toml
[llm]
provider = "ollama"
model = "gemma4:e2b"
base_url = "http://localhost:11434"

[embeddings]
model = "mxbai-embed-large"
base_url = "http://localhost:11434"
```

---

## 🛠️ Command Guide

### 1. Default Index / Quick Start
Index all files in a codebase directory. If the path is not a subcommand, TestIQ will interactively prompt you to index:
```bash
testiq D:/my_project
```

### 2. Direct Indexing
Directly index a directory without a confirmation prompt:
```bash
testiq index D:/my_project
```

### 3. Show Indexed Directories
View all directories that have been successfully indexed:
```bash
testiq show
```

### 4. Generate Unit Tests
Generate unit tests for a directory or specific file:
```bash
# Generate tests for the whole directory (outputs grouped by class)
testiq generate my_project

# Generate tests for a specific file
testiq generate my_project/math_utils.py

# Generate tests for a specific function inside a file
testiq generate my_project/math_utils.py --function add
```

### 5. Predict Test Scenarios (Assume)
Analyze code structure to predict happy path, failure scenarios, and critical failure points:
```bash
testiq assume my_project/math_utils.py
```

### 6. Explain Failing Tests
Explain a failing test in plain English and propose a code patch to fix it:
```bash
testiq explain tests/test_math_utils.py
```

### 7. Coverage Scan
Scan your codebase to locate untested functions ranked by risk:
```bash
testiq scan D:/my_project
```

### 8. Run Instructions
Get step-by-step terminal instructions to run the generated tests:
```bash
testiq run my_project
```
