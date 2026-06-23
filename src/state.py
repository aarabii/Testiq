"""
State manager for TestIQ.

Tracks successfully indexed directories in a JSON file under the config folder.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import load_config

def _get_state_file_path() -> Path:
    """Get the path to the state JSON file, ensuring its parent directory exists."""
    cfg = load_config()
    # Resolve the db path's parent, usually `.testiq`
    db_parent = Path(cfg.rag.db_path).parent.resolve()
    db_parent.mkdir(parents=True, exist_ok=True)
    return db_parent / "indexed_directories.json"

def load_state() -> dict[str, dict]:
    """Load the state of indexed directories."""
    path = _get_state_file_path()
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("indexed_directories", {})
    except Exception:
        return {}

def save_state(directories: dict[str, dict]) -> None:
    """Save the state of indexed directories."""
    path = _get_state_file_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"indexed_directories": directories}, f, indent=2)
    except Exception:
        pass

def register_directory(cleaned_name: str, original_path: str, file_count: int, chunk_count: int) -> None:
    """Register or update an indexed directory in the state."""
    dirs = load_state()
    dirs[cleaned_name] = {
        "path": str(Path(original_path).resolve()),
        "last_indexed": datetime.now().isoformat(),
        "file_count": file_count,
        "chunk_count": chunk_count,
    }
    save_state(dirs)

def get_directory_path(cleaned_name: str) -> str | None:
    """Get the original path of a cleaned directory name if it is indexed."""
    dirs = load_state()
    info = dirs.get(cleaned_name)
    if info:
        return info.get("path")
    return None

def is_directory_indexed(cleaned_name: str) -> bool:
    """Check if a directory name (cleaned with underscores) is indexed."""
    dirs = load_state()
    return cleaned_name in dirs

def get_all_directories() -> dict[str, dict]:
    """Return all indexed directories."""
    return load_state()
