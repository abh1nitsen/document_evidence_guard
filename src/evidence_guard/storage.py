"""Resolve private runtime paths outside every Git checkout."""

from pathlib import Path


def outside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if any(
        (parent / ".git").is_file() or (parent / ".git" / "HEAD").is_file()
        for parent in (resolved, *resolved.parents)
    ):
        raise ValueError("PDF inputs and runtime data must be outside any Git checkout.")
    return resolved
