#!/usr/bin/env python3
"""
make_structure.py
Generates a clean directory tree into structure.txt.

Usage (from project root):
  python make_structure.py
  python make_structure.py --max-depth 3
  python make_structure.py --only-ext .py .ipynb .csv
  python make_structure.py --ignore node_modules .git .venv .mlops-venv __pycache__ .ipynb_checkpoints

Notes:
- Paths are shown relative to the current working directory.
- Ignores are matched against directory/file names (not full paths).
"""

import os
import argparse
from pathlib import Path

DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".env",
    ".mlops-venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".DS_Store",
}


def is_hidden(p: Path) -> bool:
    name = p.name
    return name.startswith(".") and name not in {
        ".gitignore",
        ".gitattributes",
        ".gitkeep",
    }


def should_skip(name: str, ignores: set[str]) -> bool:
    return (name in ignores) or (name == "") or (name == ".")


def tree(
    root: Path, max_depth: int, ignores: set[str], only_ext: set[str]
) -> list[str]:
    lines: list[str] = []
    root = root.resolve()
    root_prefix = len(str(root)) + 1

    def rel(p: Path) -> str:
        s = str(p)
        return s[root_prefix:] if s.startswith(str(root)) else s

    def walk(dir_path: Path, depth: int):
        if max_depth is not None and depth > max_depth:
            return

        try:
            entries = sorted(
                dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
        except PermissionError:
            lines.append(f"{'  ' * depth}[PERM] {rel(dir_path)}/")
            return

        for entry in entries:
            name = entry.name
            if should_skip(name, ignores) or is_hidden(entry):
                continue

            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{name}/")
                walk(entry, depth + 1)
            else:
                if only_ext:
                    if entry.suffix.lower() not in only_ext:
                        continue
                lines.append(f"{indent}{name}")

    # header
    lines.append(f"{root.name}/")
    walk(root, 1)
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Generate a directory tree into structure.txt"
    )
    ap.add_argument(
        "--root", type=str, default=".", help="Root folder to index (default=.)"
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Max directory depth (None = unlimited)",
    )
    ap.add_argument(
        "--ignore", nargs="*", default=[], help="Names to ignore (dirs/files)"
    )
    ap.add_argument(
        "--only-ext",
        nargs="*",
        default=[],
        help="Only include these file extensions (e.g. .py .csv)",
    )
    ap.add_argument(
        "--outfile", type=str, default="structure.txt", help="Output file name"
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ignores = set(DEFAULT_IGNORES) | set(args.ignore)
    only_ext = {e.lower() for e in args.only_ext}

    lines = tree(root, args.max_depth, ignores, only_ext)

    out_path = Path(args.outfile).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ Wrote tree for '{root}' → {out_path}")


if __name__ == "__main__":
    main()
