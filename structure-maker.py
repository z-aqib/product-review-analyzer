#!/usr/bin/env python3
"""
structure_beautiful_sizes.py
Generate a pretty Unicode tree for the current directory with per-file sizes and
aggregate folder sizes, and write it to structure.txt.

Usage:
  python structure_beautiful_sizes.py            # full tree
  python structure_beautiful_sizes.py 2          # limit depth to 2
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional

# Folders/files to ignore (add more if needed)
DEFAULT_IGNORES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".DS_Store",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "env",
    ".env",
    ".ipynb_checkpoints",
    ".mlops-venv",
}


def human_size(n: int) -> str:
    """Convert bytes to a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024


def is_hidden(p: Path) -> bool:
    # Keep .gitignore/.gitattributes/.gitkeep visible if desired
    allow = {".gitignore", ".gitattributes", ".gitkeep"}
    return p.name.startswith(".") and p.name not in allow


class Node:
    __slots__ = ("path", "name", "is_dir", "size", "children", "error", "is_link")

    def __init__(self, path: Path):
        self.path: Path = path
        self.name: str = path.name or str(path)
        self.is_dir: bool = False
        self.size: int = 0
        self.children: List[Node] = []
        self.error: Optional[str] = None
        self.is_link: bool = path.is_symlink()


def scan(
    path: Path,
    ignores: set,
    follow_symlinks: bool = False,
    max_depth: Optional[int] = None,
    depth: int = 0,
) -> Node:
    """
    Build a tree of Nodes with sizes. Folder size is the sum of all descendant files.
    """
    node = Node(path)
    try:
        stat = path.lstat() if not follow_symlinks else path.stat()
    except Exception as e:
        node.error = f"{type(e).__name__}"
        node.size = 0
        return node

    node.is_dir = path.is_dir() if (follow_symlinks or not node.is_link) else False

    # Files: size from stat; Dirs: sum of children
    if not node.is_dir:
        node.size = stat.st_size
        return node

    # Directory
    if max_depth is not None and depth >= max_depth:
        # Don’t descend; compute size as 0 (unknown) to keep fast at cutoff depth
        node.size = 0
        return node

    try:
        entries = list(path.iterdir())
    except PermissionError:
        node.error = "PermissionError"
        node.size = 0
        return node
    except Exception as e:
        node.error = f"{type(e).__name__}"
        node.size = 0
        return node

    # Filter + sort: folders first, then files; alphabetical
    filtered: List[Path] = []
    for e in entries:
        name = e.name
        if is_hidden(e) or name in ignores:
            continue
        filtered.append(e)

    filtered.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

    total = 0
    for e in filtered:
        child = scan(e, ignores, follow_symlinks, max_depth, depth + 1)
        node.children.append(child)
        total += child.size
    node.size = total
    return node


def render(node: Node, prefix: str = "", is_last: bool = True) -> List[str]:
    """
    Render the Node tree with connectors and human-readable sizes.
    """
    connector = ""
    if prefix:
        connector = "└── " if is_last else "├── "

    label_parts: List[str] = [node.name]
    if node.is_link:
        # Show link target if available
        try:
            target = os.readlink(node.path)
            label_parts.append(f" -> {target}")
        except OSError:
            label_parts.append(" (symlink)")
    if node.error:
        label_parts.append(f" [! {node.error}]")

    label = "".join(label_parts)
    size_str = f" [{human_size(node.size)}]" if node.size >= 0 else " [–]"

    lines = [f"{prefix}{connector}{label}{size_str}"]

    if node.children:
        ext = "    " if is_last else "│   "
        for i, child in enumerate(node.children):
            last = i == len(node.children) - 1
            lines.extend(render(child, prefix + ext, last))
    return lines


def main():
    root = Path(".").resolve()
    # Optional: max depth from argv
    max_depth = None
    if len(sys.argv) >= 2:
        try:
            max_depth = int(sys.argv[1])
        except ValueError:
            print(
                "If provided, the first argument must be an integer max depth.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Build the model
    tree = scan(
        root, ignores=DEFAULT_IGNORES, follow_symlinks=False, max_depth=max_depth
    )

    # Compose text: show the root like "folder/"
    header = f"{root.name}/ [{human_size(tree.size)}]" if tree.size else f"{root.name}/"
    lines = [header]
    for i, child in enumerate(tree.children):
        lines.extend(render(child, "", i == len(tree.children) - 1))

    out = "\n".join(lines)
    (root / "structure.txt").write_text(out, encoding="utf-8")
    print(f"✅ Wrote tree with sizes to {root / 'structure.txt'}")


if __name__ == "__main__":
    main()
