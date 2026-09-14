#!/usr/bin/env python3
"""Normalize LaTeX delimiters for GitHub-flavored Markdown.

The textbook is authored in Markdown and rendered directly on GitHub. GitHub's
native math syntax uses `$...$` for inline math and `$$...$$` for display math.
LaTeX document delimiters such as `\(...\)` and `\[...\]` are therefore
normalized here.

The transformation deliberately skips fenced code blocks (including Mermaid,
Python, shell, text, and fenced `math` blocks) so examples and source code are
never rewritten accidentally.

Run from the repository root:

    python scripts/normalize_math.py

The script scans every Markdown file in the repository except `.git` and is
idempotent: running it repeatedly produces no further changes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Match an opening/closing fenced-code marker after optional indentation.
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})")


def _single_line_display_math(line: str) -> str:
    """Convert same-line \[ ... \] to $$ ... $$ outside code fences."""
    # Keep this conservative: do not span newlines and require both delimiters.
    return re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", line)


def normalize_markdown(text: str) -> tuple[str, int]:
    """Return normalized Markdown and the number of delimiter replacements."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    replacements = 0

    for line in lines:
        match = FENCE_RE.match(line)
        if match:
            token = match.group("fence")
            char = token[0]
            length = len(token)

            if not in_fence:
                in_fence = True
                fence_char = char
                fence_len = length
            elif char == fence_char and length >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0

            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        # Preserve indentation and newline style for display delimiters that
        # occupy their own line. GitHub renders `$$` cleanly as a block.
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[: -len(newline)] if newline else line
        stripped = body.strip()
        indent = body[: len(body) - len(body.lstrip())]

        if stripped == r"\[":
            out.append(f"{indent}$${newline}")
            replacements += 1
            continue
        if stripped == r"\]":
            out.append(f"{indent}$${newline}")
            replacements += 1
            continue

        # Same-line display math is rare in the book, but normalize it too.
        converted = _single_line_display_math(line)
        if converted != line:
            replacements += line.count(r"\[") + line.count(r"\]")
            line = converted

        # Inline LaTeX delimiters -> GitHub inline math delimiters.
        open_count = line.count(r"\(")
        close_count = line.count(r"\)")
        if open_count or close_count:
            line = line.replace(r"\(", "$").replace(r"\)", "$")
            replacements += open_count + close_count

        out.append(line)

    return "".join(out), replacements


def markdown_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if ".git" not in p.parts
    )


def main() -> int:
    changed_files = 0
    total_replacements = 0

    for path in markdown_files():
        original = path.read_text(encoding="utf-8")
        normalized, replacements = normalize_markdown(original)
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")
            changed_files += 1
            total_replacements += replacements
            print(
                f"normalized {path.relative_to(ROOT)} "
                f"({replacements} delimiter replacements)"
            )

    if changed_files == 0:
        print("Math syntax already normalized for GitHub Markdown.")
    else:
        print(
            f"Done: {changed_files} Markdown files changed, "
            f"{total_replacements} delimiter replacements."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
