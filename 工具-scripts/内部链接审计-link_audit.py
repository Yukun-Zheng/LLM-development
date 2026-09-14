#!/usr/bin/env python3
"""Check local Markdown links across the repository.

External URLs, mailto links and pure anchors are ignored. Relative Markdown
links are resolved from the source document and must point to an existing file
or directory. URL fragments and query strings are removed before resolution.

This intentionally checks local navigation only; it does not make network
requests to external sources.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

# Covers normal links and images. Fenced code is removed before matching so
# examples such as `[label](missing.md)` inside code blocks do not count.
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCED_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)


def iter_markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts
    )


def local_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if not target or target.startswith("#"):
        return None
    # Markdown permits optional titles after a URL. We only support the common
    # whitespace-separated form here and keep paths with escaped spaces intact.
    if " " in target and not target.startswith("<"):
        target = target.split(maxsplit=1)[0]
    target = target.strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    if target.startswith("mailto:"):
        return None
    path = unquote(parsed.path)
    return path or None


def audit_file(path: Path) -> list[tuple[str, str]]:
    text = FENCED_RE.sub("", path.read_text(encoding="utf-8"))
    failures: list[tuple[str, str]] = []
    for raw in LINK_RE.findall(text):
        target = local_target(raw)
        if target is None:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            failures.append((raw, "escapes repository root"))
            continue
        if not resolved.exists():
            failures.append((raw, resolved.relative_to(ROOT).as_posix()))
    return failures


def main() -> int:
    failures: list[tuple[Path, str, str]] = []
    files = iter_markdown_files()
    for path in files:
        for raw, resolved in audit_file(path):
            failures.append((path, raw, resolved))

    print(f"Markdown files audited: {len(files)}")
    if not failures:
        print("Internal Markdown links: PASS")
        return 0

    print(f"Broken internal links: {len(failures)}")
    for source, raw, resolved in failures:
        print(
            f"- {source.relative_to(ROOT).as_posix()}: {raw!r} -> {resolved}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
