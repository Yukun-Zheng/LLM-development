#!/usr/bin/env python3
"""Make top-level textbook part headings Chinese-first while preserving Part IDs.

Examples:
    # Part I ...   -> # 第一篇（Part I） ...
    # Part VIII ... -> # 第八篇（Part VIII） ...

Only the first line of files under 教材-book/ is considered. The script is
idempotent and deliberately does not rewrite technical terms inside the body.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "教材-book"

PARTS = {
    "I": "第一篇",
    "II": "第二篇",
    "III": "第三篇",
    "IV": "第四篇",
    "V": "第五篇",
    "VI": "第六篇",
    "VII": "第七篇",
    "VIII": "第八篇",
    "IX": "第九篇",
    "X": "第十篇",
    "XI": "第十一篇",
    "XII": "第十二篇",
}

PART_RE = re.compile(r"^# Part (XII|XI|IX|VIII|VII|VI|IV|V|III|II|I)([\u3000\s]+)(.*)$")


def localize(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if not lines:
        return False

    first = lines[0].rstrip("\r\n")
    newline = "\r\n" if lines[0].endswith("\r\n") else "\n" if lines[0].endswith("\n") else ""
    match = PART_RE.match(first)
    if not match:
        return False

    roman, spacing, rest = match.groups()
    lines[0] = f"# {PARTS[roman]}（Part {roman}）{spacing}{rest}{newline}"
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> int:
    changed = []
    for path in sorted(BOOK.glob("*.md")):
        if path.name == "README.md":
            continue
        if localize(path):
            changed.append(path.relative_to(ROOT))

    if changed:
        print("Localized chapter headings:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Chapter headings already use Chinese-first labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
