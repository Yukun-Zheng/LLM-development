#!/usr/bin/env python3
"""Audit textbook chapters for primary-source coverage.

This is a lightweight evidence-quality linter. It does not decide whether a
citation is scientifically correct; instead it makes missing evidence visible.

Usage:
    python 工具-scripts/原始资料审计-source_audit.py

The script scans Markdown chapters under 教材-book and reports:
- total external links;
- likely primary-source links;
- whether the chapter contains an 原始资料/原始资料包 section;
- whether it contains source-first evidence vocabulary;
- a rough maturity warning.

Exit code is always 0 by default so legacy chapters are not blocked while the
book is being upgraded. Pass --strict to return non-zero when chapters have no
likely primary-source citation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "教材-book"

URL_RE = re.compile(r"https?://[^)\s>]+")

PRIMARY_HOST_SUFFIXES = (
    "arxiv.org",
    "openai.com",
    "cdn.openai.com",
    "deploymentsafety.openai.com",
    "anthropic.com",
    "transformer-circuits.pub",
    "deepmind.google",
    "ai.meta.com",
    "mistral.ai",
    "x.ai",
    "data.x.ai",
    "pytorch.org",
    "github.com",
    "dl.acm.org",
)

SOURCE_SECTION_PATTERNS = (
    "原始资料",
    "原始资料包",
    "Primary Source",
    "Primary Sources",
)

EVIDENCE_TERMS = (
    "原论文",
    "官方代码",
    "技术报告",
    "model card",
    "system card",
    "独立复现",
    "后续证据",
    "原始主张",
)


def is_primary(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in PRIMARY_HOST_SUFFIXES)


def analyze(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    urls = URL_RE.findall(text)
    primary = [u for u in urls if is_primary(u)]
    has_source_section = any(term.lower() in text.lower() for term in SOURCE_SECTION_PATTERNS)
    evidence_hits = sum(text.lower().count(term.lower()) for term in EVIDENCE_TERMS)
    return {
        "file": path.name,
        "external_links": len(urls),
        "primary_links": len(primary),
        "source_section": has_source_section,
        "evidence_terms": evidence_hits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    chapters = sorted(p for p in BOOK.glob("*.md") if p.name != "README.md")
    rows = [analyze(path) for path in chapters]

    print("# Primary-source coverage audit")
    print()
    print("| Chapter | Links | Primary-like | Source section | Evidence markers |")
    print("|---|---:|---:|---|---:|")

    failures = 0
    for row in rows:
        primary = int(row["primary_links"])
        section = "yes" if row["source_section"] else "no"
        if primary == 0:
            failures += 1
        print(
            f'| {row["file"]} | {row["external_links"]} | {primary} | '
            f'{section} | {row["evidence_terms"]} |'
        )

    print()
    print(f"Chapters audited: {len(rows)}")
    print(f"Chapters with no likely primary-source links: {failures}")
    print()
    print(
        "Note: this is a coverage heuristic, not a scientific-validity checker. "
        "A chapter can cite primary sources and still interpret them incorrectly."
    )

    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
