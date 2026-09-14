#!/usr/bin/env python3
"""Audit textbook and agent chapters for primary-source coverage.

This is a lightweight evidence-quality linter. It does not decide whether a
citation is scientifically correct; instead it makes missing evidence visible.

Usage:
    python 工具-scripts/原始资料审计-source_audit.py
    python 工具-scripts/原始资料审计-source_audit.py --strict

The script scans Markdown chapters under 教材-book and 智能体-agent and reports:
- total external links;
- likely primary-source links;
- whether the chapter contains an 原始资料/Primary Source section;
- whether it contains source-first evidence vocabulary;
- chapters with no likely primary-source citation.

Default mode is report-only so legacy chapters can be upgraded incrementally.
Strict mode returns non-zero when any audited chapter has no primary-like link.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOTS = (ROOT / "教材-book", ROOT / "智能体-agent")

URL_RE = re.compile(r"https?://[^)\s>]+")

PRIMARY_HOST_SUFFIXES = (
    # Paper / proceedings hosts
    "arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "aclanthology.org",
    "dl.acm.org",
    # Model / lab / vendor originals
    "openai.com",
    "cdn.openai.com",
    "deploymentsafety.openai.com",
    "anthropic.com",
    "transformer-circuits.pub",
    "deepmind.google",
    "research.google",
    "ai.google.dev",
    "developers.googleblog.com",
    "ai.meta.com",
    "mistral.ai",
    "x.ai",
    "data.x.ai",
    "huggingface.co",
    # Protocol / systems originals
    "modelcontextprotocol.io",
    "a2a-protocol.org",
    "pytorch.org",
    "vllm.ai",
    "docs.nvidia.com",
    "developer.nvidia.com",
    # Official source repositories are often the primary implementation.
    "github.com",
)

SOURCE_SECTION_PATTERNS = (
    "原始资料",
    "原始资料包",
    "一手资料",
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
    "官方规范",
)


def is_primary(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in PRIMARY_HOST_SUFFIXES)


def analyze(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    urls = URL_RE.findall(text)
    primary = [url for url in urls if is_primary(url)]
    lowered = text.lower()
    has_source_section = any(term.lower() in lowered for term in SOURCE_SECTION_PATTERNS)
    evidence_hits = sum(lowered.count(term.lower()) for term in EVIDENCE_TERMS)
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "external_links": len(urls),
        "primary_links": len(primary),
        "source_section": has_source_section,
        "evidence_terms": evidence_hits,
    }


def markdown_chapters() -> list[Path]:
    chapters: list[Path] = []
    for content_root in CONTENT_ROOTS:
        if not content_root.exists():
            continue
        chapters.extend(
            path
            for path in content_root.glob("*.md")
            if path.name != "README.md"
        )
    return sorted(chapters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rows = [analyze(path) for path in markdown_chapters()]

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
