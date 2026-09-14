#!/usr/bin/env python3
"""Rename reader-facing repository paths to Chinese-first bilingual names.

This is intentionally idempotent. It updates chapter / appendix filenames,
reader-facing top-level directories, and textual references in Markdown/Python.
GitHub workflow files are intentionally left untouched here and are updated
separately through the GitHub connector, avoiding workflow-token restrictions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILE_RENAMES = {
    "book/00-preface.md": "book/00-导读-preface.md",
    "book/01-foundations-transformer.md": "book/01-Transformer基础-foundations-transformer.md",
    "book/02-scaling-gpt3.md": "book/02-规模定律与GPT3-scaling-gpt3.md",
    "book/03-alignment-chatgpt.md": "book/03-对齐与ChatGPT-alignment-chatgpt.md",
    "book/04-modern-architecture.md": "book/04-现代LLM架构-modern-architecture.md",
    "book/05-open-source-efficient-ft.md": "book/05-开源与高效微调-open-source-efficient-ft.md",
    "book/06-rag-multimodal-agent.md": "book/06-RAG多模态与智能体-rag-multimodal-agent.md",
    "book/07-moe-china.md": "book/07-MoE与中国大模型-moe-china.md",
    "book/08-reasoning-era.md": "book/08-推理模型时代-reasoning-era.md",
    "book/09-training-systems.md": "book/09-训练系统-training-systems.md",
    "book/10-inference-systems.md": "book/10-推理服务系统-inference-systems.md",
    "book/11-evaluation-safety.md": "book/11-评测与安全-evaluation-safety.md",
    "book/12-frontier-2026.md": "book/12-2026前沿-frontier-2026.md",
    "appendices/A-math.md": "appendices/A-数学基础-math.md",
    "appendices/B-minigpt.md": "appendices/B-从零实现MiniGPT-minigpt.md",
    "appendices/C-post-training-math.md": "appendices/C-后训练数学-post-training-math.md",
}

DIR_RENAMES = {
    "appendices": "附录-appendices",
    "book": "教材-book",
    "code": "代码-code",
    "figures": "图表-figures",
    "references": "参考文献-references",
    "scripts": "工具-scripts",
}

SCRIPT_RENAMES = {
    "scripts/normalize_math.py": "scripts/数学公式规范化-normalize_math.py",
    "scripts/localize_headings.py": "scripts/章节标题中文化-localize_headings.py",
    "scripts/localize_file_tree.py": "scripts/文件树中文化-localize_file_tree.py",
}

TEXT_SUFFIXES = {".md", ".py"}


def git_mv(old: str, new: str) -> bool:
    src = ROOT / old
    dst = ROOT / new
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "mv", old, new], cwd=ROOT, check=True)
    print(f"renamed: {old} -> {new}")
    return True


def all_replacements() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for old, new in {**FILE_RENAMES, **SCRIPT_RENAMES}.items():
        pairs.append((old, new))
        pairs.append((Path(old).name, Path(new).name))
    for old, new in DIR_RENAMES.items():
        pairs.append((f"{old}/", f"{new}/"))
        pairs.append((f'ROOT / "{old}"', f'ROOT / "{new}"'))
        pairs.append((f"ROOT / '{old}'", f"ROOT / '{new}'"))
    return sorted(set(pairs), key=lambda x: len(x[0]), reverse=True)


def rewrite_text_references() -> int:
    pairs = all_replacements()
    changed = 0
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or ".github" in path.parts
            or path.suffix not in TEXT_SUFFIXES
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = text
        for old, replacement in pairs:
            new = new.replace(old, replacement)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
            print(f"updated references: {path.relative_to(ROOT)}")
    return changed


def main() -> int:
    renamed = 0
    for old, new in FILE_RENAMES.items():
        renamed += int(git_mv(old, new))
    for old, new in SCRIPT_RENAMES.items():
        renamed += int(git_mv(old, new))

    rewritten = rewrite_text_references()

    for old, new in DIR_RENAMES.items():
        renamed += int(git_mv(old, new))

    print(f"Done: {renamed} paths renamed, {rewritten} text files updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
