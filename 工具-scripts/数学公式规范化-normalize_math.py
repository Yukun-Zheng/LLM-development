#!/usr/bin/env python3
r"""Normalize textbook math for GitHub-flavored Markdown.

The textbook is authored in Markdown and rendered directly on GitHub. GitHub's
math renderer supports `$...$` for inline math and `$$...$$` for display math,
but its Markdown/HTML preprocessing and math sanitizer are not identical to a
full TeX installation.

This normalizer therefore performs compatibility passes outside fenced code
blocks:

1. Normalize math delimiters
   `\(...\)` -> `$...$`
   `\[...\]` -> `$$...$$`

2. Rewrite known GitHub-incompatible macros
   `\operatorname{foo}` -> `\mathrm{foo}`
   `\operatorname*{foo}` -> `\mathrm{foo}`

3. Rewrite raw angle relations *inside math only*
   `<=` -> `\le`
   `>=` -> `\ge`
   `<`  -> `\lt`
   `>`  -> `\gt`

The third pass is important for expressions such as `x_{<t}`. Raw angle
brackets can interact badly with GitHub's Markdown/HTML preprocessing and lead
to misleading MathJax errors such as "Extra open brace or missing close brace".

Fenced code blocks (Mermaid, Python, shell, text, fenced math examples, etc.) are
left untouched so literal examples and source code are never rewritten.

Run from the repository root:

    python 工具-scripts/数学公式规范化-数学公式规范化-normalize_math.py

The script scans every Markdown file in the repository except `.git` and is
idempotent: running it repeatedly produces no further changes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Match an opening/closing fenced-code marker after optional indentation.
FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})")

# GitHub rejects \operatorname in Markdown math. \mathrm gives the upright
# visual treatment needed for names such as softmax, Var, MHA, FFN, Attention,
# Concat, KL, etc. The optional star covers \operatorname* too.
OPERATORNAME_RE = re.compile(r"\\operatorname\*?\{([^{}]+)\}")

# Same-line GitHub math spans. Display spans are processed before inline spans.
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$")
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _single_line_display_math(line: str) -> tuple[str, int]:
    r"""Convert same-line \[ ... \] to $$ ... $$ outside code fences."""
    return re.subn(r"\\\[(.+?)\\\]", r"$$\1$$", line)


def _inline_math(line: str) -> tuple[str, int]:
    r"""Convert same-line \( ... \) to $ ... $ outside code fences."""
    return re.subn(r"\\\((.+?)\\\)", r"$\1$", line)


def _github_compatible_macros(line: str) -> tuple[str, int]:
    r"""Rewrite LaTeX macros known to be rejected by GitHub's math renderer."""
    return OPERATORNAME_RE.subn(r"\\mathrm{\1}", line)


def _safe_relations(fragment: str) -> tuple[str, int]:
    r"""Replace raw angle relations in one math fragment with TeX commands."""
    count = 0

    # Do two-character relations first so <= is not converted into \lt =.
    fragment, n = re.subn(r"(?<!\\)<=", r"\\le ", fragment)
    count += n
    fragment, n = re.subn(r"(?<!\\)>=", r"\\ge ", fragment)
    count += n

    fragment, n = re.subn(r"(?<!\\)<", r"\\lt ", fragment)
    count += n
    fragment, n = re.subn(r"(?<!\\)>", r"\\gt ", fragment)
    count += n

    return fragment, count


def _rewrite_same_line_math_relations(line: str) -> tuple[str, int]:
    """Rewrite angle relations only inside same-line $...$ / $$...$$ spans."""
    replacements = 0

    def replace_display(match: re.Match[str]) -> str:
        nonlocal replacements
        content, count = _safe_relations(match.group(1))
        replacements += count
        return f"$${content}$$"

    line = DISPLAY_MATH_RE.sub(replace_display, line)

    def replace_inline(match: re.Match[str]) -> str:
        nonlocal replacements
        content, count = _safe_relations(match.group(1))
        replacements += count
        return f"${content}$"

    line = INLINE_MATH_RE.sub(replace_inline, line)
    return line, replacements


def normalize_markdown(text: str) -> tuple[str, int]:
    """Return normalized Markdown and the number of replacements."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    in_display_math = False
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

        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        body = line[: -len(newline)] if newline else line
        stripped = body.strip()
        indent = body[: len(body) - len(body.lstrip())]

        # Convert standalone LaTeX display delimiters while tracking whether
        # subsequent lines are inside display math.
        if stripped == r"\[":
            out.append(f"{indent}$${newline}")
            in_display_math = True
            replacements += 1
            continue
        if stripped == r"\]":
            out.append(f"{indent}$${newline}")
            in_display_math = False
            replacements += 1
            continue

        # Existing GitHub display-math delimiters also toggle the state.
        if stripped == "$$":
            in_display_math = not in_display_math
            out.append(line)
            continue

        converted, count = _single_line_display_math(line)
        replacements += count

        converted, count = _inline_math(converted)
        replacements += count

        converted, count = _github_compatible_macros(converted)
        replacements += count

        if in_display_math:
            # Every character on this line belongs to the display formula.
            converted, count = _safe_relations(converted)
            replacements += count
        else:
            # Outside display math, touch angle brackets only inside explicit
            # same-line inline/display math spans; prose and HTML stay intact.
            converted, count = _rewrite_same_line_math_relations(converted)
            replacements += count

        out.append(converted)

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
                f"({replacements} replacements)"
            )

    if changed_files == 0:
        print("Math syntax already normalized for GitHub Markdown.")
    else:
        print(
            f"Done: {changed_files} Markdown files changed, "
            f"{total_replacements} replacements."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
