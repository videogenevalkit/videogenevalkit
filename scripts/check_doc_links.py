#!/usr/bin/env python3
"""check_doc_links.py — verify internal markdown links resolve.

Per docs/REVIEW_PROTOCOL.md §2: run on every PR to catch dead links in
design docs early.

Scope:
  * Scans docs/*.md, README.md, .claude/skills/**/*.md, .github/*.md
  * For each `[text](path)` link OUTSIDE fenced code blocks and inline code:
      - Skip external (http://, https://, mailto:)
      - Skip anchors-only (#xxx)
      - Resolve relative path against the doc's directory
      - Check the target file exists; if it has a #anchor, that's fine
        [we don't check anchor names — too noisy for v0.2]
  * Default: report broken links, exit 0 [warn-only — project-lead docs may
    have existing broken links not in our scope to fix in this PR].
  * `--strict`: exit 1 on any broken link.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Matches Markdown [text](path) link — not images ![alt](path)
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
# Fenced code block boundary
FENCE_RE = re.compile(r"^```")


def strip_code_blocks_and_inline(text: str) -> str:
    """Replace fenced code-block content and inline `code` with whitespace so
    link regex doesn't match inside them."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line.lstrip()):
            in_fence = not in_fence
            out.append("\n")
            continue
        if in_fence:
            out.append("\n")
            continue
        # Strip inline `code` content but preserve length
        scrubbed = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), line)
        out.append(scrubbed)
    return "".join(out)


def find_docs(root: Path) -> list[Path]:
    """All .md files in scope."""
    paths: list[Path] = []
    paths.extend((root / "docs").rglob("*.md"))  # recurse into wiki/ design/
    if (root / "README.md").exists():
        paths.append(root / "README.md")
    for d in [root / ".claude/skills", root / ".github"]:
        if d.is_dir():
            paths.extend(d.rglob("*.md"))
    return sorted(paths)


def is_external(url: str) -> bool:
    return url.startswith(("http://", "https://", "mailto:", "ftp://"))


def check_doc(doc: Path, repo_root: Path) -> list[str]:
    """Return list of error strings for this doc."""
    errors: list[str] = []
    raw_text = doc.read_text(encoding="utf-8", errors="replace")
    text = strip_code_blocks_and_inline(raw_text)
    for m in LINK_RE.finditer(text):
        link_text, url = m.group(1), m.group(2).strip()
        if is_external(url):
            continue
        # Strip any anchor fragment
        target_path = url.split("#", 1)[0].strip()
        if not target_path:
            # pure anchor link (#section) — fine
            continue
        # Resolve relative to the doc's directory
        target = (doc.parent / target_path).resolve()
        # Also allow absolute /-paths to be relative to repo root
        if target_path.startswith("/"):
            target = (repo_root / target_path.lstrip("/")).resolve()
        if not target.exists():
            line_no = text[: m.start()].count("\n") + 1
            rel_doc = doc.relative_to(repo_root)
            errors.append(
                f"  {rel_doc}:{line_no}  [{link_text}]({url})  →  not found: {target}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on any broken link; default warn-only.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    docs = find_docs(repo_root)
    if not docs:
        print("check_doc_links: no markdown files found", file=sys.stderr)
        return 0

    total_errors: list[str] = []
    for doc in docs:
        total_errors.extend(check_doc(doc, repo_root))

    if total_errors:
        level = "FAIL" if args.strict else "WARN"
        print(f"check_doc_links: {level} — {len(total_errors)} broken link(s):",
              file=sys.stderr)
        for e in total_errors:
            print(e, file=sys.stderr)
        return 1 if args.strict else 0
    print(f"check_doc_links: OK — {len(docs)} docs scanned, all internal links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
