#!/usr/bin/env python3
"""check_pr_template_filled.py — verify the PR description follows the template.

Per docs/REVIEW_PROTOCOL.md §2 + .github/pull_request_template.md:

  Required H2 sections in PR body:
    ## What
    ## Why
    ## How tested
    ## Type label
    ## Checklist
    ## Risks

  Additionally:
    * Each section must have non-empty body [not just the heading]
    * At least one `## Type label` checkbox must be ticked [- [x] ...]
    * Allow ## Out of scope as optional

  Exit 0 if PR body conforms; exit 1 with structured diff if not.

Usage:
  python scripts/check_pr_template_filled.py <pr_body_file>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = ["What", "Why", "How tested", "Type label", "Checklist", "Risks"]

H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
CHECKED_RE = re.compile(r"^\s*-\s*\[\s*[xX]\s*\]", re.MULTILINE)


def parse_sections(body: str) -> dict[str, str]:
    """Return {section_heading: body_text} for all ## sections found."""
    matches = list(H2_RE.finditer(body))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[heading] = body[start:end].strip()
    return sections


def is_placeholder(content: str) -> bool:
    """Return True if section body is empty or just an HTML comment."""
    if not content:
        return True
    # Strip HTML comments
    no_comments = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL).strip()
    return not no_comments


def check_body(body: str) -> list[str]:
    """Return list of error messages [empty if OK]."""
    errors: list[str] = []
    sections = parse_sections(body)
    for required in REQUIRED_SECTIONS:
        if required not in sections:
            errors.append(f"missing required section: ## {required}")
        elif is_placeholder(sections[required]):
            errors.append(f"section ## {required} is empty / placeholder only")

    # At least one type label must be ticked
    type_section = sections.get("Type label", "")
    if not CHECKED_RE.search(type_section):
        errors.append(
            "## Type label has no checked box ([x]) — pick at least one "
            "PR type per REVIEW_PROTOCOL §3"
        )

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <pr_body_file>", file=sys.stderr)
        return 2
    body_file = Path(sys.argv[1])
    if not body_file.is_file():
        print(f"PR body file not found: {body_file}", file=sys.stderr)
        return 2
    body = body_file.read_text(encoding="utf-8", errors="replace")
    errors = check_body(body)
    if errors:
        print("check_pr_template_filled: FAIL", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nFix: open the PR and edit description to follow "
            ".github/pull_request_template.md", file=sys.stderr,
        )
        return 1
    print("check_pr_template_filled: OK — all required sections present + filled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
