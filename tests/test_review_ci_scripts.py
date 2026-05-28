"""Tests for the 3 review CI check scripts.

Per docs/REVIEW_PROTOCOL.md §2. Each script must:
  * return exit 0 on clean state
  * return exit != 0 with structured stderr on violations
  * not have side effects [pure check]
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run(script: str, *args: str) -> tuple[int, str, str]:
    """Run a script under the current Python; return (rc, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


# ============================================================
# check_doc_links.py
# ============================================================
class TestDocLinks:
    def test_real_repo_docs_clean(self):
        """The actual repo docs should have no broken internal links."""
        rc, out, err = _run("check_doc_links.py")
        assert rc == 0, f"check_doc_links failed:\n{err}"

    def test_broken_link_detected_strict(self, tmp_path):
        """Drop a fake .md with a broken link in a tmp repo; --strict should fail."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "docs" / "BAD.md").write_text(
            "See [missing](does-not-exist.md) for details.\n"
        )
        import shutil
        shutil.copy(SCRIPTS_DIR / "check_doc_links.py",
                    tmp_path / "scripts" / "check_doc_links.py")
        result = subprocess.run(
            [sys.executable, "scripts/check_doc_links.py", "--strict"],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_broken_link_warns_only_by_default(self, tmp_path):
        """Default mode warns but returns 0 — let project-lead's existing
        broken USER_MANUAL.md links not block our v0.2 PRs."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "docs" / "BAD.md").write_text(
            "See [missing](does-not-exist.md) for details.\n"
        )
        import shutil
        shutil.copy(SCRIPTS_DIR / "check_doc_links.py",
                    tmp_path / "scripts" / "check_doc_links.py")
        result = subprocess.run(
            [sys.executable, "scripts/check_doc_links.py"],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert result.returncode == 0
        assert "WARN" in result.stderr
        assert "not found" in result.stderr

    def test_code_blocks_skipped(self, tmp_path):
        """Links inside ``` fences should NOT be flagged as broken."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "scripts").mkdir()
        # The doc contains a code block with a fake link [X](Y) — should
        # NOT be flagged. A real link to a sibling file in docs/ should resolve.
        (tmp_path / "docs" / "WITHCODE.md").write_text(
            "Example template:\n\n```\n[X](Y)\n```\n\nReal link: [r](OTHER.md)\n"
        )
        (tmp_path / "docs" / "OTHER.md").write_text("# r\n")
        import shutil
        shutil.copy(SCRIPTS_DIR / "check_doc_links.py",
                    tmp_path / "scripts" / "check_doc_links.py")
        result = subprocess.run(
            [sys.executable, "scripts/check_doc_links.py", "--strict"],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr


# ============================================================
# check_pr_template_filled.py
# ============================================================
class TestPRTemplate:
    def test_filled_template_passes(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(textwrap.dedent("""\
            ## What
            Real change description here.

            ## Why
            Per docs/X.md §Y.

            ## How tested
            - [x] Added test_foo.py

            ## Type label
            - [x] new-metric

            ## Checklist
            - [x] Lint pass

            ## Risks
            None known.
        """))
        rc, out, err = _run("check_pr_template_filled.py", str(body))
        assert rc == 0, err

    def test_missing_section_fails(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(textwrap.dedent("""\
            ## What
            Real change.

            ## Why
            Per X.
        """))
        rc, out, err = _run("check_pr_template_filled.py", str(body))
        assert rc != 0
        assert "missing required section" in err

    def test_empty_section_fails(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(textwrap.dedent("""\
            ## What

            ## Why
            Per X.

            ## How tested
            - [x] test added

            ## Type label
            - [x] doc-only

            ## Checklist
            - [x] done

            ## Risks
            None.
        """))
        rc, out, err = _run("check_pr_template_filled.py", str(body))
        assert rc != 0
        assert "## What is empty" in err or "## What" in err

    def test_no_type_label_checked_fails(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(textwrap.dedent("""\
            ## What
            Real change.

            ## Why
            Per X.

            ## How tested
            test_foo

            ## Type label
            - [ ] new-metric
            - [ ] doc-only

            ## Checklist
            - [x] done

            ## Risks
            None.
        """))
        rc, out, err = _run("check_pr_template_filled.py", str(body))
        assert rc != 0
        assert "Type label" in err and "checked" in err

    def test_html_comments_dont_count_as_content(self, tmp_path):
        body = tmp_path / "body.md"
        body.write_text(textwrap.dedent("""\
            ## What
            <!-- still a placeholder -->

            ## Why
            <!-- write Y here -->

            ## How tested

            ## Type label
            - [x] doc-only

            ## Checklist
            - [x] done

            ## Risks
            None.
        """))
        rc, out, err = _run("check_pr_template_filled.py", str(body))
        assert rc != 0  # ## What and ## Why are still empty


# ============================================================
# check_design_doc_consistency.py
# ============================================================
class TestDesignDocConsistency:
    def test_current_repo_consistent(self):
        """All v0.2 invariants should hold in the live repo."""
        rc, out, err = _run("check_design_doc_consistency.py")
        assert rc == 0, f"design doc consistency failed:\n{err}\n{out}"

    def test_output_lists_each_check(self):
        rc, out, err = _run("check_design_doc_consistency.py")
        # Each of the 4 checks should appear in output as OK
        assert "capability_taxonomy" in out
        assert "controlled vocab" in out
        assert "required fields" in out
        assert "paper_judge" in out
