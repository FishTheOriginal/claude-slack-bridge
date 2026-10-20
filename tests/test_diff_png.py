"""tools/diff_png.py: unified-diff parsing + GitHub-style PNG rendering."""

import os
import sys

import pytest

pytest.importorskip("PIL")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import diff_png  # noqa: E402

SAMPLE = (
    "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -3,3 +3,3 @@ def f():\n"
    " keep\n-old\n+new\n\n"
    "--- a/gone.sql\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-bye\n--- sql comment\n"
)


def test_parse_numbers_lines_and_names_files():
    (f1, r1), (f2, r2) = diff_png.parse(SAMPLE)
    assert f1 == "x.py"
    assert r1 == [
        ("@", "", "", "@@ -3,3 +3,3 @@ def f():"),
        (" ", 3, 3, "keep"),
        ("-", 4, "", "old"),
        ("+", "", 4, "new"),
        (" ", 5, 5, ""),  # context line whose leading space was stripped
    ]
    # Deleted file takes its name from the --- line; a removed "-- comment"
    # inside the hunk is content, not a file header.
    assert f2 == "gone.sql"
    assert r2 == [
        ("@", "", "", "@@ -1,2 +0,0 @@"),
        ("-", 1, "", "bye"),
        ("-", 2, "", "-- sql comment"),
    ]


def test_render_writes_one_png_per_file(tmp_path):
    outs = diff_png.render(SAMPLE, str(tmp_path / "c"))
    assert outs == [str(tmp_path / "c-01.png"), str(tmp_path / "c-02.png")]
    assert all(os.path.getsize(p) > 0 for p in outs)
