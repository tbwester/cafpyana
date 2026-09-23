"""What wrote a product: the kaon format version and the cafpyana commit.

Stamped onto every row of the ``file`` table, which every kaon tier carries, so any
input file can be traced to the cafpyana that processed it. Provenance only: nothing
downstream checks it, and tiers from different commits join as before.

Read the columns one at a time. A row of int16/uint64/int8 upcasts to float64 and
truncates the commit.

A leaf module, like ``provenance``: kaonana imports it to decode the stamp, and
``make_kaon_df`` cannot be imported off the grid.

Three gotchas decide the shape of this:

- **Numeric only.** ``run_df_maker`` writes ``format="fixed"``, where a string column
  pickles the block at ~1 MB per product (see ``make_file_df``). The commit is stored
  as the integer of its first 16 hex digits.
- **Grid jobs have no git.** ``run_grid`` ships ``git archive HEAD``, so the commit
  comes from ``COMMIT``, which ``export-subst`` fills in at archive time. The same
  archive also guarantees a grid product is clean: uncommitted edits are not shipped.
- **Pool mode reads the working tree,** where ``COMMIT`` still holds its placeholder.
  There the commit and dirty state are asked of git, once per process.
"""

import functools
import subprocess
from pathlib import Path

import numpy as np

#: The kaon product format, bumped on any change to a kaon table's columns, index, or
#: meaning. 4 is the first stamped format (kaonana v4).
KAON_FORMAT = np.int16(4)

#: ``producer_commit`` when no commit could be determined.
UNKNOWN_COMMIT = np.uint64(0)

#: ``producer_dirty`` values.
CLEAN, DIRTY, UNKNOWN_DIRTY = np.int8(0), np.int8(1), np.int8(-1)

COMMIT_FILE = Path(__file__).with_name("COMMIT")
_PLACEHOLDER = "$Format:"
_HEX_DIGITS = 16


def commit_to_int(sha: str) -> np.uint64:
    """The stored form of a commit: its first 16 hex digits as a uint64."""
    if len(sha) < _HEX_DIGITS:
        raise ValueError(f"need at least {_HEX_DIGITS} hex digits, got {sha!r}")
    return np.uint64(int(sha[:_HEX_DIGITS], 16))


def int_to_commit(value) -> str:
    """The 16-digit hex prefix a stored commit decodes to; resolve with ``git rev-parse``."""
    return f"{int(value):0{_HEX_DIGITS}x}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(COMMIT_FILE.parent), *args],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()


@functools.lru_cache(maxsize=1)
def producer() -> tuple[np.uint64, np.int8]:
    """``(commit, dirty)`` for the code that is running."""
    text = COMMIT_FILE.read_text().strip() if COMMIT_FILE.exists() else ""
    if text and not text.startswith(_PLACEHOLDER):
        return commit_to_int(text), CLEAN
    try:
        sha = _git("rev-parse", "HEAD")
        # Tracked files only: cafpyana's working tree always holds untracked
        # environments and caches that are never imported.
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT, UNKNOWN_DIRTY
    return commit_to_int(sha), DIRTY if dirty else CLEAN


def stamp_columns() -> dict:
    """The columns ``make_file_df`` adds, one value each."""
    commit, dirty = producer()
    return {"kaon_format": KAON_FORMAT, "producer_commit": commit, "producer_dirty": dirty}
