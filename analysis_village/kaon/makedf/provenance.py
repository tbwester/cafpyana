"""The one definition of a flatcaf's key.

`file_key` used to exist twice -- here in ``make_kaon_df`` and again in
``kaonana.analysis.provenance`` -- because the reading end could not import the
writing end: ``make_kaon_df`` imports ``makedf.chi2pid``, which opens its dE/dx
templates from ``/cvmfs`` at module scope, so importing it off the grid fails
before any function is reached.  The two copies were held together by a golden
digest in ``tests/test_analysis_provenance.py``.

Two copies was tolerable while the key only decided a fold.  It is not tolerable
now that it joins two *files*: a key computed one way in the writer and another
way in the reader does not raise, it returns zero matched rows, and that reads
like a data problem rather than a schema one.

So the definition lives here, in a leaf module that imports nothing heavier than
``hashlib`` and ``numpy``.  ``make_kaon_df`` imports it, ``kaonana`` imports it,
and the digest test now pins one function instead of reconciling two.

**Changing anything in this module invalidates every product ever written.**
The key is stored in the products, and a join between a product written before
a change and one written after silently matches nothing.  If it must ever
change, bump the version string the writers record alongside it so the
mismatch is loud.
"""

import hashlib

import numpy as np

#: ``file_key`` of a file that could not be named -- an unpatched product, or a
#: loader path that lost the name.  A real name collides with probability 2**-64.
UNKNOWN_FILE_KEY = np.uint64(0)

#: What the key *is*, recorded by writers so a reader can refuse a mismatch
#: rather than silently joining nothing.  Bump on any change to :func:`file_key`.
FILE_KEY_VERSION = "blake2b8(basename)/v1"


def file_key(name: str) -> np.uint64:
    """64-bit key for a flatcaf basename.

    blake2b, NOT the built-in ``hash()``: string hashing is salted per process,
    so ``hash()`` would give a different key in every worker of the same run.

    Truncated to 64 bits because that is what fits a numpy column and is
    comfortably enough -- the birthday collision probability is ~2.7e-8 at 1e6
    files (32 bits would expect ~116 collisions there, and does not).  Over the
    14,821 names in ``lists/kex_list.txt`` there are none.

    The *basename* is hashed, not the path, so the key survives a file being
    moved or read through a different mount.  That makes it the production's
    guarantee of unique filenames that this rests on, rather than ours.
    """
    if not name:
        return UNKNOWN_FILE_KEY
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "big"))
