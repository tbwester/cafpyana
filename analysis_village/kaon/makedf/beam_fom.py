"""The SBN-doc-49735 beam-quality result for each spill of a data flatcaf.

`spill_quality` runs the vendored sbncode PR #678 code (`sbn_bnb_fom/`) on one file: Tier 1 per
spill, then the Tier-2 fixups with the SBND options over the file's spills, with the neighbour
fill (step B) off. Each event's triggering spill takes the decision of the same spill in
rec.hdr.bnbinfo; with an empty bnbinfo (blinded CAFs) it keeps its own Tier-1 result.

The neighbour fill is off because a triggered spill is the last of its event's block, so a fill
limited to one file recovers untriggered spills and almost never triggered ones. That would make
the removed spills trigger-rich and bias every rate per POT (kaonana studies/krolldq/fommaker:
-0.24% on kdev, none without the fill).
"""

import os
import sys

import numpy as np
import pandas as pd

PR678_COMMIT = "4e8960f1a495c1a9d6c04b8691c2eb9a92096734"
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbn_bnb_fom")

#: The beam-quality fields added to ``beam`` and ``trigger_spill``.
FIELDS = {"FOM_best": np.float32, "fom_source": np.int8, "fom_status": np.int32,
          "tor_used": np.float64}

#: fom_fixups' neighbour fill needs both neighbours' FOM above this; above 1 it never fires.
NO_NEIGHBOR_FILL = 1.1

_cache = {}


def _modules():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import bnb_fom
    import bnb_fom_fast  # noqa: F401  (installs the numba kernels into bnb_fom)
    import dump_bnb_quality
    import fom_fixups
    from neighbor_fill import keyhash
    return bnb_fom, dump_bnb_quality, fom_fixups, keyhash


def _tier1(bnb_fom, df, mw):
    dff, filled = bnb_fom.fill_offsets(df)
    fix = bnb_fom.compute_fom(dff, mw, mode="fixed", offsets_filled=filled)
    out = df.copy()
    for k, v in fix.items():
        out[k] = v
    return out


def _tier2(fom_fixups, sb):
    d = {k: sb[k].to_numpy() for k in fom_fixups.COLS if k in sb}
    t = d["spill_time_sec"].astype(np.int64) * 1_000_000_000 + d["spill_time_nsec"].astype(np.int64)
    o = np.lexsort((t, d["run"]))
    d = {k: v[o] for k, v in d.items()}
    t = t[o]
    keep = np.ones(len(t), bool)
    keep[1:] = ~((t[1:] == t[:-1]) & (d["run"][1:] == d["run"][:-1]))
    d = {k: v[keep] for k, v in d.items()}
    d["t"] = t[keep] / 1e9
    stats = dict.fromkeys(_STATS, 0.0)
    saved = fom_fixups.NB_MIN_FOM
    fom_fixups.NB_MIN_FOM = NO_NEIGHBOR_FILL
    try:
        upd = fom_fixups.fix_group(d, stats)
    finally:
        fom_fixups.NB_MIN_FOM = saved
    o = np.argsort(upd["h"])
    return {k: v[o] for k, v in upd.items()}


_STATS = ("pot", "pot_before", "hptg2_pot", "hptg2_spills", "hptg2_dpos_sum", "tor875_spills",
          "tor875_pot_gain", "fallback_pot", "wide_pot", "wide_nb_pot", "mmb_pot", "mmb_nb_pot",
          "nan_bpm_pot", "nb_fill_pot", "nb_fill_fallback_pot", "nan_pot_before", "nan_pot_after",
          "p95_before", "p95_after", "sel_pot", "sel_pass_before", "sel_pass_after",
          "sel_nan_before", "sel_nan_after", "sel_tu_pot", "sel_tu_pass")


def _apply(fom_fixups, keyhash, df, upd):
    out = pd.DataFrame(index=df.index)
    out["tor_used"] = fom_fixups.tor_used(df.TOR860.to_numpy(), df.TOR875.to_numpy())[0]
    for k in ("FOM_best", "fom_source", "fom_status"):
        out[k] = df[k].to_numpy().copy()
    if upd is not None and len(upd["h"]):
        h = keyhash(df.run.to_numpy(), df.subrun.to_numpy(), df.spill_time_sec.to_numpy(),
                    df.spill_time_nsec.to_numpy())
        j = np.clip(np.searchsorted(upd["h"], h), 0, len(upd["h"]) - 1)
        hit = upd["h"][j] == h
        for k in ("FOM_best", "fom_source", "fom_status"):
            v = out[k].to_numpy().copy()
            v[hit] = upd[k][j[hit]].astype(v.dtype)
            out[k] = v
    return out.astype(FIELDS)


def spill_quality(f):
    """(every bnbinfo spill, each record's triggered spill) of the flatcaf `f`, in the order of
    rec.hdr.bnbinfo flattened over records and of the records, each with `FIELDS` and the spill
    time to align on. Cached for the last file, since two makers ask for it."""
    tree = f["recTree"]
    key = (id(tree), getattr(f, "file_path", ""))
    if key in _cache:
        return _cache[key]
    bnb_fom, dump, fom_fixups, keyhash = _modules()
    # read_file opens its path; hand it the tree cafpyana already has open.
    opener = dump.open_tree
    dump.open_tree = lambda _path: tree
    try:
        sp, sp_mw, sb, sb_mw = dump.to_frames([dump.read_file(key[1] or "flatcaf")])
    finally:
        dump.open_tree = opener
    sb1 = _tier1(bnb_fom, sb, sb_mw) if len(sb) else None
    sp1 = _tier1(bnb_fom, sp, sp_mw) if len(sp) else None
    upd = _tier2(fom_fixups, sb1) if sb1 is not None else None
    time_cols = ["spill_time_sec", "spill_time_nsec"]
    beam = (pd.concat([sb1[time_cols].reset_index(drop=True),
                       _apply(fom_fixups, keyhash, sb1, upd).reset_index(drop=True)], axis=1)
            if sb1 is not None else None)
    trig = (pd.concat([sp1[time_cols].reset_index(drop=True),
                       _apply(fom_fixups, keyhash, sp1, upd).reset_index(drop=True)], axis=1)
            if sp1 is not None else None)
    _cache.clear()
    _cache[key] = (beam, trig)
    return beam, trig


def attach(frame, quality):
    """`frame` (beam or trigger_spill, rows in the same order as `quality`) with `FIELDS` added;
    the spill times must agree row by row."""
    if quality is None or len(quality) != len(frame):
        raise ValueError(f"beam quality has {0 if quality is None else len(quality)} rows for "
                         f"{len(frame)} spills")
    # Compared in the frame's dtype: trigger_spill narrows the times to uint32, which turns the
    # no-spill sentinel (uint64 max) into 4294967295.
    for c in ("spill_time_sec", "spill_time_nsec"):
        if not np.array_equal(frame[c].to_numpy(), quality[c].to_numpy().astype(frame[c].dtype)):
            raise ValueError(f"beam quality is not aligned with the spills ({c})")
    out = frame.copy()
    for k in FIELDS:
        out[k] = quality[k].to_numpy()
    return out
