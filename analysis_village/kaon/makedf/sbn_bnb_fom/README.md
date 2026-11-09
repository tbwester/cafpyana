# sbn_bnb_fom (vendored)

These are unchanged copies of SBNSoftware/sbncode PR #678 (production/sbnd-gen2, "BNB beam
quality for Gen2"). The source is `sbncode/BeamSpillInfoRetriever/python/` at commit
4e8960f1a495c1a9d6c04b8691c2eb9a92096734. They are the reference implementation of
SBN-doc-49735 v1.1 (BNB beam data quality).

| file | used for |
|---|---|
| `dump_bnb_quality.py` | `read_file` / `to_frames`: rec.hdr.bnbinfo and rec.hdr.spillbnbinfo, with the multiwire profiles |
| `bnb_fom.py` | Tier 1: `fill_offsets`, `compute_fom` (legacy and fixed) |
| `bnb_fom_fast.py` | numba kernels installed into `bnb_fom` on import |
| `fom_fixups.py` | Tier 2 per flatcaf (`fix_group`) with the SBND options, neighbour fill off |
| `neighbor_fill.py` | `keyhash` and the fill cuts `fom_fixups` imports |

`../beam_fom.py` is the only caller. Do not edit these files. Update them by copying a new
commit of the PR and changing `PR678_COMMIT` in `../beam_fom.py`, because `getFOM.cpp` and
`bnb_fom.py` are versioned together upstream. numba and scipy are needed (`envs/pip_requirements.txt`).
