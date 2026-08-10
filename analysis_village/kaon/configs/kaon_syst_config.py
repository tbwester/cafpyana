"""Pass 2 of the two-file production: the reweight systematics, and nothing else.

``syst`` is one row per true interaction and 5,465 columns, and it depends on no
part of reconstruction -- so it is produced on its own and **reused across every
reco reprocessing**.  Previously every re-export rewrote it, bit-identical to the
last time, because a chi2 patch had moved or the pairing had changed.

Why ``hdr`` and ``file`` are here
--------------------------------
They are the join key, and they are cheap.  The two files are written by
independent passes, so each must be able to compute the key from scratch:

    (file_key, run, subrun, evt, rec.mc.nu..index)

``file`` gives ``file_key`` -- the blake2b hash of the flatcaf basename, which is
what makes the key *structurally* unique rather than accidentally so, since
``(run, subrun, evt)`` overlaps between productions.  ``hdr`` gives
``run``/``subrun``/``evt``.  ``rec.mc.nu..index`` is already the second level of
``syst``'s own index.

Nothing reconstruction-side is needed, which is the point: this product does not
know what a slice is.

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_syst_config.py ...

DO NOT merge these products with pass-1 products.  They have disjoint table sets
and a different index granularity; they are joined, not concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import make_file_df, make_syst_df
from makedf.makedf import make_hdrdf

DFS = [make_syst_df, make_hdrdf, make_file_df]
NAMES = ["syst", "hdr", "file"]
