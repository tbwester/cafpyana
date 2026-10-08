"""A light pass: the CAF's shower reconstruction per pfp, with its backtracked truth.

``make_shower_df`` reads shower branches and ``rec.true_particles``, no hits, and recomputes
nothing, so this pass costs a small fraction of ``kaon_reco_config``. One row per pfp,
unfiltered beyond the clear-cosmic drop that ``track`` also applies. The Pandora hierarchy is
``kaon_geom_config``'s, on the same key; the per-universe dE/dx and energy are
``kaon_shower_calo_config``'s.

Why ``hdr`` and ``file`` are here
--------------------------------
``kaon_geom_config``'s reason: ``__ntuple`` is assigned from the order a job read its inputs, so
this pass and a reco pass number the same flatcafs differently. The key that survives is::

    (file_key, entry, rec.slc..index, pfp_index)

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_shower_config.py ...

DO NOT merge these products with the reco products. They are joined, not concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import make_file_df, make_shower_df
from makedf.makedf import make_hdrdf

DFS = [make_shower_df, make_hdrdf, make_file_df]
NAMES = ["shower", "hdr", "file"]
