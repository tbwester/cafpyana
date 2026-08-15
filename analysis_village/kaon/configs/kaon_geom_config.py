"""A third pass: the per-pfp directions and Pandora hierarchy, and nothing else.

``make_pfp_geom_df`` reads nine branches, no hits, and recomputes no chi2, so this pass costs
a small fraction of ``kaon_reco_config`` and does not have to wait for a reco reprocessing to
add an angle to the analysis.  Nine narrow columns per pfp, unfiltered beyond the clear-cosmic
drop that ``track`` also applies.

Why ``hdr`` and ``file`` are here
--------------------------------
The same reason ``kaon_syst_config`` carries them, and it matters more here because this
table joins to a *reconstruction* table rather than to truth.  ``__ntuple`` is assigned from
the order a job read its inputs, so a geom pass and a reco pass over the same flatcafs number
them differently and an index join across the two is wrong while looking right.  The key that
survives is::

    (file_key, entry, rec.slc..index, pfp_index)

``file`` gives ``file_key``, the hash of the input flatcaf's basename, and ``hdr`` gives
run/subrun/evt.  ``kaonana.data.samples.event_key_frame`` builds exactly this.

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_geom_config.py ...

DO NOT merge these products with pass-1 or pass-2 products.  They are joined, not
concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import make_file_df, make_pfp_geom_df
from makedf.makedf import make_hdrdf

DFS = [make_pfp_geom_df, make_hdrdf, make_file_df]
NAMES = ["geom", "hdr", "file"]
