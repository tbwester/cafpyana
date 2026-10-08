"""The shower dE/dx and energy in every calorimetry universe of the ``default`` treatment.

Its own pass, as ``kaon_calo_config`` is for tracks, so a calorimetry model change is a respin
of this tier and not of ``kaon_shower_config``. It reads **no hits**: each universe is derived
from the CAF's per-plane shower dE/dx and energy (``make_shower_calo_df``), so the pass is
cheap. ``joint`` is not produced -- its per-hit smear does not commute with the shower dE/dx
median -- and ``make_shower_calo_df`` refuses it.

On data only ``cv`` is written, as ``make_calo_df`` does.

Why ``hdr`` and ``file`` are here
--------------------------------
``kaon_geom_config``'s reason; the key that survives is::

    (file_key, entry, rec.slc..index, pfp_index)

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_shower_calo_config.py ...

DO NOT merge these products with the reco or shower products. They are joined, not
concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import make_file_df, make_shower_calo_default_df
from makedf.makedf import make_hdrdf

DFS = [make_shower_calo_default_df, make_hdrdf, make_file_df]
NAMES = ["shower_calo_default", "hdr", "file"]
