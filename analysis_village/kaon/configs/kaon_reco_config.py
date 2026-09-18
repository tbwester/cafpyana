"""Pass 1 of the two-file production: everything reconstruction, no reweights.

``kaon_config`` minus ``make_syst_df``.  The GENIE and flux reweights are pure
truth, so they go to ``kaon_syst_config`` and are produced once.

**The calorimetry variations no longer stay** (kaonana CALO.199).  They used to,
on the argument that they are *recomputed reco* and so belong with the track they
describe -- which is true and turned out not to be the deciding consideration.
They are deltas about a chosen central value, so carrying one ``cv`` per product
makes two calorimetry treatments impossible to hold side by side; and they were
**81% of a product's bytes**, 108 columns on ``track`` and 240 duplicated onto
``pair``'s two legs.  They are now ``kaon_calo_config``, joined on
``(file_key, entry, rec.slc..index, pfp_index)``.  What stays here is the twelve
BARE ``chi2_{hyp}_I{plane}`` columns LArSoft itself wrote, which are the
regression check that the chain reproduces the CAF.

Not ``kaon_train_config``, which also drops ``syst``: that one additionally drops
``true_kaon``, which the analysis export carries, and warns against merging its
products into an analysis df because ``merge_analysis_df`` validates the syst
column set but not ``track`` or ``true_type``, so a mixed merge would silently
concatenate a NaN-filled union.  This config is the analysis one.

``syst`` was ~84% of a normal product's bytes, so this is the file day-to-day work
touches: training, the central value, the calorimetry systematic, every
selection scan.

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_reco_config.py ...
"""

from analysis_village.kaon.makedf.make_kaon_df import (
    make_file_df,
    make_pair_df,
    make_slice_df,
    make_track_df,
    make_true_kaon_df,
    make_true_type_df,
)
from makedf.makedf import make_hdrdf

DFS = [make_true_type_df, make_true_kaon_df, make_slice_df,
       make_track_df, make_pair_df, make_hdrdf, make_file_df]
NAMES = ["true_type", "true_kaon", "slice", "track", "pair", "hdr", "file"]
