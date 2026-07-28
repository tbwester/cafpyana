from analysis_village.kaon.makedf.make_kaon_df import (
    make_pair_df,
    make_slice_df,
    make_syst_df,
    make_track_df,
    make_true_type_df,
)
from makedf.makedf import make_hdrdf

DFS = [make_true_type_df, make_slice_df, make_syst_df, make_track_df, make_pair_df, make_hdrdf]
NAMES = ["true_type", "slice", "syst", "track", "pair", "hdr"]
