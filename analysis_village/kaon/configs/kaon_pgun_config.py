from makedf.makedf import make_hdrdf
from analysis_village.kaon.makedf.legacy_make_kaon_df import (
    make_kaon_mcdf_truthcols,
    make_kaon_recodf_save_track_truth,
)

DFS = [make_kaon_mcdf_truthcols, make_kaon_recodf_save_track_truth, make_hdrdf]
NAMES = ["kmc", "kreco", "hdr"]
