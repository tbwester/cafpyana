from makedf.makedf import make_hdrdf
from analysis_village.kaon.makedf import make_kaon_mcdf, make_kaon_recodf_drop_track_truth

DFS = [make_kaon_mcdf, make_kaon_recodf_drop_track_truth, make_hdrdf]
NAMES = ["kmc", "kreco", "hdr"]
