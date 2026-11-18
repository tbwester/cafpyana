"""The calorimetry control sample's hits: every hit of its tracks, on all three planes.

The Noise Model and the calorimetry constants are fitted to hits, which the ``calo`` tier
collapses to twelve chi2 numbers per track, so without this tier the fits read the flatcafs.
The tracks are geomcalo's ``2trk`` and ``3trk`` control topologies (a long leg over 140 cm, a
short contained one in 10-50 cm, at most one other vertex primary in ``3trk``), selected in the
pass by ``make_kaon_df.select_control_tracks`` on top of the per-pfp row cut every tier has.

Two tables:

``control_track``
    One row per track of a passing slice: its leg (mu, p, x), length rank, topology, FV
    containment and length.  Small; it says which tracks the hits belong to.
``control_hit``
    Every hit of those tracks as ``make_trkhitdf`` returns it, raw (``dqdx``, not a calibrated
    dE/dx), float32.  ``hit`` restarts per plane, so a hit is keyed with ``plane``.  ``run`` is
    not stored: SBND's lifetime correction reads it, so a reader takes it from ``hdr``.
``control_seed``
    ``calo_seed`` per plane, the seed ``make_calo_df`` smears with.  It hashes the whole plane's
    hit table, which this tier does not hold, so a treatment applied to these hits reads it here.

Data products are NOT box-filtered: the signal box needs the BDT, so it is applied where the
tier is read.  Some 0.1% of MC control slices sit in the box (14% of the box itself).

Why ``hdr`` and ``file`` are here
--------------------------------
``kaon_geom_config``'s reason; the key that survives is::

    (file_key, entry, rec.slc..index, pfp_index)

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_hit_config.py ...

DO NOT merge these products with the reco or calo products.  They are joined, not concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import (
    make_control_hit_df,
    make_control_seed_df,
    make_control_track_df,
    make_file_df,
)
from makedf.makedf import make_hdrdf

DFS = [make_control_track_df, make_control_hit_df, make_control_seed_df, make_hdrdf, make_file_df]
NAMES = ["control_track", "control_hit", "control_seed", "hdr", "file"]

# kaon_format 5 products are blosc:zstd (run_df_maker reads these two names).
COMPLIB, COMPLEVEL = "blosc:zstd", 4
