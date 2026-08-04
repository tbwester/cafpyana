"""Slim products for pair-BDT training.

kaon_config minus the two tables a training set cannot use:

  * syst      5465 columns and 84% of the bytes of a normal product. Training
              never reweights, so not one of them is read.
  * true_kaon per-kaon truth, superseded here by the per-track backtracked
              truth (truth_pdg / truth_G4ID / truth_parent). true_kaon is keyed
              on the interaction, so it cannot say which reconstructed track is
              which kaon, which is the question a *pair* label asks.

The saving is in the bytes, not the wall clock: dropping syst takes a product to
roughly a sixth of its size, and make_syst_df is ~17% of df-making runtime
(upstream finding 11), of which the largest single piece is building one
5465-column DataFrame.

THE CALORIMETRY VARIATIONS STAY, AND THIS IS THE POINT OF THE CONFIG.

The reason to retrain at all is that the model was trained on nominal chi2, so
the nominal is by construction the configuration on which its discrimination is
best -- every one of the eight +-1 sigma calorimetry variations therefore *loses*
signal, in both directions of every parameter, and the systematic is a one-sided
degradation rather than an uncertainty about a central value. The fix is to train
on varied chi2 so the nominal stops being privileged, which needs the 192 varied
chi2 columns on the pair table. Anything that drops them defeats the exercise.

What stays, and why each is needed to reproduce
pair_bdt/scripts/extract_nu_pairs.py's population and labels:

  pair       the features under the nominal calorimetry AND all eight
             variations, dist_to_parent_end, n_close_to_parent_end, and the
             per-leg backtracked truth
  track      NOT redundant: the pair table holds only the parent's end point
             and the daughter's start point, so both Euclidean lengths the model
             eats (track_len is the daughter's, ktrack_len the parent's) and the
             trackScore > 0.5 filter need this to join against
  slice      nu_score, barycenterFM_score and the reconstructed vertex
  true_type  the slice's true_type, which splits is_signal 1 from 2
  hdr        run/subrun/evt, for deduplication against other productions
  file       one row per input flatcaf, so a training pair can be traced back to
             the file it came from -- see make_file_df

histpotdf and histgenevtdf are appended by run_df_maker.py regardless.

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_train_config.py ...

DO NOT merge these products into an analysis df. merge_analysis_df.py validates
the syst column set but not track or true_type, so a mixed merge would
concatenate a NaN-filled union rather than fail.
"""

from analysis_village.kaon.makedf.make_kaon_df import (
    make_file_df,
    make_pair_df,
    make_slice_df,
    make_track_df,
    make_true_type_df,
)
from makedf.makedf import make_hdrdf

DFS = [make_true_type_df, make_slice_df, make_track_df, make_pair_df, make_hdrdf,
       make_file_df]
NAMES = ["true_type", "slice", "track", "pair", "hdr", "file"]
