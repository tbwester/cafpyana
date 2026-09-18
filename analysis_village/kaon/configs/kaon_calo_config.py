"""A fourth pass: the calorimetry chi2 universes, two TREATMENTS side by side.

Split out of ``kaon_reco_config`` at kaonana CALO.199, for two reasons that are not the obvious
one.

Why the split
-------------
**A variation is a delta about a chosen central value.**  ``ccal_p`` under the uncorrected chain
and ``ccal_p`` under the joint chain are different numbers answering different questions, and one
``cv`` per product cannot hold two chains side by side -- which is exactly what comparing
calorimetry treatments needs.  A *treatment* is a central value plus the universes that are deltas
about it, and ``chi2pid.SBND_CALO_TREATMENTS`` is the registry.

**And the columns were 81% of a reco product**, measured: ``track`` carried 108 suffixed chi2
columns and ``make_pair_df`` duplicated 240 more onto its two legs, so a kcv product was 26 MB of
which 21 MB was chi2.  Dropping them takes ``track`` from 141 columns to 33 and ``pair`` from 272
to 56.

What it does NOT buy is much time.  Measured on one kcv flatcaf in one environment, the calo block
is roughly **half to two-thirds** of a reco pass -- reading hits is the expensive part, and a
calo-only pass still reads them.  On the grid it is worse, because a calo-only pass transfers the
same flatcaf.  Justify this split on bookkeeping, not on speed.

Two treatments
--------------
``calo_default``
    No chain -- gains, lifetime, YZ, then the recombination inversion.  The uncorrected baseline
    a treatment comparison is measured against.  Nine universes.

    **It is NOT LArSoft's own number and must not be read as a check on it.**  The chain
    recomputes dE/dx with this repo's gains and inverts on MC's ModBox for *both* samples
    (CALO.86), so ``calo_default``'s ``cv`` sits near 0.49 of the CAF's proton chi2 and 0.71 of
    its kaon chi2 -- measured, and the same structure the old products showed at 0.4591 and
    0.6602 with the joint chain.  RUNLOG's 2.1e-15 was *production against production* with the
    constants unchanged, which is a different comparison.
``calo_joint``
    The shipped joint-shaped chain (CALO.197), with the eight recombination universes and eight
    joint ones.  Seventeen universes, including **both readings of the smear amplitude's
    uncertainty** -- ``jointampboot`` and ``jointampnu`` -- because which one to book is a
    coverage question and the study needs both produced.  Exactly one may enter a covariance;
    ``kaonana.params.variations.active_pairs`` is where that is chosen.

Both treatments read the hits ONCE.  ``make_calo_df`` caches them per file, which is safe because
``ntuples.dataframes`` yields one tuple of frames per file, so every builder in ``DFS`` runs on a
given file before the next is opened.  Two passes would pay the hit read twice.

On data every universe collapses onto ``cv`` -- the chain is MC-only -- so a data product carries
twelve columns rather than 204 instead of 192 columns of identical zeros.

Why ``hdr`` and ``file`` are here
--------------------------------
``kaon_geom_config``'s reason, and it applies identically: ``__ntuple`` is assigned from the order
a job read its inputs, so this pass and a reco pass number the same flatcafs differently and an
index join across the two is wrong while looking right.  The key that survives is::

    (file_key, entry, rec.slc..index, pfp_index)

``kaonana.schema.CALO_TABLES`` declares it and ``kaonana.data.samples.event_key_frame`` builds it.

ONE CONSTRAINT ON RE-RUNNING THIS.  ``calo_seed`` is derived from the hit table's own track index,
so a separate pass reproduces the draw -- that is what makes this tier sound.  It does **not**
survive a different pandas major version, so a universe produced in one environment must never be
differenced against a central value produced in another (CALO.198).  Produce a treatment set in
one pass, in one container.

    python run_df_maker.py -c analysis_village/kaon/configs/kaon_calo_config.py ...

DO NOT merge these products with the reco products.  They are joined, not concatenated.
"""

from analysis_village.kaon.makedf.make_kaon_df import (
    make_calo_default_df,
    make_calo_joint_df,
    make_file_df,
)
from makedf.makedf import make_hdrdf

DFS = [make_calo_default_df, make_calo_joint_df, make_hdrdf, make_file_df]
NAMES = ["calo_default", "calo_joint", "hdr", "file"]
