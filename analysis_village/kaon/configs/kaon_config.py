"""Superseded: use ``kaon_reco_config`` and ``kaon_syst_config``.

This was the single-file analysis production -- reco tables and reweight
systematics in one product.  It is deliberately no longer runnable, because
``make_syst_df`` now returns **one row per true interaction** rather than one per
slice, and a product mixing the two granularities cannot be read safely:
``format="fixed"`` does not store index level names, so
``kaonana.analysis.tables.read_syst`` names them itself and would label an
interaction index as a slice index without complaint.

Rather than leave that trap runnable, the two passes are separate configs:

    kaon_reco_config    everything reconstruction, including all eight
                        calorimetry variations.  Rebuilt whenever reco changes.
    kaon_syst_config    the reweights alone, plus hdr and file for the join key.
                        Depends on no part of reconstruction, so it is produced
                        once and reused.

They are joined on ``(file_key, run, subrun, evt, rec.mc.nu..index)``, not
concatenated.
"""

raise ImportError(
    "kaon_config is superseded by kaon_reco_config + kaon_syst_config; "
    "make_syst_df is now per-interaction and cannot share a product with the "
    "per-slice reco tables. See this file's docstring."
)
