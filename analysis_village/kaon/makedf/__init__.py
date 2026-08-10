"""Kaon df makers.

Deliberately empty of imports.  This package used to re-export ten names from
``legacy_make_kaon_df`` and ``make_kaon_df``, which meant that importing
*anything* under it -- including a leaf module with no dependencies -- pulled in
``make_kaon_df`` and therefore ``makedf.chi2pid``, which opens its dE/dx
templates from ``/cvmfs`` at module scope.  Off the grid that is an ImportError
before any function is reached, so ``kaonana`` could not share so much as a hash
function with the writer and kept its own copy.

The two live configs (``kaon_config``, ``kaon_train_config``) always imported
from ``.make_kaon_df`` directly and never used the re-exports at all; the four
legacy configs now do the same against ``.legacy_make_kaon_df``.  Import from
the module that defines the name.
"""
