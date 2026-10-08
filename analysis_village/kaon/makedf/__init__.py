"""Kaon df makers.

Deliberately empty of imports.  Re-exported names would make importing *anything*
under it -- including a leaf module with no dependencies -- pull in ``make_kaon_df``
and therefore ``makedf.chi2pid``, which opens its dE/dx templates from ``/cvmfs`` at
module scope.  Off the grid that is an ImportError before any function is reached.

Every config imports from the module that defines the name.
"""
