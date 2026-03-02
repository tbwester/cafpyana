#!/usr/bin/env python3

import numpy as np
import pandas as pd

import cafpyana.pyanalib.pandas_helpers as ph


_INDEX_COLS = [("__ntuple", "", "", "", "", ""), ("entry", "", "", "", "", "")] 
_LEVELS_RECO = ['__ntuple', 'entry', 'rec.slc..index']
_LEVELS_MC = ['__ntuple', 'entry', 'rec.mc.nu..index']


def df_index_size(df: pd.DataFrame, levels: list[int | str] | None=None) -> int:
    """
    Return number of rows of a data frame based on a list of multi index levels (ints or strings).
    Useful for calculating, e.g., number of slices even if a dataframe that has rows for each pfp
    """
    idx = df.index
    if levels is None:
        return idx.nunique()

    if not hasattr(idx, "levels"):
        return idx.nunique()

    idx_df = idx.to_frame(index=False)
    if not isinstance(levels, (list, tuple)):
        levels = [levels]

    # get all the columns by name or index
    cols = [idx_df.columns[l] if isinstance(l, int) else l for l in levels]

    return len(idx_df[cols].drop_duplicates())


def recodf_tmatch(recodf: pd.DataFrame) -> pd.DataFrame:
    """
    Split a recodf into truth match categories:
        - Has a truth match and is the highest efficiency in the event for that match
        - Has a truth match, but is not the highest efficiency in the event for
          that match (e.g., a true interaction that produced an extra slice)
        - Does not have a truth match
        """

    # first split slices into whether or not they have a truth match
    no_tmatch_mask = recodf.slc.tmatch.eff.isna()
    recodf_no_tmatch = recodf[no_tmatch_mask]
    recodf_tmatch = recodf[~no_tmatch_mask]

    # now get slices with max efficiency per event. Careful: could be multiple
    # true interactions so we include the tmatch index while grouping --  OK if
    # two slices in the event match to different true interactions
    best_tmatch_slices = recodf_tmatch.slc.tmatch.eff[
            recodf_tmatch.slc.tmatch.eff == (recodf_tmatch.slc.tmatch
            .groupby(['__ntuple', 'entry', 'idx']).eff.transform("max")
    )]

    best_tmatch_mask = recodf_tmatch.slc.tmatch.index.droplevel(-1).isin(best_tmatch_slices.index.droplevel(-1))
    recodf_best_tmatch = recodf_tmatch[best_tmatch_mask]
    recodf_bad_tmatch = recodf_tmatch[~best_tmatch_mask]

    assert df_index_size(recodf, _LEVELS_RECO) == (
            df_index_size(recodf_no_tmatch, _LEVELS_RECO) \
                    + df_index_size(recodf_best_tmatch, _LEVELS_RECO) \
                    + df_index_size(recodf_bad_tmatch, _LEVELS_RECO)
            )

    return recodf_best_tmatch, recodf_bad_tmatch, recodf_no_tmatch


def slice_purity(mcdf: pd.DataFrame, recodf: pd.DataFrame, true_type_col):
    """
    Compute the purity (true interaction content per slice) for a recodf + mcdf
    true_type_col: mcdf column containing true type labels
    """

    # get all slices & true interactions
    # drop duplicates to correctly count slices even if user has merged in other dfs (track, pfp, etc.)
    true_type_df = mcdf[[true_type_col]].groupby(level=[0, 1, 2]).first()
    slice_df = recodf[[("slc", "tmatch", "idx", "", "", "")]].groupby(level=[0, 1, 2]).first()

    # keep all slices, match to true interactions
    matchdf = ph.multicol_merge(slice_df.reset_index(), true_type_df.reset_index(),
                                left_on=_INDEX_COLS + [("slc", "tmatch", "idx", "", "", "")],
                                right_on=_INDEX_COLS + [("rec.mc.nu..index", "", "", "", "", "")],
                                how="left").set_index(_LEVELS_RECO).sort_index()
    
    # count all slices
    ntotal = df_index_size(matchdf, _LEVELS_RECO)
    result = {'total': ntotal}

    assert ntotal == df_index_size(recodf, _LEVELS_RECO)

    matchdf.loc[pd.isna(matchdf.true_type), true_type_col] = np.nan

    true_type_counts = (matchdf
        .reset_index()
        .drop_duplicates(_INDEX_COLS + [('rec.slc..index', '', '', '', '', '')])
        .groupby(true_type_col, dropna=False)
        .size()
    )

    for tt in matchdf[true_type_col].unique():
        try:
            result[tt] = true_type_counts.loc[tt]
        except KeyError:
            pass

    return result


def slice_efficiency(mcdf: pd.DataFrame, recodf: pd.DataFrame, true_type_col):
    """
    Compute the efficiency (slices / true interactions) for a recodf.
    True interactions are from a colum in MCdf
    """

    # get all slices & true interactions
    # drop duplicates to correctly count slices even if user has merged in other dfs (track, pfp, etc.)
    slice_df = recodf[[("slc", "tmatch", "idx", "", "", "")]].groupby(level=[0, 1, 2]).first()
    true_type_df = mcdf[[true_type_col]].groupby(level=[0, 1, 2]).first()

    # dict mapping types to counts
    true_type_counts = (true_type_df
        .groupby(true_type_col, dropna=False)
        .size()
    )

    matchdf = ph.multicol_merge(true_type_df.reset_index(), slice_df.reset_index(),
                                left_on=_INDEX_COLS + [("rec.mc.nu..index", "", "", "", "", "")],
                                right_on=_INDEX_COLS + [("slc", "tmatch", "idx", "", "", "")],
                                how="left").set_index(_LEVELS_MC).sort_index()


    assert df_index_size(matchdf, _LEVELS_MC) == df_index_size(mcdf, _LEVELS_MC)
    result = {}

    # regularize NaNs to numpy nan in case there are pandas NAs too
    matchdf.loc[pd.isna(matchdf[true_type_col]), true_type_col] = np.nan
    matchdf['valid_reco'] = ~pd.isna(matchdf.slc.tmatch.idx)

    true_type_pass = (matchdf
        .reset_index()
        .drop_duplicates(_INDEX_COLS + [('rec.mc.nu..index', '', '', '', '', '')])
        .groupby([true_type_col, 'valid_reco'], dropna=False)
        .size()
    )

    for tt in matchdf[true_type_col].unique():
        ntotal = true_type_counts.loc[tt]
        try:
            npass = true_type_pass.loc[(tt, True)]
        except KeyError:
            npass = 0
        result[(tt, 'pass')] = npass
        result[(tt, 'total')] = ntotal

    return result


def run_subrun_event(hdrdf: pd.DataFrame, df: pd.DataFrame):
    """Return run, subrun, and event number from a dataframe, squashing duplicates, e.g., from a track df)."""
    return hdrdf.loc[df.index.droplevel([
        l for l in df.index.names if l not in ['__ntuple', 'entry']
    ]).unique(), ['run', 'subrun', 'evt']].sort_index()


class CafpyanaAccumulator:
    """
    Class to keep running totals from multiple splits.
    """
    def __init__(self, func, mode='add', norm: float=1.0):
        self._func = func

        self._add_mode = (mode == 'add')
        self._is_dict = False
        self._counts = None
        self._norm = norm

    def add(self, *args):
        """
        Add call _func with args and add to counts
        If we are appending the result to a list, then assume there is no binning yet
        """

        # counts can be numpy array or single float, or dict of either
        counts = self._func(*args)

        # no counts yet, just set
        if self._counts is None:
            self._counts = counts
            self._is_dict = isinstance(self._counts, dict)
            return

        # simple histograms without multiple series
        if not self._is_dict:
            if self._add_mode:
                self._counts += counts
            else:
                self._counts = np.concatenate([self._counts, counts])
            return

        # hist with multiple series (dict returned by _func)
        for k, v in counts.items():
            try:
                # if we are not appending, counts are added element-wise per numpy
                if self._add_mode:
                    self._counts[k] += v
                else:
                    self._counts[k] = np.concatenate([self._counts[k], v])
            except KeyError:
                self._counts[k] = v

    def plot(self):
        """Normalize counts and return."""
        result = None
        if self._is_dict:
            # sort by key before returning
            result = {
                k: self._counts[k] * self._norm for k in sorted(self._counts.keys(),
                    key=lambda x: float('inf') if (pd.isna(x) or isinstance(x, str)) else x)
            }
        else:
            result = self._norm * self._counts
        return result
