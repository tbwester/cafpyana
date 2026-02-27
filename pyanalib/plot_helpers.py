#!/usr/bin/env python3

_LEVELS_RECO = ['__ntuple', 'entry', 'rec.slc..index']


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


def run_subrun_event(hdrdf: pd.DataFrame, df: pd.DataFrame):
    """Return run, subrun, and event number from a dataframe, squashing duplicates, e.g., from a track df)."""
    return hdrdf.loc[df.index.droplevel([
        l for l in df.index.names if l not in ['__ntuple', 'entry']
    ]).unique(), ['run', 'subrun', 'evt']].sort_index()


class CafpyanaHist:
    """
    Class to keep running total histogram from multiple df splits.
    If mode is not 'binned', values are added to a list for binning later.
    """
    def __init__(self, histfunc, mode='binned', bins=None, norm: float=1.0):
        self._bins = bins
        self._func = histfunc
        self._norm = norm

        self._binned_mode = (mode == 'binned')
        self._is_dict = False
        self._counts = None

    @property
    def norm(self) -> float:
        return self._norm

    @norm.setter
    def norm(self, val: float) -> None:
        self._norm = val

    def add(self, *args):
        """
        Add call _func with args and add to total hist.
        If we are appending the result to a list, then assume there is no binning yet
        """
        if self._binned_mode:
            # func should return bins and counts
            bins, counts = self._func(*args, bins=self._bins)
        else:
            # func only needs to return "counts" (values)
            counts = self._func(*args)

        # no counts yet, just set
        if self._counts is None:
            self._counts = counts
            self._is_dict = isinstance(self._counts, dict)
            return

        # simple histograms without multiple series
        if not self._is_dict:
            if self._binned_mode:
                self._counts += counts
            else:
                self._counts = np.concatenate([self._counts, counts])
            return

        # hist with multiple series (dict returned by _func)
        for k, v in counts.items():
            try:
                # if we are not appending, counts are added element-wise per numpy
                if self._binned_mode:
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
                    key=lambda x: float('inf') if pd.isna(x) else x)
            }
        else:
            result = self._norm * self._counts
        return self._bins, result
