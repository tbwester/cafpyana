"""
Kaon analysis data frame maker
"""

import pandas as pd

import makedf.makedf as makedf
import pyanalib.pandas_helpers as pd_helpers
import makedf.util as util


KPDG = {
    'kplus': makedf.PDG["kaon_p"][0],
    'kzero': makedf.PDG["kaon_0"][0]
}
KMASS = {
    'kplus': makedf.PDG["kaon_p"][2],
    'kzero': makedf.PDG["kaon_0"][2]
}

# TODO pick a better number
TRUE_KE_CUT = 0.


def make_kaon_mcdf(f: pd.DataFrame) -> pd.DataFrame:
    mcdf = makedf.make_mcdf(f)
    mcprimdf = makedf.make_mcprimdf(f)
    while mcprimdf.columns.nlevels > 2:
        mcprimdf.columns = mcprimdf.columns.droplevel(0)
    mcprimdf.index = mcprimdf.index.rename(mcdf.index.names[:2] + mcprimdf.index.names[2:])

    # add kaon info: Number above threshold & primary info
    for kname in ('kplus', 'kzero'):
        ke = mcprimdf[mcprimdf.pdg==KPDG[kname]].genE - KMASS[kname] 
        mcdf = pd_helpers.multicol_add(mcdf, ((mcprimdf.pdg==KPDG[kname]) \
                                              & (ke > TRUE_KE_CUT)).groupby(level=[0,1]).sum().rename(f'n{kname}'))
        kdf = mcprimdf[mcprimdf.pdg==KPDG[kname]].sort_values(mcprimdf.index.names[:2] + [("genE", "")]).groupby(level=[0,1]).last()
        kdf.columns = pd.MultiIndex.from_tuples([tuple([kname] + list(c)) for c in kdf.columns])
        mcdf = pd_helpers.multicol_merge(mcdf, kdf, left_index=True, right_index=True, how="left", validate="one_to_one")

    # drop things we don't need
    mask = (
        (mcdf.columns.get_level_values(0) != "plane")
        & (mcdf.columns.get_level_values(0) != "momentum")
        & (mcdf.columns.get_level_values(0) != "position")
        & (mcdf.columns.get_level_values(0) != "genie_evtrec_idx")
        & (mcdf.columns.get_level_values(0) != "start")
        & (mcdf.columns.get_level_values(0) != "end")
        & (mcdf.columns.get_level_values(0) != "genp")
        & (~mcdf.columns.get_level_values(0).isin([
            'baseline', 'time', 'bjorkenX', 'inelasticityY', 'Q2', 'w',
            'parent', 'parent_pdg'
        ]))
    )

    mcdf = mcdf.loc[:, mask]

    for c in mcdf.columns:
        print(c)
    return mcdf


# use this in configs
make_kaon_mcdf_truthcols = functools.partial(make_kaon_mcdf, signal_cut_columns=True)


def make_kaon_recodf(f: pd.DataFrame, save_track_truth: bool=False) -> pd.DataFrame:
    pandora_df = makedf.make_pandora_df(f, trkDistCut=0)

    # precuts
    pandora_df = pandora_df[InFV_SBND(pandora_df.slc.vertex)]
    pandora_df = pandora_df[pandora_df.slc.is_clear_cosmic == 0]
    # print(pandora_df[[('pfp', 'trk', 'chi2pid', 'I2', 'chi2_muon', ''), ('pfp', 'dist_to_vertex', '', '', '', '')]])

    # daughter info
    '''
    rec
                                                                                             slc
                                                                                            reco
                                                                                             pfp
                                                                                       daughters
        entry rec.slc..index rec.slc.reco.pfp..index rec.slc.reco.pfp.daughters..index
        0     0              1                       0                                         3
              1              1                       0                                         6
              2              3                       0                                        14
                                                     1                                        27
                                                     2                                        38
    daughter_to_pfp = daughterdf.reset_index().set_index(['entry', 'rec.slc..index', ('rec', 'slc', 'reco', 'pfp', 'daughters')])
    daughter_df = pandora_df[pandora_df.index.isin(daughter_to_pfp.index)]
    print(daughter_df.pfp)
    '''

    daughterdf = ph.loadbranches(f["recTree"], branches.pfp_daughter_branch)
    ndaughterdf = daughterdf.groupby(level=[0, 1, 2]).count()
    daughter_id = (
            daughterdf[
                daughterdf.index.isin(ndaughterdf[ndaughterdf.rec.slc.reco.pfp.daughters == 1].index)
            ].reset_index(level=-1)
    )[[('rec', 'slc', 'reco', 'pfp', 'daughters')]].droplevel([0, 1, 2], axis='columns')

    pandora_df = ph.multicol_merge(pandora_df, daughter_id, left_index=True, right_index=True, how='left')

    # kaon pid score
    kpid = ph.loadbranches(f["recTree"], ['rec.slc.reco.pfp.trk.chi2pid.2.chi2_kaon']).rec.slc.reco
    pandora_df = ph.multicol_merge(pandora_df, kpid, left_index=True, right_index=True, how='left')

    # barycenter flash match
    bcfm_df = ph.loadbranches(f["recTree"], branches.barycenterFMbranches)

    # surely there is a better way to do this?
    bcfm_score_df = bcfm_df[[('rec', 'slc', 'barycenterFM', 'score')]]
    pandora_df[("slc", "bcfm_score", "", "", "", "")] = (pandora_df.index
        .map(lambda idx: bcfm_score_df.loc[idx[:2],:].rec.slc.barycenterFM.score)
    )

    # drop things we don't need
    mask = (
        (
            (pandora_df.columns.get_level_values(0) == "pfp")
            & (
                pandora_df.columns.get_level_values(1).isin(
                ["trk", "trackScore", "dist_to_vertex", "t0", "daughters"]
            ) & (save_track_truth & (pandora_df.columns.get_level_values(2) == "truth")))
        )
        | (
            (pandora_df.columns.get_level_values(0) == "slc")
            & (
                pandora_df.columns.get_level_values(1).isin(
                ["tmatch", "vertex", "bcfm_score", "nu_score"]
            ))
        )
    )
    pandora_df = pandora_df.loc[:, mask]

    return pandora_df


make_kaon_recodf_save_track_truth = functools.partial(make_kaon_recodf, save_track_truth=True)
make_kaon_recodf_drop_track_truth = functools.partial(make_kaon_recodf, save_track_truth=False)


def make_kaon_mcdf_lite(f: pd.DataFrame) -> pd.DataFrame:
    """Bare-bones check for any k events."""
    df = ph.loadbranches(f["recTree"], ['rec.mc.nu.prim.pdg', 'rec.mc.nu.prim.genE']).rec.mc.nu.prim
    for kname in ('kplus', 'kzero'):
        ke = df[df.pdg==KPDG[kname]].genE - KMASS[kname] 
        df = ph.multicol_add(df, ((df.pdg==KPDG[kname]) \
                                              & (ke > TRUE_KE_CUT)).groupby(level=[0, 1]).sum().rename(f'n{kname}'))

    return (df[(df.nkplus > 0) | (df.nkzero > 0)]
            .drop(['pdg', 'genE'], axis=1)
            .droplevel(-1)
            .groupby(level=[0,1]).first()
    )
