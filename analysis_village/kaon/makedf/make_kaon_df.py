"""
Kaon analysis data frame maker
"""
import functools

import numpy as np
import pandas as pd

import makedf.makedf as makedf
from makedf import branches
import pyanalib.pandas_helpers as ph
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

# For daughter df merging: This ensures we use the equivalent mc.nu.prim branch
# names (SRTrueParticle) as the daughter branches
PRIM_BRANCHES = list(set(b.replace('.true_particles.', '.mc.nu.prim.') for b in branches.trueparticlebranches))


# InFV requires inzback but it does nothing for SBND case
# put NaN here so we'll hopefully get an error if this ever changes
InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')
InAV_SBND = functools.partial(util.InAV, det='SBND')


def k_has_daughter(tpartdf: pd.DataFrame, ktype: str, require_contained: bool=True) -> pd.MultiIndex:
    """
    Signal identification using backward hierarchy traversal.
    Starts from daughters (mu+, pi+) and traces parentage up to a primary Kaon.
    Handles arbitrary scattering depths (K+ -> K+ -> K+ ... -> mu+).
    """
    if tpartdf.empty:
        return pd.MultiIndex(levels=[[], []], codes=[[], []], names=[tpartdf.index.names[0], 'interaction_id'])

    # 1. Simplify tpartdf for traversal
    df = tpartdf.copy()
    df.columns = ['.'.join(n for n in c if n != '') for c in df.columns]
    df = df.reset_index()

    # 2. Identify potential signal daughters (mu+, pi+)
    target_pdgs = [-13, 211] if ktype == 'kplus' else []
    # We track the "active" particles being traced upwards
    active = df[df.pdg.isin(target_pdgs)].copy()
    
    if active.empty:
        return pd.MultiIndex(levels=[[], []], codes=[[], []], names=[tpartdf.index.names[0], 'interaction_id'])


    # Initial daughter-level containment
    if require_contained:
        active = active[active.cont_tpc == 1]
        # MIP must also stop in active volume
        dtr_end = active[['end.x', 'end.y', 'end.z']].rename(columns=lambda x: x.split('.')[-1])
        active = active[InAV_SBND(dtr_end)]
    
    if active.empty:
        return pd.MultiIndex(levels=[[], []], codes=[[], []], names=[tpartdf.index.names[0], 'interaction_id'])

    # This will store success (entry, interaction_id)
    signal_interactions = []
    k_pdg = KPDG[ktype]

    # 3. Trace backwards generation by generation
    while not active.empty:
        # Get parents of active particles
        parents = active.merge(
            df[['entry', 'G4ID', 'pdg', 'parent', 'cont_tpc']].rename(columns={
                'G4ID': 'G4ID_p',
                'pdg': 'pdg_p',
                'parent': 'parent_p',
                'cont_tpc': 'cont_tpc_p'
            }),
            left_on=['entry', 'parent'],
            right_on=['entry', 'G4ID_p']
        )
        
        if parents.empty:
            break

        # Is the parent a Kaon?
        is_k_parent = (parents.pdg_p == k_pdg)
        
        # If Kaon parent and it is primary (parent 10000000), we found a signal chain!
        is_primary_k = is_k_parent & (parents.parent_p == 10000000)
        
        # All particles in the chain must be contained
        if require_contained:
            is_primary_k &= (parents.cont_tpc_p == 1)
            
        successes = parents[is_primary_k]
        if not successes.empty:
            signal_interactions.append(successes[['entry', 'interaction_id']])
        
        # Continue tracing for those whose parent is a Kaon but not yet primary
        active = parents[is_k_parent & ~is_primary_k].copy()
        if not active.empty:
            # Shift generation: the parent becomes the active particle
            active['G4ID'] = active['G4ID_p']
            active['parent'] = active['parent_p']
            active['cont_tpc'] = active['cont_tpc_p']
            active = active[['entry', 'G4ID', 'parent', 'cont_tpc', 'interaction_id']]
            if require_contained:
                active = active[active.cont_tpc == 1]

    if not signal_interactions:
        return pd.MultiIndex(levels=[[], []], codes=[[], []], names=[tpartdf.index.names[0], 'interaction_id'])

    # 4. Consolidate results
    res_df = pd.concat(signal_interactions).drop_duplicates()
    return pd.MultiIndex.from_frame(res_df)


def signal(mcdf: pd.DataFrame, signal_idx: pd.MultiIndex, cc: bool=True) -> pd.Series:
    """
    Signal definition for mcdf.
    Uses pre-computed hierarchy-traced signal indices.
    """
    cc_nc = (mcdf.iscc == cc)
    # Match the interaction-level index (entry, nu_idx)
    return mcdf.index.droplevel(list(range(2, mcdf.index.nlevels))).isin(signal_idx) & mcdf.is_true_fv & cc_nc


def make_kaon_mcdf(f: pd.DataFrame, signal_cut_columns: bool=False) -> pd.DataFrame:
    mcdf = ph.loadbranches(f["recTree"], branches.mcbranches).rec.mc.nu
    # prevent name clash with mcprim pdg column
    mcdf.columns = [
        ('nu_pdg', '') if col == ('pdg', '') else col
        for col in mcdf.columns
    ]

    mcprimdf = ph.loadbranches(f["recTree"], PRIM_BRANCHES).rec.mc.nu.prim
    mcprimdf['is_primary'] = True

    # add number of primaries above threshold
    for kname in ('kplus', 'kzero'):
        # number of kaons above KE threshold
        ke = mcprimdf[mcprimdf.pdg==KPDG[kname]].genE - KMASS[kname] 
        mcdf = ph.multicol_add(mcdf, ((mcprimdf.pdg==KPDG[kname]) \
                                              & (ke > TRUE_KE_CUT)).groupby(level=[0, 1]).sum().rename(f'n{kname}'))

    # daughter info
    tpartdf = ph.loadbranches(f["recTree"], branches.trueparticlebranches).rec.true_particles
    tpartdf = tpartdf.reset_index().set_index(['entry', 'G4ID'])

    # Compute signal indices using hierarchy traversal
    kp_signal_idx = k_has_daughter(tpartdf, 'kplus', require_contained=True)
    kp_signal_idx_nocont = k_has_daughter(tpartdf, 'kplus', require_contained=False)

    mcprimdaughtersdf = makedf.make_mcprimdaughtersdf(f).rec.mc.nu.prim
    daughterdf = mcprimdaughtersdf[mcprimdaughtersdf.index.droplevel(-1).isin(mcprimdf.index)]
    daughter_tpartdf = tpartdf[
        tpartdf.index.isin(pd.MultiIndex.from_frame(daughterdf.reset_index()[['entry', 'daughters']]))
    ]
    daughterdf = ph.multicol_merge(daughterdf, daughter_tpartdf, how="left", left_on=['entry', 'daughters'], right_index=True)
    daughterdf['is_primary'] = False
    daughterdf = daughterdf.drop(columns=[('rec.true_particles..index', '', '', '')])

    # .rename doesn't work for me, so do this instead
    daughterdf.columns = [
        ('G4ID', '', '', '') if col == ('daughters', '', '', '') else col
        for col in daughterdf.columns
    ]

    # add daughter index to primaries as "-1" to allow concat
    mcprimdf["rec.mc.nu.prim.daughters..index"] = -1
    mcprimdf = mcprimdf.set_index("rec.mc.nu.prim.daughters..index", append=True)
    mcprimdf = pd.concat([mcprimdf, daughterdf], sort=True).sort_index()

    # finally, merge primaries into mc
    mcdf = ph.multicol_merge(mcdf, mcprimdf, how="left", left_index=True, right_index=True, validate="one_to_one")

    mcdf['is_true_fv'] = InFV_SBND(mcdf.position)
    mcdf['is_signal_kp_cc'] = signal(mcdf, kp_signal_idx, cc=True)
    mcdf['is_signal_kp_nc'] = signal(mcdf, kp_signal_idx, cc=False)

    # extra columns for truth studies
    if signal_cut_columns:
        mcdf['has_daughter'] = mcdf.index.droplevel(list(range(2, mcdf.index.nlevels))).isin(kp_signal_idx_nocont)
        mcdf['has_daughter_cont'] = mcdf.index.droplevel(list(range(2, mcdf.index.nlevels))).isin(kp_signal_idx)

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


    # drop daughters after selection
    # mcdf = mcdf.xs(-1, level=3, drop_level=False)

    return mcdf


# use this in configs
make_kaon_mcdf_truthcols = functools.partial(make_kaon_mcdf, signal_cut_columns=True)


def make_kaon_recodf(f: pd.DataFrame, save_track_truth: bool=False) -> pd.DataFrame:
    pandora_df = makedf.make_pandora_df(f, trkDistCut=0)

    # precuts
    pandora_df = pandora_df[InFV_SBND(pandora_df.slc.vertex)]
    pandora_df = pandora_df[pandora_df.slc.is_clear_cosmic == 0]

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

    # kaon and other pid scores not saved by default
    kpid = ph.loadbranches(f["recTree"], [
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_pion',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_pion',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_pion',
    ]).rec.slc.reco
    pandora_df = ph.multicol_merge(pandora_df, kpid, left_index=True, right_index=True, how='left')

    # kaon momentum estimators
    kmom = ph.loadbranches(f["recTree"], [
        'rec.slc.reco.pfp.trk.mcsP.fwdP_kaon',
    ]).rec.slc.reco
    pandora_df = ph.multicol_merge(pandora_df, kmom, left_index=True, right_index=True, how='left')

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
            ) | (save_track_truth & (pandora_df.columns.get_level_values(2) == "truth")))
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
