import functools

import cafpyana.makedf.getsyst as getsyst
import cafpyana.makedf.util as util
import cafpyana.pyanalib.pandas_helpers as ph
import numpy as np
import pandas as pd

KPDG = {'kplus': 321, 'kzero': 311}
KMASS = {'kplus': 0.493677, 'kzero': 0.497611}
TRUE_KE_CUT = 0.

InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')

def k_has_daughter(tpartdf: pd.DataFrame, ktype: str, require_contained: bool=True) -> pd.Series:
    """
    Signal identification using backward hierarchy traversal.
    Returns a Series mapping (entry, interaction_id) -> daughter_pdg.
    Handles arbitrary scattering depths and preserves the MIP type.
    """
    nulldf = pd.Series(dtype=float, name='k_daughter_pdg')
    if tpartdf.empty:
        return nulldf

    df = tpartdf.copy()
    df.columns = ['.'.join(n for n in c if n != '') for c in df.columns]
    df = df.reset_index()

    dtr_counts = df.groupby(['entry', 'parent']).size().rename('ndaughters')
    df = df.merge(dtr_counts, left_on=['entry', 'G4ID'], right_index=True, how='left')
    df['ndaughters'] = df['ndaughters'].fillna(0)

    target_pdgs = [-13, 211] if ktype == 'kplus' else []
    active = df[df.pdg.isin(target_pdgs)].copy()
    active['daughter_pdg'] = active['pdg']

    if active.empty:
        return nulldf

    if require_contained:
        dtr_start = active[['start.x', 'start.y', 'start.z']].rename(columns=lambda x: x.split('.')[-1])
        dtr_end = active[['end.x', 'end.y', 'end.z']].rename(columns=lambda x: x.split('.')[-1])
        active = active[InFV_SBND(dtr_start) & InFV_SBND(dtr_end)]

    if active.empty:
        return nulldf

    signal_results = []
    k_pdg = KPDG[ktype]
    is_first_step = True

    while not active.empty:
        parents = active.merge(
            df[['entry', 'G4ID', 'pdg', 'parent', 'ndaughters',
                'start.x', 'start.y', 'start.z', 'end.x', 'end.y', 'end.z']].rename(columns={
                'G4ID': 'G4ID_p',
                'pdg': 'pdg_p',
                'parent': 'parent_p',
                'ndaughters': 'ndaughters_p',
                'start.x': 'start.x_p', 'start.y': 'start.y_p', 'start.z': 'start.z_p',
                'end.x': 'end.x_p', 'end.y': 'end.y_p', 'end.z': 'end.z_p'
            }),
            left_on=['entry', 'parent'],
            right_on=['entry', 'G4ID_p']
        )

        if parents.empty:
            break

        is_k_parent = (parents.pdg_p == k_pdg)

        if is_first_step:
            is_k_parent &= (parents.ndaughters_p <= 2)
            is_first_step = False

        is_primary_k = is_k_parent & (parents.parent_p == 10000000)

        if require_contained:
            p_start = parents[['start.x_p', 'start.y_p', 'start.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
            p_end = parents[['end.x_p', 'end.y_p', 'end.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
            is_primary_k &= (InFV_SBND(p_start) & InFV_SBND(p_end))

        successes = parents[is_primary_k]
        if not successes.empty:
            signal_results.append(successes[['entry', 'interaction_id', 'daughter_pdg']])

        active = parents[is_k_parent & ~is_primary_k].copy()
        if not active.empty:
            if require_contained:
                p_start = active[['start.x_p', 'start.y_p', 'start.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
                p_end = active[['end.x_p', 'end.y_p', 'end.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
                active = active[InFV_SBND(p_start) & InFV_SBND(p_end)]

            active['G4ID'] = active['G4ID_p']
            active['parent'] = active['parent_p']
            active = active[['entry', 'G4ID', 'parent', 'interaction_id', 'daughter_pdg']]

    if not signal_results:
        return nulldf

    res_df = pd.concat(signal_results).drop_duplicates()

    counts = res_df.groupby(['entry', 'interaction_id']).daughter_pdg.nunique()
    unambiguous = counts[counts == 1].index

    final_pdgs = res_df.set_index(['entry', 'interaction_id']).loc[unambiguous]
    return final_pdgs.groupby(level=[0, 1]).first().daughter_pdg


def make_true_type_df(f) -> pd.DataFrame:
    """
    Creates a minimal dataframe with exactly 1 row per true neutrino interaction.
    """
    # 1. Load basic neutrino info
    nu_branches = ['rec.mc.nu.E', 'rec.mc.nu.iscc', 'rec.mc.nu.position.x', 'rec.mc.nu.position.y', 'rec.mc.nu.position.z']
    mcdf = ph.loadbranches(f["recTree"], nu_branches).rec.mc.nu

    is_cc = mcdf['iscc'] == 1
    true_E_nu = mcdf['E']
    is_true_fv = InFV_SBND(mcdf['position'])

    # 2. Load primary info (specifically for K+)
    prim_branches = [
        'rec.mc.nu.prim.pdg', 'rec.mc.nu.prim.genE',
        'rec.mc.nu.prim.end.x', 'rec.mc.nu.prim.end.y', 'rec.mc.nu.prim.end.z'
    ]
    mcprimdf = ph.loadbranches(f["recTree"], prim_branches).rec.mc.nu.prim

    kplus_mask = (mcprimdf.pdg == KPDG['kplus'])
    kplus_df = mcprimdf[kplus_mask].copy()

    ke = kplus_df.genE - KMASS['kplus']
    nkplus = (ke > TRUE_KE_CUT).groupby(level=[0, 1]).sum()

    kplus_first = kplus_df.groupby(level=[0, 1]).first()
    true_E_kaon = kplus_first['genE']
    true_P_kaon = np.sqrt(np.maximum(0, true_E_kaon**2 - KMASS['kplus']**2)).fillna(0)

    kplus_end = kplus_first['end'].copy()
    kplus_end.columns = [c[0] for c in kplus_end.columns]
    is_k_contained = InFV_SBND(kplus_end)
    is_k_contained = is_k_contained.fillna(False).astype(bool)

    # 3. Daughter association (hierarchy traversal)
    tpart_branches = [
        'rec.true_particles.pdg',
        'rec.true_particles.parent',
        'rec.true_particles.interaction_id',
        'rec.true_particles.G4ID',
        'rec.true_particles.start.x',
        'rec.true_particles.start.y',
        'rec.true_particles.start.z',
        'rec.true_particles.end.x',
        'rec.true_particles.end.y',
        'rec.true_particles.end.z',
    ]
    tpartdf = ph.loadbranches(f["recTree"], tpart_branches).rec.true_particles
    tpartdf = tpartdf.reset_index().set_index(['entry', 'G4ID'])

    k_daughter_pdg = k_has_daughter(tpartdf, 'kplus', require_contained=True)

    # 4. Consolidate into our minimal dataframe
    res = pd.DataFrame(index=mcdf.index)
    res['is_cc'] = is_cc.fillna(False).astype(bool)
    res['is_true_fv'] = is_true_fv.fillna(False).astype(bool)
    res['true_E_nu'] = true_E_nu

    res['nkplus'] = nkplus
    res['nkplus'] = res['nkplus'].fillna(0).astype(int)

    res['true_E_kaon'] = true_E_kaon
    res['true_E_kaon'] = res['true_E_kaon'].fillna(0)

    res['true_P_kaon'] = true_P_kaon
    res['true_P_kaon'] = res['true_P_kaon'].fillna(0)

    res['is_k_contained'] = is_k_contained
    res['is_k_contained'] = res['is_k_contained'].fillna(False).astype(bool)

    res['k_daughter_pdg'] = k_daughter_pdg

    # 5. Compute 'true_type' (Standard 1, 2, 3, 4, 98, 99)
    true_type = pd.Series(np.nan, index=res.index)

    true_type[~res['is_true_fv']] = 99
    true_type[res['is_true_fv'] & (res['nkplus'] < 1)] = 98

    is_k_fv = res['is_true_fv'] & (res['nkplus'] > 0)
    true_type[is_k_fv & (res['k_daughter_pdg'] == -13)] = 1
    true_type[is_k_fv & (res['k_daughter_pdg'] == 211)] = 2
    true_type[is_k_fv & res['is_k_contained'] & true_type.isna()] = 3
    true_type[is_k_fv & ~res['is_k_contained'] & true_type.isna()] = 4

    res['true_type'] = true_type.astype(int)

    # Return exactly the 6 columns we need downstream!
    keep_cols = ['true_type', 'is_cc', 'is_k_contained', 'true_E_nu', 'true_E_kaon', 'true_P_kaon']
    return res[keep_cols]




def make_slice_df(f: dict) -> pd.DataFrame:
    """
    Extracts a lightweight dataframe containing exactly 1 row per slice.
    Used downstream to keep track of the 100% pre-cut denominator for efficiency
    and applying basic slice-level pre-cuts. Systematic weights are merged in.
    """
    branches_to_load = [
        'rec.slc.nu_score',
        'rec.slc.is_clear_cosmic',
        'rec.slc.barycenterFM.score',
        'rec.slc.tmatch.index'
    ]

    # Load slice branches
    slc_df = ph.loadbranches(f["recTree"], branches_to_load).rec.slc

    # Simplify the column names from the deeply nested cafpyana tuples
    if isinstance(slc_df.columns, pd.MultiIndex):
        slc_df.columns = ["_".join([str(c) for c in col if c]).strip() for col in slc_df.columns.values]

    return slc_df

def make_syst_df(f: dict) -> pd.DataFrame:
    """
    Extracts systematic weights only for slices that pass basic pre-cuts.
    This saves massive amounts of memory compared to saving weights for all slices.
    """
    branches_to_load = [
        'rec.slc.is_clear_cosmic',
        'rec.slc.tmatch.index'
    ]

    # Load slice branches
    slc_df = ph.loadbranches(f["recTree"], branches_to_load).rec.slc
    if isinstance(slc_df.columns, pd.MultiIndex):
        slc_df.columns = ["_".join([str(c) for c in col if c]).strip() for col in slc_df.columns.values]

    # PRE-CUT 1: Only keep slices that are not clear cosmics
    slc_df = slc_df[slc_df.is_clear_cosmic == 0]

    # Get systematic weights, limiting multisims to 100 universes
    systs = getsyst.get_all_syst_df(f, multisim_nuniv=100)

    if systs is None or systs.empty:
        return pd.DataFrame(index=slc_df.index)

    # Flatten MultiIndex columns of systs if they are tuples
    if isinstance(systs.columns, pd.MultiIndex):
        systs.columns = ["_".join([str(c) for c in col if c]).strip() for col in systs.columns.values]

    # Drop any slices that didn't match to a true neutrino to get the matching indices
    valid_matches = slc_df['tmatch_idx'].dropna().astype(int)

    # Use cafpyana's highly optimized reindexing function to pull the weights
    if not valid_matches.empty:
        syst_df = getsyst.filter_systs_nuind(f, systs, valid_matches)
    else:
        # Create an empty dataframe with correct columns if no valid matches
        syst_df = pd.DataFrame(columns=systs.columns, index=valid_matches.index)

    # Slices without a valid truth match (NaN tmatch_idx) were ignored by the filter.
    # We reindex to include them and fill their weights with 1.0 (nominal).
    syst_df = syst_df.reindex(slc_df.index, fill_value=1.0)

    return syst_df

def _extract_base_track_df(f: dict):
    """
    Helper function that extracts single-track features and computes
    all systematic calorimetry variations.
    """
    branches_to_load = [
        'rec.slc.reco.pfp.trk.len',
        'rec.slc.reco.pfp.trk.start.x',
        'rec.slc.reco.pfp.trk.start.y',
        'rec.slc.reco.pfp.trk.start.z',
        'rec.slc.reco.pfp.trk.end.x',
        'rec.slc.reco.pfp.trk.end.y',
        'rec.slc.reco.pfp.trk.end.z',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
    ]

    pandora_df = ph.loadbranches(f["recTree"], branches_to_load).rec.slc.reco
    slice_idx_names = list(pandora_df.index.names)[:-1]
    pfp_idx_col = pandora_df.index.names[-1]

    flat_df = pandora_df.loc[:, ~pandora_df.columns.duplicated()].reset_index(level=pfp_idx_col)

    # --- CALO VARIATIONS (Dummy Implementation) ---
    variations = {
        "ccal_p": 1.02, "ccal_m": 0.98,
        "alpha_p": 1.05, "alpha_m": 0.95,
        "beta_p": 1.03, "beta_m": 0.97,
        "R_p": 1.04, "R_m": 0.96,
    }

    for var_name, dummy_factor in variations.items():
        flat_df[("pfp", "trk", "chi2pid", "I2", f"chi2_muon_{var_name}")] = flat_df[("pfp", "trk", "chi2pid", "I2", "chi2_muon")] * dummy_factor
        flat_df[("pfp", "trk", "chi2pid", "I2", f"chi2_proton_{var_name}")] = flat_df[("pfp", "trk", "chi2pid", "I2", "chi2_proton")] * dummy_factor
        flat_df[("pfp", "trk", "chi2pid", "I2", f"chi2_kaon_{var_name}")] = flat_df[("pfp", "trk", "chi2pid", "I2", "chi2_kaon")] * dummy_factor
    # ----------------------------------------------

    # Rename complex awkward tuples to flat sensible names
    pfp_col_key = (pfp_idx_col, '', '', '', '')
    col_map = {
        ("pfp", "trk", "start", "x", ""): "start_x",
        ("pfp", "trk", "start", "y", ""): "start_y",
        ("pfp", "trk", "start", "z", ""): "start_z",
        ("pfp", "trk", "end", "x", ""): "end_x",
        ("pfp", "trk", "end", "y", ""): "end_y",
        ("pfp", "trk", "end", "z", ""): "end_z",
        ("pfp", "trk", "len", "", ""): "trk_len",
        ("pfp", "trk", "chi2pid", "I2", "chi2_kaon"): "chi2_kaon",
        ("pfp", "trk", "chi2pid", "I2", "chi2_muon"): "chi2_muon",
        ("pfp", "trk", "chi2pid", "I2", "chi2_proton"): "chi2_proton",
        pfp_col_key: "pfp_index",
    }

    for var in variations.keys():
        col_map[("pfp", "trk", "chi2pid", "I2", f"chi2_muon_{var}")] = f"chi2_muon_{var}"
        col_map[("pfp", "trk", "chi2pid", "I2", f"chi2_proton_{var}")] = f"chi2_proton_{var}"
        col_map[("pfp", "trk", "chi2pid", "I2", f"chi2_kaon_{var}")] = f"chi2_kaon_{var}"

    cols_to_keep = {k: v for k, v in col_map.items() if k in flat_df.columns}
    clean_df = flat_df[list(cols_to_keep.keys())].copy()
    clean_df.columns = list(cols_to_keep.values())

    # Restore index: entry and slice index are already in the index, so just append pfp_index
    clean_df = clean_df.set_index("pfp_index", append=True).sort_index()

    return clean_df, slice_idx_names


def make_track_df(f: dict) -> pd.DataFrame:
    """
    Extracts Single-Track features for the Track BDT evaluation.
    Contains 1 row per PFP track, preserving all calorimetry systematic variations.
    """
    track_df, _ = _extract_base_track_df(f)
    return track_df


def make_pair_df(f: dict, proximity_cm: float = 1.0) -> pd.DataFrame:
    """
    Extracts PFP pair features from a flatcaf file for Pair BDT training and evaluation.
    """
    track_df, slice_idx_names = _extract_base_track_df(f)
    if track_df.empty:
        return pd.DataFrame()

    # 1. Format Parent DF
    p_rename = {
        "end_x": "end_x",
        "end_y": "end_y",
        "end_z": "end_z",
        "trk_len": "parent_len",
    }
    for c in track_df.columns:
        if c.startswith("chi2"):
            p_rename[c] = f"parent_{c}"

    p_df = track_df[list(p_rename.keys())].copy()
    p_df.columns = list(p_rename.values())
    p_df.index.names = [*slice_idx_names, "parent_pfp_index"]

    # 2. Format Daughter DF
    d_rename = {
        "start_x": "start_x",
        "start_y": "start_y",
        "start_z": "start_z",
        "trk_len": "daughter_len",
    }
    for c in track_df.columns:
        if c.startswith("chi2"):
            d_rename[c] = f"daughter_{c}"

    d_df = track_df[list(d_rename.keys())].copy()
    d_df.columns = list(d_rename.values())
    d_df.index.names = [*slice_idx_names, "daughter_pfp_index"]

    # 3. Cartesian Join
    p_flat = p_df.reset_index(level="parent_pfp_index")
    d_flat = d_df.reset_index(level="daughter_pfp_index")
    pairs = p_flat.join(d_flat, how="inner")

    pairs = pairs[pairs["parent_pfp_index"] != pairs["daughter_pfp_index"]]
    if pairs.empty:
        return pd.DataFrame()

    # 4. Proximity Cut
    dx = pairs["end_x"] - pairs["start_x"]
    dy = pairs["end_y"] - pairs["start_y"]
    dz = pairs["end_z"] - pairs["start_z"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    pairs["dist_to_parent_end"] = dist

    close_pairs = pairs[dist < proximity_cm].copy()
    if close_pairs.empty:
        return pd.DataFrame()

    # 5. Multiplicity
    multiplicity = close_pairs.groupby([*slice_idx_names, "parent_pfp_index"]).size().rename("n_close_to_parent_end")
    close_pairs = close_pairs.reset_index().merge(
        multiplicity.reset_index(),
        on=[*slice_idx_names, "parent_pfp_index"],
        how="left"
    )

    # 6. Finalize
    final_idx = [*slice_idx_names, "parent_pfp_index", "daughter_pfp_index"]
    pair_df = close_pairs.set_index(final_idx).sort_index()

    return pair_df
