import functools
import hashlib
import os

import makedf.getsyst as getsyst
import makedf.util as util
import pyanalib.pandas_helpers as ph
import numpy as np
import pandas as pd
from makedf import chi2pid
from makedf.makedf import make_mchdrdf, make_trkhitdf

from . import range_momentum
from .provenance import UNKNOWN_FILE_KEY, file_key

KPDG = {'kplus': 321, 'kzero': 311}
KMASS = {'kplus': 0.493677, 'kzero': 0.497611}
TRUE_KE_CUT = 0.

# Final-state species counted per true interaction: (pdg, mass GeV, KE cut GeV).
# The counts are needed for studies that select on matched final states, e.g.
# comparing the systematic budget for 1-pion against 1-kaon topologies.
#
# Thresholds approximate SBND reconstruction thresholds; they are deliberately
# fixed rather than tuned. A cut of 0. means "count all":
#   pi0/Lambda  decay immediately, the daughters carry the real threshold
#   neutron     not reconstructable; stored for truth bookkeeping only, do NOT
#               use it in a "matched final state" definition
#   kaons       true kaon KE is stored separately (true_E_kaon/true_P_kaon), so
#               the kaon threshold is applied downstream, on the fly
PRIM_SPECIES = {
    "n_piplus":  (  211, 0.139570, 0.025),
    "n_piminus": ( -211, 0.139570, 0.025),
    "n_pi0":     (  111, 0.134977, 0.0),
    "n_kplus":   (  321, 0.493677, 0.0),
    "n_kminus":  ( -321, 0.493677, 0.0),
    "n_k0":      (  311, 0.497611, 0.0),
    "n_klong":   (  130, 0.497611, 0.0),
    "n_kshort":  (  310, 0.497611, 0.0),
    "n_proton":  ( 2212, 0.938272, 0.050),
    "n_neutron": ( 2112, 0.939565, 0.0),
    "n_mu":      (   13, 0.105658, 0.025),
    "n_mubar":   (  -13, 0.105658, 0.025),
    "n_e":       (   11, 0.000511, 0.010),
    "n_ebar":    (  -11, 0.000511, 0.010),
    "n_gamma":   (   22, 0.0,      0.010),
    "n_lambda":  ( 3122, 1.115683, 0.0),
}

InFV_SBND = functools.partial(util.InFV, inzback=np.nan, det='SBND')

# Decay modes, keyed by the sorted multiset of the PDGs of the kaon's decay
# products. Recorded because true_type cannot express it: a three-body decay
# belongs in type 3, while a two-body decay whose MIP left the FV is acceptance
# loss, and today both land there with nothing to tell them apart.
DECAY_MODE_UNKNOWN = 0
DECAY_MODES = {
    (-13, 14): 1,          # Kmu2   K+ -> mu+ nu
    (-13, 14, 111): 2,     # Kmu3   K+ -> pi0 mu+ nu
    (111, 211): 3,         # K2pi   K+ -> pi+ pi0
    (-211, 211, 211): 4,   # K3pi   K+ -> pi+ pi+ pi-
    (-11, 12, 111): 5,     # Ke3    K+ -> pi0 e+ nu
    (111, 111, 211): 6,    # K3pi0  K+ -> pi+ pi0 pi0
}

# ``rec.true_particles.start_process`` for G4's Decay process. Which daughters
# belong to the decay is read off this rather than guessed from their PDGs: a
# delta ray is eIoni (14), an inelastic product is 13 and a recoiling nucleus is
# 31 or 44, so one comparison separates all three from the decay products. A PDG
# rule cannot -- the Ke3 positron and a delta ray are both an e+.
G4_PROCESS_DECAY = 3


def k_has_daughter(tpartdf: pd.DataFrame, ktype: str, require_contained: bool=True) -> pd.DataFrame:
    """
    Signal identification using backward hierarchy traversal.

    Returns a DataFrame indexed by (entry, interaction_id) with columns:
        daughter_pdg      the MIP type the kaon decayed to (-13 or 211)
        n_k_interactions  how many hadronic interactions the kaon underwent
                          before producing that MIP

    Handles arbitrary scattering depths and preserves the MIP type.

    n_k_interactions counts kaon->kaon steps in the chain, so 0 means the
    primary kaon decayed straight to the MIP, 1 means it re-interacted once and
    the resulting kaon segment decayed, and so on. Each iteration of the
    traversal below walks back exactly one generation, and every row in
    `active` sits at the same depth by construction, so a scalar counter is
    sufficient.

    Both the MIP and the `ndaughters_p <= 2` two-body guard are restricted to
    G4 *decay* products. Counting every daughter instead counts whatever else
    G4 recorded off the same track, and the recoiling argon nucleus of a decay
    at rest is enough to push a clean Kmu2 to three daughters and refuse it --
    4.5% of contained two-body decays over the 194 reference flatcafs, all of
    which then land in true_type 3.
    """
    nullcols = ['daughter_pdg', 'n_k_interactions']
    nulldf = pd.DataFrame(columns=nullcols, dtype=float)
    if tpartdf.empty:
        return nulldf

    df = tpartdf.copy()
    df.columns = ['.'.join(n for n in c if n != '') for c in df.columns]
    df = df.reset_index()

    decay_products = df[df.start_process == G4_PROCESS_DECAY]
    dtr_counts = decay_products.groupby(['entry', 'parent']).size().rename('ndaughters')
    df = df.merge(dtr_counts, left_on=['entry', 'G4ID'], right_index=True, how='left')
    df['ndaughters'] = df['ndaughters'].fillna(0)

    target_pdgs = [-13, 211] if ktype == 'kplus' else []
    # The MIP must itself be a decay product. Without this the guard above would
    # admit a pion knocked out of an inelastic interaction: such a kaon has no
    # decay daughters at all, so its ndaughters is 0 and `<= 2` passes vacuously.
    active = df[df.pdg.isin(target_pdgs)
                & (df.start_process == G4_PROCESS_DECAY)].copy()
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
    n_k_interactions = 0

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
            found = successes[['entry', 'interaction_id', 'daughter_pdg']].copy()
            found['n_k_interactions'] = n_k_interactions
            signal_results.append(found)

        active = parents[is_k_parent & ~is_primary_k].copy()
        if not active.empty:
            if require_contained:
                p_start = active[['start.x_p', 'start.y_p', 'start.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
                p_end = active[['end.x_p', 'end.y_p', 'end.z_p']].rename(columns=lambda x: x.split('_')[0].split('.')[-1])
                active = active[InFV_SBND(p_start) & InFV_SBND(p_end)]

            active['G4ID'] = active['G4ID_p']
            active['parent'] = active['parent_p']
            active = active[['entry', 'G4ID', 'parent', 'interaction_id', 'daughter_pdg']]

        # every surviving row has stepped back one more kaon generation
        n_k_interactions += 1

    if not signal_results:
        return nulldf

    res_df = pd.concat(signal_results).drop_duplicates()

    counts = res_df.groupby(['entry', 'interaction_id']).daughter_pdg.nunique()
    unambiguous = counts[counts == 1].index

    final_pdgs = res_df.set_index(['entry', 'interaction_id']).loc[unambiguous]
    # daughter_pdg is unique within each group by the `unambiguous` filter, so
    # 'first' is well defined. n_k_interactions is NOT: an interaction can hold
    # several MIPs of the same type from different kaons, each with its own
    # chain length. Take the shortest -- the least-reinteracted association.
    return final_pdgs.groupby(level=[0, 1]).agg(
        daughter_pdg=('daughter_pdg', 'first'),
        n_k_interactions=('n_k_interactions', 'min'),
    )

# ---------------------------------------------------------------------------
# True kaon bookkeeping
# ---------------------------------------------------------------------------
# Charged kaons only: neutral kaons leave no track, and a K0 appears here as a
# *producer* (charge exchange) rather than as a row of its own.
K_PDGS_TRACKED = (321, -321)

G4_PRIMARY_PARENT = 10000000

# How a kaon came to exist. Drives true_type 5-8.
KORIGIN_UNKNOWN = -1  # parent record missing from the particle list
KORIGIN_PRIMARY = 0   # produced at the neutrino vertex
KORIGIN_PI = 1        # pi+/- -> K
KORIGIN_K0 = 2        # K0 / K0L / K0S -> K, i.e. charge exchange
KORIGIN_P = 3         # proton -> K
KORIGIN_OTHER = 4     # anything else: neutron, muon, hyperons, ...

_K0_PDGS = (311, -311, 130, 310)
KORIGIN_COUNT_COLS = {
    KORIGIN_PRIMARY: "n_k_primary",
    KORIGIN_PI: "n_k_from_pi",
    KORIGIN_K0: "n_k_from_k0",
    KORIGIN_P: "n_k_from_p",
    KORIGIN_OTHER: "n_k_from_other",
}

TPART_BRANCHES = [
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
    'rec.true_particles.genE',
    'rec.true_particles.endE',
    'rec.true_particles.genp.x',
    'rec.true_particles.genp.y',
    'rec.true_particles.genp.z',
    'rec.true_particles.length',
    'rec.true_particles.contained',
    'rec.true_particles.start_process',
    'rec.true_particles.end_process',
]

# make_true_type_df and make_true_kaon_df both need the true-particle table and
# kaon_config lists both, so it is memoised on the file object exactly as the
# base track frame is. See the note on _base_track_df for why the key is the
# file object rather than id().
_TPART_CACHE = None  # (file_object, tpartdf)


def _true_particles(f):
    """rec.true_particles indexed by (entry, G4ID), memoised on the file."""
    global _TPART_CACHE
    if _TPART_CACHE is not None and _TPART_CACHE[0] is f:
        return _TPART_CACHE[1]
    tpartdf = ph.loadbranches(f["recTree"], TPART_BRANCHES).rec.true_particles
    tpartdf = tpartdf.reset_index().set_index(['entry', 'G4ID'])
    _TPART_CACHE = (f, tpartdf)
    return tpartdf


def clear_true_particles_cache():
    """Drop the cached true-particle table."""
    global _TPART_CACHE
    _TPART_CACHE = None


def _flat_true_particles(tpartdf):
    """Flatten the loadbranches column tuples and lift the index into columns."""
    df = tpartdf.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['.'.join(n for n in c if n != '') for c in df.columns]
    return df.reset_index()


def _classify_producer(pdg):
    pdg = np.asarray(pdg)
    out = np.full(pdg.shape, KORIGIN_OTHER, dtype=np.int64)
    out[np.abs(pdg) == 211] = KORIGIN_PI
    out[np.isin(pdg, _K0_PDGS)] = KORIGIN_K0
    out[pdg == 2212] = KORIGIN_P
    return out


def _empty_origins():
    return pd.DataFrame(
        {"origin": [], "origin_pdg": [], "n_reinteractions": []},
        index=pd.MultiIndex.from_arrays([[], []], names=["entry", "G4ID"]),
    ).astype(int)


def resolve_k_origins(tpartdf, max_depth=32):
    """Classify every charged kaon in the event by what ultimately produced it.

    Walks back through K -> K reinteraction chains, so a primary kaon that
    scatters is still classed as primary rather than as "produced by a kaon".

    Returns one row per kaon, keyed by (entry, G4ID):
        origin            KORIGIN_* class
        origin_pdg        pdg of the ultimate non-kaon ancestor, 0 if primary
        n_reinteractions  kaon -> kaon steps back to that ancestor

    Named n_reinteractions, not n_k_interactions, so that a per-kaon row can be
    joined against true_type_df without a column collision -- true_type_df
    already carries an n_k_interactions for its matched signal chain.

    Unlike k_has_daughter this does NOT require the kaon to decay to anything:
    that function starts from MIP candidates and so can only ever see kaons
    that produced one.
    """
    df = _flat_true_particles(tpartdf)
    kaons = df[df.pdg.isin(K_PDGS_TRACKED)]
    if kaons.empty:
        return _empty_origins()

    lookup = df[['entry', 'G4ID', 'pdg', 'parent']].rename(
        columns={'G4ID': 'pG4', 'pdg': 'ppdg', 'parent': 'pparent'})

    cur = kaons[['entry', 'G4ID', 'parent']].copy()
    cur['k_G4ID'] = cur['G4ID']  # the kaon this row is resolving, fixed as we walk
    found, depth = [], 0

    def _record(rows, origin, origin_pdg):
        r = rows[['entry', 'k_G4ID']].copy()
        r['origin'] = origin
        r['origin_pdg'] = origin_pdg
        r['n_reinteractions'] = depth
        found.append(r)

    while not cur.empty and depth < max_depth:
        at_vertex = cur.parent == G4_PRIMARY_PARENT
        if at_vertex.any():
            _record(cur[at_vertex], KORIGIN_PRIMARY, 0)
        cur = cur[~at_vertex]
        if cur.empty:
            break

        m = cur.merge(lookup, left_on=['entry', 'parent'],
                      right_on=['entry', 'pG4'], how='left')
        lost = m.ppdg.isna()
        if lost.any():
            _record(m[lost], KORIGIN_UNKNOWN, 0)
        m = m[~lost]

        is_k = m.ppdg.isin(K_PDGS_TRACKED)
        stop = m[~is_k]
        if not stop.empty:
            _record(stop, _classify_producer(stop.ppdg.to_numpy()),
                    stop.ppdg.to_numpy().astype(np.int64))

        cur = m[is_k][['entry', 'k_G4ID', 'pG4', 'pparent']].rename(
            columns={'pG4': 'G4ID', 'pparent': 'parent'})
        depth += 1

    if not found:
        return _empty_origins()
    res = pd.concat(found).rename(columns={'k_G4ID': 'G4ID'})
    return res.set_index(['entry', 'G4ID'])[
        ['origin', 'origin_pdg', 'n_reinteractions']].astype(int)


def k_origin_counts(tpartdf):
    """Per-interaction counts of kaon PRODUCTIONS by origin class, for true_type 5-8.

    Reinteraction segments are not counted again -- see the filter below.
    """
    cols = list(KORIGIN_COUNT_COLS.values())
    empty = pd.DataFrame(
        {c: [] for c in cols},
        index=pd.MultiIndex.from_arrays([[], []], names=["entry", "interaction_id"]),
    ).astype(int)

    df = _flat_true_particles(tpartdf)
    kaons = df[df.pdg.isin(K_PDGS_TRACKED)][['entry', 'G4ID', 'interaction_id']]
    origins = resolve_k_origins(tpartdf)
    if kaons.empty or origins.empty:
        return empty

    j = kaons.join(origins, on=['entry', 'G4ID'])
    j = j[j.origin.notna()].copy()
    # Count PRODUCTIONS, not track segments. A kaon that reinteracts leaves a
    # second G4 track that is also a kaon; n_reinteractions == 0 marks the first
    # segment after a production point, so filtering on it counts each kaon once
    # however many times it scattered. The per-segment rows are all kept in
    # make_true_kaon_df, which is about tracks rather than productions.
    j = j[j.n_reinteractions == 0]
    if j.empty:
        return empty
    # an unresolvable parent is counted with "other" rather than dropped
    j['origin'] = j['origin'].astype(int).replace(KORIGIN_UNKNOWN, KORIGIN_OTHER)

    counts = j.pivot_table(index=['entry', 'interaction_id'], columns='origin',
                           values='G4ID', aggfunc='count', fill_value=0)
    counts = counts.rename(columns=KORIGIN_COUNT_COLS)
    for c in cols:
        if c not in counts.columns:
            counts[c] = 0
    return counts[cols].astype(int)


def k_decay_mode(df: pd.DataFrame, kaons: pd.DataFrame) -> np.ndarray:
    """DECAY_MODES code per kaon, from the PDGs of its Geant4 decay products.

    The daughters are taken from the *parent links* already in `df` -- the rows
    whose `parent` is this kaon's `G4ID` -- and not from
    `rec.true_particles.daughters`. The explicit list would be the more direct
    source and cannot be used: `loadbranches` refuses it alongside `pdg`
    ("different vector structures in the CAF"), because it is jagged per particle
    where every other branch here is one value per particle. The two agree --
    checked on 113 primary kaons over 60 flatcafs, identical daughter multisets
    in every case -- so nothing is lost but the directness.

    A daughter counts iff `start_process` is Decay. Selecting on PDG instead
    cannot express Ke3: dropping e+ as a delta ray drops the positron that
    identifies the mode, leaving (12, 111) and no entry to match, so mode 5 is
    unreachable and its ~5% of decays fall into DECAY_MODE_UNKNOWN alongside the
    absorptions.
    """
    by_parent = {}
    for e, p, g, start in zip(df['entry'], df['parent'], df['pdg'],
                              df['start_process']):
        if start == G4_PROCESS_DECAY:
            by_parent.setdefault((e, p), []).append(g)

    out = []
    for e, g4 in zip(kaons['entry'], kaons['G4ID']):
        products = sorted(by_parent.get((e, g4), ()))
        out.append(DECAY_MODES.get(tuple(products), DECAY_MODE_UNKNOWN))
    return np.asarray(out, dtype=np.int64)


#: What the CAF writes into a true particle's trajectory when it stored none: the
#: whole record at once -- ``endE``, ``start.*``, ``end.*`` -- and ``length`` 0. It
#: is the only negative value any of those take, so a sign test identifies it
#: exactly; a real kaon's ``endE`` starts at its mass and nothing lands between.
CAF_NO_TRAJECTORY = -9999.

#: Columns that carry it, and therefore must not be arithmetic.
TRAJECTORY_COLUMNS = ('endE', 'start.x', 'start.y', 'start.z',
                      'end.x', 'end.y', 'end.z', 'length')


def _mask_missing_trajectory(kaons: pd.DataFrame) -> pd.DataFrame:
    """NaN the trajectory of a kaon the CAF stored none for.

    11% of kaons over the reference flatcafs, all with the CAF's own
    ``contained == 0``. Left raw, the sentinel becomes a *number* in every derived
    column: ``ke_end`` reads -9999.49 GeV, and since that column exists to say
    whether the kaon stopped, a ``ke_end < 0.05`` cut then admits every one of them
    as stopped -- backwards, and silent. ``length`` reads 0 for a kaon of several
    GeV.

    Containment is unchanged by design rather than by luck: ``InFV`` is False on
    -9999 and on NaN alike, and ``is_fv_contained`` fills NaN to False, so these
    kaons stay uncontained -- now for the right reason. ``genE`` comes from the
    generator and is untouched, so ``E``, ``KE`` and ``P`` remain valid.
    """
    kaons = kaons.copy()
    missing = kaons['endE'] < 0
    if not missing.any():
        return kaons
    for column in TRAJECTORY_COLUMNS:
        if column in kaons.columns:
            kaons.loc[missing, column] = np.nan
    return kaons


def make_true_kaon_df(f: dict) -> pd.DataFrame:
    """One row per true charged kaon, with its provenance and kinematics.

    Indexed (entry, rec.mc.nu..index, ikaon) -- the same leading levels as
    true_type_df plus a per-interaction counter -- so the two join directly:

        kdf.join(ttdf)                       # interaction info onto each kaon
        ttdf.join(kdf.groupby(level=[0,1,2]).size().rename("n_k"))

    Most interactions have no kaon at all, so this table is tiny; keeping it
    separate is what lets an interaction carry several kaons without squeezing
    them into one row of true_type_df.

    Kaons not associated with a neutrino interaction (cosmic-induced) carry
    interaction_id == -1 and are RETAINED here -- they simply never align with
    a true_type_df row, which is the correct behaviour.

    start_process / end_process are raw caf enum codes (see sbnanaobj
    SREnums.h); end_process distinguishes a decay from an inelastic absorption.

    daughter_contained is the FV containment of the MIP named by daughter_pdg,
    which is the acceptance cut true_type folds in silently. To recover the
    per-interaction view:

        ttdf.join(kdf.groupby(level=[0, 1, 2]).daughter_contained.any()
                     .rename("has_contained_mip"))
    """
    tpartdf = _true_particles(f)
    df = _flat_true_particles(tpartdf)
    kaons = df[df.pdg.isin(K_PDGS_TRACKED)].copy()
    if kaons.empty:
        return pd.DataFrame()

    kaons = _mask_missing_trajectory(kaons)
    kaons = kaons.join(resolve_k_origins(tpartdf), on=['entry', 'G4ID'])

    # immediate parent, which may itself be a kaon if this segment came from a
    # reinteraction -- origin_pdg is the ultimate non-kaon ancestor instead
    par = df[['entry', 'G4ID', 'pdg']].rename(
        columns={'G4ID': 'pG4', 'pdg': 'parent_pdg'})
    kaons = kaons.merge(par, left_on=['entry', 'parent'],
                        right_on=['entry', 'pG4'], how='left').drop(columns='pG4')
    kaons['parent_pdg'] = kaons['parent_pdg'].fillna(0).astype(int)

    # charged-MIP daughter, if any. No containment requirement is imposed on
    # which kaons are kept -- the kdf is raw truth and the analysis applies its
    # own cuts -- but the daughter's containment is RECORDED, because that is
    # the cut k_has_daughter applies (require_contained) before it will call a
    # chain signal. Without it, "the kaon never decayed to a MIP" and "it did,
    # but the MIP left the FV" both collapse into true_type 3 and cannot be
    # told apart downstream. On the exclusive-kaon sample the second case is
    # about two thirds of that category, so the distinction is not marginal:
    # it separates a real background from a pure acceptance loss.
    mip_cols = ['entry', 'parent', 'pdg',
                'start.x', 'start.y', 'start.z', 'end.x', 'end.y', 'end.z']
    mips = df[df.pdg.abs().isin((13, 211))][mip_cols].copy()
    mips = mips.rename(columns={'pdg': 'daughter_pdg'})
    mip_start = mips[['start.x', 'start.y', 'start.z']].rename(
        columns=lambda c: c.split('.')[-1])
    mip_end = mips[['end.x', 'end.y', 'end.z']].rename(
        columns=lambda c: c.split('.')[-1])
    mips['daughter_contained'] = (
        InFV_SBND(mip_start) & InFV_SBND(mip_end)).fillna(False)

    # Both aggregations are 'first' over the same groupby, so daughter_contained
    # describes the same MIP that daughter_pdg names, not some other daughter.
    # n_mip_daughters says when that choice was not unique.
    grouped = mips.groupby(['entry', 'parent'])
    first_mip = grouped[['daughter_pdg', 'daughter_contained']].first()
    n_mip = grouped.size().rename('n_mip_daughters')
    kaons = kaons.join(first_mip, on=['entry', 'G4ID']).join(n_mip, on=['entry', 'G4ID'])
    kaons['daughter_pdg'] = kaons['daughter_pdg'].fillna(0).astype(int)
    kaons['daughter_contained'] = kaons['daughter_contained'].fillna(False).astype(bool)
    kaons['n_mip_daughters'] = kaons['n_mip_daughters'].fillna(0).astype(int)

    kaons['decay_mode'] = k_decay_mode(df, kaons)

    # Kinetic energy at the END of the track. E/KE below are from genE, the
    # energy at *production*, which does not say whether the kaon stopped -- and
    # stopping is what makes a Kmu2 muon monochromatic at 258.15 MeV, the
    # sharpest truth handle in the sample. NaN where the CAF stored no
    # trajectory, which _mask_missing_trajectory has already marked: "no end
    # recorded" and "ended at rest" must not be the same value in the one column
    # asked to tell them apart.
    kaons['ke_end'] = kaons['endE'] - KMASS['kplus']

    p = np.sqrt(kaons['genp.x']**2 + kaons['genp.y']**2 + kaons['genp.z']**2)
    kaons['P'] = p
    kaons['KE'] = kaons['genE'] - KMASS['kplus']

    start = kaons[['start.x', 'start.y', 'start.z']].rename(
        columns=lambda c: c.split('.')[-1])
    end = kaons[['end.x', 'end.y', 'end.z']].rename(columns=lambda c: c.split('.')[-1])
    kaons['is_fv_contained'] = (InFV_SBND(start) & InFV_SBND(end)).fillna(False)

    kaons = kaons.rename(columns={
        'start.x': 'start_x', 'start.y': 'start_y', 'start.z': 'start_z',
        'end.x': 'end_x', 'end.y': 'end_y', 'end.z': 'end_z',
        'genp.x': 'px', 'genp.y': 'py', 'genp.z': 'pz',
        'genE': 'E', 'interaction_id': 'rec.mc.nu..index',
    })

    keep = ['pdg', 'G4ID', 'parent', 'parent_pdg', 'origin', 'origin_pdg',
            'n_reinteractions', 'daughter_pdg', 'daughter_contained',
            'n_mip_daughters', 'decay_mode', 'ke_end',
            'E', 'KE', 'P', 'px', 'py', 'pz', 'length',
            'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
            'contained', 'is_fv_contained', 'start_process', 'end_process']
    keep = [c for c in keep if c in kaons.columns]

    kaons = kaons.sort_values(['entry', 'rec.mc.nu..index', 'G4ID'])
    kaons['ikaon'] = kaons.groupby(['entry', 'rec.mc.nu..index']).cumcount()
    out = kaons.set_index(['entry', 'rec.mc.nu..index', 'ikaon'])[keep].sort_index()
    return out


def k_chain_contained(tpartdf, ktype: str = 'kplus') -> pd.Series:
    """Is every segment of the interaction's kaon FV-contained, start and end?

    This is the containment ``k_has_daughter`` actually requires. That function
    walks back along the kaon chain and applies ``InFV_SBND(start) & InFV_SBND(end)``
    to the MIP (line 93), to the primary kaon when it reaches one (line 132), and
    to **every intermediate segment on the way** (lines 140-145). A kaon that
    re-interacts is several G4 tracks, and all of them have to be inside.

    ``make_true_type_df`` used to test something different and looser: the *end
    point only*, of the first primary kaon, off ``rec.mc.nu.prim``. Two
    containment definitions for one object, twenty lines apart, neither
    mentioning the other -- and the looser one decides type 3 against type 4
    while the stricter one decides type 1 against type 3. An interaction could
    therefore be "contained" for the 3/4 split and "not contained" for the 1/3
    one, which is not a distinction anybody intended.

    Segments are grouped by ``resolve_k_origins``' ``KORIGIN_PRIMARY`` class, so
    a kaon produced later in a shower does not drag the flag down. An interaction
    holding two vertex kaons ANDs both; that is 0 of 194 flatcafs here and the
    per-interaction flag has no way to say "one of them" anyway.

    Returns a bool Series indexed by (entry, interaction_id). Interactions with
    no charged kaon are absent, and the caller reindexes to False.
    """
    df = _flat_true_particles(tpartdf)
    # This kaon's own charge, not K_PDGS_TRACKED. is_k_contained describes the
    # K+ of the interaction, and an uncontained K- alongside it must not drag the
    # flag down -- that produced exactly one true_type 1 interaction flagged
    # "kaon not contained" in an early version, which is the invariant this
    # function exists to protect.
    kaons = df[df.pdg == KPDG[ktype]]
    if kaons.empty:
        return pd.Series(dtype=bool)

    origins = resolve_k_origins(tpartdf)
    kaons = kaons.join(origins['origin'], on=['entry', 'G4ID'])
    kaons = kaons[kaons['origin'] == KORIGIN_PRIMARY]
    if kaons.empty:
        return pd.Series(dtype=bool)

    start = kaons[['start.x', 'start.y', 'start.z']].rename(
        columns=lambda c: c.split('.')[-1])
    end = kaons[['end.x', 'end.y', 'end.z']].rename(
        columns=lambda c: c.split('.')[-1])
    contained = (InFV_SBND(start) & InFV_SBND(end)).fillna(False)
    return contained.groupby(
        [kaons['entry'], kaons['interaction_id']]).all()


def make_true_type_df(f) -> pd.DataFrame:
    """
    Creates a minimal dataframe with exactly 1 row per true neutrino interaction.
    """
    # 1. Load basic neutrino info
    nu_branches = ['rec.mc.nu.E', 'rec.mc.nu.iscc', 'rec.mc.nu.genie_mode',
                   'rec.mc.nu.position.x', 'rec.mc.nu.position.y', 'rec.mc.nu.position.z']
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

    kplus_first = kplus_df.groupby(level=[0, 1]).first()
    true_E_kaon = kplus_first['genE']
    true_P_kaon = np.sqrt(np.maximum(0, true_E_kaon**2 - KMASS['kplus']**2)).fillna(0)

    # The containment k_has_daughter requires -- every segment of the kaon,
    # start and end -- rather than the end point of the first one. See
    # k_chain_contained for why the two used to differ and what that cost.
    is_k_contained = k_chain_contained(_true_particles(f))
    is_k_contained = is_k_contained.reindex(mcdf.index).fillna(False).astype(bool)

    # 3. Daughter association (hierarchy traversal)
    tpartdf = _true_particles(f)

    k_info = k_has_daughter(tpartdf, 'kplus', require_contained=True)
    k_counts = k_origin_counts(tpartdf)

    # 4. Consolidate into our minimal dataframe
    res = pd.DataFrame(index=mcdf.index)
    res['is_cc'] = is_cc.fillna(False).astype(bool)
    res['is_true_fv'] = is_true_fv.fillna(False).astype(bool)
    res['true_E_nu'] = true_E_nu

    # GENIE interaction mode, as the raw genie::EScatteringType integer
    # (0 QE, 1 RES, 2 DIS, 3 COH, 10 MEC, ...). Deliberately not mapped to
    # labels here: the mapping is a presentation choice, and a value this
    # code does not recognise has to stay visible rather than fall into an
    # "other" bucket. -1 means the interaction carries no mode at all, which
    # is not the same as mode 0.
    res['genie_mode'] = mcdf['genie_mode'].fillna(-1).astype(int)

    # Final-state multiplicities above threshold, one column per species.
    # n_kplus supersedes the old standalone 'nkplus'.
    for _name, (_pdg, _mass, _ke_cut) in PRIM_SPECIES.items():
        _sel = (mcprimdf.pdg == _pdg) & ((mcprimdf.genE - _mass) > _ke_cut)
        res[_name] = _sel.groupby(level=[0, 1]).sum().fillna(0).astype(int)

    res['true_E_kaon'] = true_E_kaon
    res['true_E_kaon'] = res['true_E_kaon'].fillna(0)

    res['true_P_kaon'] = true_P_kaon
    res['true_P_kaon'] = res['true_P_kaon'].fillna(0)

    res['is_k_contained'] = is_k_contained
    res['is_k_contained'] = res['is_k_contained'].fillna(False).astype(bool)

    res['k_daughter_pdg'] = k_info['daughter_pdg']

    # Number of hadronic interactions the matched kaon underwent. -1 means no
    # kaon -> MIP chain was matched at all, which is not the same as 0.
    res['n_k_interactions'] = k_info['n_k_interactions']
    res['n_k_interactions'] = res['n_k_interactions'].fillna(-1).astype(int)

    # How many kaons this interaction has, split by what produced them. Counted
    # over every kaon in the event, whether or not it decayed -- see
    # resolve_k_origins. Full per-kaon detail lives in make_true_kaon_df.
    for _col in KORIGIN_COUNT_COLS.values():
        res[_col] = k_counts[_col] if _col in k_counts.columns else 0
        res[_col] = res[_col].fillna(0).astype(int)

    # 5. Compute 'true_type'
    #     1  K+ -> mu+          (signal)
    #     2  K+ -> pi+          (signal)
    #     3  other contained K+
    #     4  uncontained K+
    #     5  secondary K from a pion       )
    #     6  secondary K from a K0         )  no K+ at the vertex, but one was
    #     7  secondary K from a proton     )  produced later in the shower
    #     8  secondary K from anything else)
    #    98  no K+ in FV
    #    99  neutrino vertex out of FV
    true_type = pd.Series(np.nan, index=res.index)

    true_type[~res['is_true_fv']] = 99
    true_type[res['is_true_fv'] & (res['n_kplus'] < 1)] = 98

    # Secondary kaons carve out of 98: no primary K+ at the vertex, but a kaon
    # was produced later in a hadronic interaction. Such a kaon decaying to a
    # MIP reconstructs like signal, so it is a background whose rate carries the
    # pi -> K (and p -> K) production uncertainty.
    #
    # Assigned in reverse precedence so pi wins over K0 over p over other when
    # an interaction holds several; the n_k_from_* counts are kept as columns so
    # an analysis can be finer than this single label.
    no_prim_k = res['is_true_fv'] & (res['n_kplus'] < 1)
    true_type[no_prim_k & (res['n_k_from_other'] > 0)] = 8
    true_type[no_prim_k & (res['n_k_from_p'] > 0)] = 7
    true_type[no_prim_k & (res['n_k_from_k0'] > 0)] = 6
    true_type[no_prim_k & (res['n_k_from_pi'] > 0)] = 5

    is_k_fv = res['is_true_fv'] & (res['n_kplus'] > 0)
    true_type[is_k_fv & (res['k_daughter_pdg'] == -13)] = 1
    true_type[is_k_fv & (res['k_daughter_pdg'] == 211)] = 2
    true_type[is_k_fv & res['is_k_contained'] & true_type.isna()] = 3
    true_type[is_k_fv & ~res['is_k_contained'] & true_type.isna()] = 4

    res['true_type'] = true_type.astype(int)

    # is_true_fv was previously computed and then dropped; it is needed to
    # reconstruct the true_type decision downstream.
    keep_cols = (['true_type', 'is_cc', 'genie_mode', 'is_true_fv', 'is_k_contained',
                  'true_E_nu', 'true_E_kaon', 'true_P_kaon', 'n_k_interactions']
                 + list(KORIGIN_COUNT_COLS.values())
                 + list(PRIM_SPECIES))
    return res[keep_cols]




# --- file provenance --------------------------------------------------------
# Which flatcaf a selected slice came from, so an event can be traced back to
# its input and its hits re-read for a dE/dx profile.
#
# The join key already exists. pyanalib.ntuple_glob tags every df it builds with
# __ntuple, the position of the file in that job's input list, and that level is
# on every row of every table. What has never existed is the right-hand side of
# the join: nothing in a product records which file a given __ntuple was. It
# cannot be reconstructed either -- run_df_maker.run_grid deals the input list
# round-robin across jobs (flistForEachJob[i_line % ngrid]) and each job
# renumbers from zero, so recovering it needs ngrid, the list, and which files
# failed, none of which the product carries.
#
# Almost a maker rather than a loader change: _execute_load hands each entry in a
# config's DFS the open uproot file, and uproot's ReadOnlyDirectory carries
# file_path, so the file's own name is reachable from in here.
#
# One loader line is needed, though, and the first version of this patch was wrong
# to claim otherwise. _execute_load does not call every maker on every file: one
# with no recTree, or TotalEvents == 0, skips the whole DFS list, so make_file_df
# never ran on it and the label it consumed got no row -- 621 of 14,821 flatcafs
# (4.2%) on the first patched production. See loader_all_files.patch.

#: ``n_entry`` when the count could not be read.  Not 0, which is a legitimate
#: answer for a file holding no events.
UNKNOWN_N_ENTRY = -1


def make_file_df(f) -> pd.DataFrame:
    """One row identifying the flatcaf this ntuple was built from.

    Indexed on ``entry`` like ``make_histpotdf``, the other per-file table, so
    the loader's ``__ntuple`` tagging leaves it with the ``(__ntuple, entry)``
    index every other single-valued table has.

    The name itself is deliberately NOT a column.  PyTables cannot map a Python
    object column, or a numpy bytes column, to a C type in ``format="fixed"``,
    which is what ``run_df_maker.run_pool`` writes with -- so it pickles the
    block, and the pickled node costs ~1.09 MB however few rows it holds.
    Measured against a 1.055 MB ``dfs/kex`` product: +23 kB for the key alone,
    +30 kB with ``n_entry``, +1089 kB with the name, i.e. the name would roughly
    double the product.  ``file_key`` is a pure function of the basename, so the
    names come back by hashing the input list, which is version-controlled in
    ``lists/``.  If a name ever fails to resolve, the key simply misses, which is
    visible rather than wrong.

    ``n_entry`` is the one fact here that becomes unrecoverable once a flatcaf is
    deleted from dCache, and it is what makes a partial read detectable; it costs
    ~7 kB per product.  Drop it if that is not worth it.

    Runs on every input file, including one with no recTree or no events, via the
    ``runs_without_rectree`` flag set below.  ``n_entry`` then separates the cases:
    ``-1`` no recTree, ``0`` a recTree holding nothing, ``N`` a normal file -- so an
    absence from the export means "the loader never got this file" rather than
    pooling that with "the file was empty".

    Never returns None: ``run_pool`` zips ``NAMES`` against the maker results in
    reverse, and a None return is dropped from that list rather than held as a
    gap, which would silently shift every table's name.

    Must stay LAST in the config's DFS.  The reverse zip means the makers that run
    without a recTree have to be a suffix of DFS for the names to line up;
    _file_level_dfs raises if they are not.
    """
    name = ""
    path = getattr(f, "file_path", None)
    if path:
        # Verbatim, not normalised. In grid mode -- how the production runs --
        # run_grid xrdcp's each file into the job's cwd and passes bare
        # basenames, so file_path IS the basename; in pool mode it is an xroot
        # URL or a /pnfs path, and basename handles both. Two loader paths do
        # rename the file first (the streaming-timeout failover to
        # {uuid4()}_{basename}, and PREPROCESS to temp{i}_{uuid4()}...), but
        # failover needs a network read, which grid mode does not do, and no
        # kaon config sets PREPROCESS.
        name = os.path.basename(str(path))

    n_entry = UNKNOWN_N_ENTRY
    try:
        if f is not None and "recTree" in f:
            n_entry = int(f["recTree"].num_entries)
    except Exception:
        pass

    df = pd.DataFrame(
        {
            "file_key": pd.Series([file_key(name)], dtype="uint64"),
            "n_entry": pd.Series([n_entry], dtype="int64"),
        }
    )
    df.index.name = "entry"
    return df


#: Run this maker even when the file has no usable recTree -- see _file_level_dfs
#: in pyanalib/ntuple_glob.py.  It is the only maker here that does not read
#: recTree, and the only one whose value is in existing for files that hold nothing.
make_file_df.runs_without_rectree = True


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
        'rec.slc.tmatch.index',
        # The reconstructed slice vertex. legacy_make_kaon_df.make_kaon_recodf
        # applied InFV_SBND(slc.vertex) as a production pre-cut, so kaonana's
        # cutflow inherited it and the pair BDT was trained downstream of it --
        # but it is in no cut sequence and in none of the new products, so the
        # port has been running without it. Exported rather than applied, per
        # the analysis-time-filtering policy: 3 floats on the smallest table.
        'rec.slc.vertex.x',
        'rec.slc.vertex.y',
        'rec.slc.vertex.z',
    ]

    # Load slice branches
    slc_df = ph.loadbranches(f["recTree"], branches_to_load).rec.slc

    # Simplify the column names from the deeply nested cafpyana tuples
    if isinstance(slc_df.columns, pd.MultiIndex):
        slc_df.columns = ["_".join([str(c) for c in col if c]).strip() for col in slc_df.columns.values]

    # PRE-CUT: Only keep slices that are not clear cosmics
    slc_df = slc_df[slc_df.is_clear_cosmic == 0]

    return slc_df

#: Multisim universes kept per knob.  Already low -- 49 multisim knobs at 100
#: universes are 4,900 of the table's 5,465 columns, and a multisim covariance
#: converges as 1/sqrt(N), so going below this trades a real input for bytes.
MULTISIM_NUNIV = 100


def make_syst_df(f: dict) -> pd.DataFrame:
    """Reweight systematics, **one row per true interaction**.

    Indexed ``(entry, rec.mc.nu..index)`` -- the same leading levels as
    ``make_true_type_df`` -- so it joins to truth directly and needs nothing from
    reconstruction.

    Why per interaction, and not per slice
    --------------------------------------
    A reweight is a property of the interaction: GENIE and flux weights are
    functions of the true final state and the parent hadron, and no part of
    reconstruction enters them.  The previous version mapped them onto slices
    through ``rec.slc.tmatch.index``, which had three costs:

    * **Duplication.**  Two slices matching one interaction stored two identical
      copies of 5,465 floats, and the count scales with the slice gate rather
      than with the physics.
    * **Coupling.**  The table could not be produced without the reco matching,
      so it had to be rebuilt whenever reconstruction changed even though not one
      weight moved.  Kept separate, it is written once and reused across every
      reprocessing.
    * **A decision taken silently.**  Unmatched slices were reindexed to weight
      **1.0**, which asserts "nominal" for an object that has no interaction
      behind it at all.  With low matching efficiency that is a physics choice
      buried in a df maker.  Per interaction there is nothing to assert: a slice
      that matched nothing simply has no partner, and how to treat it belongs to
      the analysis, where it can be counted.

    Consumers join on the interaction, which is what
    ``kaonana.analysis.selection.attach_true_type`` already does for
    ``true_type``.
    """
    systs = getsyst.get_all_syst_df(f, multisim_nuniv=MULTISIM_NUNIV)

    if systs is None or systs.empty:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["entry", "rec.mc.nu..index"]
            )
        )

    # Flatten MultiIndex columns of systs if they are tuples
    if isinstance(systs.columns, pd.MultiIndex):
        systs.columns = ["_".join([str(c) for c in col if c]).strip() for col in systs.columns.values]

    # get_all_syst_df indexes (entry, inu); rename to the products' MC convention
    # so the table lines up with true_type and true_kaon without a translation.
    systs.index.names = ["entry", "rec.mc.nu..index"]

    return systs.sort_index()

def _extract_base_track_df(f: dict):
    """
    Helper function that extracts single-track features and computes
    all systematic calorimetry variations.
    """
    branches_to_load = [
        'rec.slc.reco.pfp.trackScore',
        'rec.slc.reco.pfp.trk.len',
        'rec.slc.reco.pfp.trk.start.x',
        'rec.slc.reco.pfp.trk.start.y',
        'rec.slc.reco.pfp.trk.start.z',
        'rec.slc.reco.pfp.trk.end.x',
        'rec.slc.reco.pfp.trk.end.y',
        'rec.slc.reco.pfp.trk.end.z',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_kaon',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_muon',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_proton',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_proton',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_proton',
        'rec.slc.reco.pfp.trk.chi2pid.0.chi2_pion',
        'rec.slc.reco.pfp.trk.chi2pid.1.chi2_pion',
        'rec.slc.reco.pfp.trk.chi2pid.2.chi2_pion',
        # Range-momentum under the pion hypothesis. Not used by any selection --
        # it is the closure reference for trk_rangeP_kaon, which we compute
        # ourselves because caf::SRTrkRange has no kaon slot. sbncode builds
        # p_pion by mass-scaling the muon CSDA table (RangePAllPID_module.cc:
        # 89-95), which is exactly the transformation range_momentum.p_kaon
        # applies with m_K, so reproducing this column from trk_len is what
        # demonstrates the kaon column is what a patched sbncode would write.
        # Drop this line and the col_map entry if the extra float is not wanted;
        # nothing downstream reads it.
        'rec.slc.reco.pfp.trk.rangeP.p_pion',
        # Backtracked truth of the track's best-matched true particle. Needed
        # to say what a selected pair actually was -- chi2 PID answers what it
        # looks like, which is a different question and the only one the
        # products could answer before this.
        'rec.slc.reco.pfp.trk.truth.p.pdg',
        'rec.slc.reco.pfp.trk.truth.p.G4ID',
        'rec.slc.reco.pfp.trk.truth.p.interaction_id',
        'rec.slc.reco.pfp.trk.truth.p.genE',
        # parent and the true trajectory end points are what a *pair* label
        # needs, as opposed to a single track's identity:
        #   parent      daughter.truth_parent == parent.truth_G4ID is the test
        #               that this daughter really is that parent's decay
        #               product, rather than a coincidental muon in the slice.
        #   start/end   the true containment of both legs.
        # Together with pdg these are the four terms in
        # pair_bdt/scripts/extract_nu_pairs.py's signal definition; without
        # them a retrain cannot label its own training set.
        'rec.slc.reco.pfp.trk.truth.p.parent',
        'rec.slc.reco.pfp.trk.truth.p.start.x',
        'rec.slc.reco.pfp.trk.truth.p.start.y',
        'rec.slc.reco.pfp.trk.truth.p.start.z',
        'rec.slc.reco.pfp.trk.truth.p.end.x',
        'rec.slc.reco.pfp.trk.truth.p.end.y',
        'rec.slc.reco.pfp.trk.truth.p.end.z',
    ]

    pandora_df = ph.loadbranches(f["recTree"], branches_to_load).rec.slc.reco
    slice_idx_names = list(pandora_df.index.names)[:-1]
    pfp_idx_col = pandora_df.index.names[-1]

    # PRE-CUT: Drop clear cosmics to save space
    slc_df = ph.loadbranches(f["recTree"], ['rec.slc.is_clear_cosmic']).rec.slc
    if isinstance(slc_df.columns, pd.MultiIndex):
        slc_df.columns = ["_".join([str(c) for c in col if c]).strip() for col in slc_df.columns.values]
    is_cosmic = slc_df['is_clear_cosmic'] == 1
    levels_to_drop = list(range(is_cosmic.index.nlevels, pandora_df.index.nlevels))
    is_cosmic_aligned = is_cosmic.reindex(pandora_df.index.droplevel(levels_to_drop))
    pandora_df = pandora_df[~is_cosmic_aligned.values]

    # loadbranches sets the column arity from the DEEPEST branch requested and
    # pads everything shallower with trailing "". So adding a branch one level
    # deeper than the previous deepest renumbers *every* key below.
    # truth.p.start.x is exactly that case: 6 levels against chi2pid's 5.
    #
    # This does fail rather than pass quietly, but it fails unrecognisably --
    # the hardcoded 5-tuples still test True against a 6-level MultiIndex
    # (pandas reads a short tuple as a partial key), so cols_to_keep looks
    # right and the frame selection below then raises a bare
    #   AssertionError: Length of new_levels (6) must be <= self.nlevels (5)
    # from inside pandas, which says nothing about branch depth. Keys are
    # therefore given at their natural depth and padded to the frame's arity
    # here, so adding a deeper branch needs no edits below and a key that is
    # genuinely too deep raises with its own name in the message.
    _depth = pandora_df.columns.nlevels

    def _key(*parts):
        if len(parts) > _depth:
            raise ValueError(f"column key {parts} is deeper than the frame ({_depth} levels)")
        return tuple(parts) + ("",) * (_depth - len(parts))

    # --- CALO VARIATIONS ---
    det = ph.loadbranches(f["recTree"], ["rec.hdr.det"]).rec.hdr.det
    det = "SBND" if (1 == det.unique()) else "ICARUS"

    hdrdf = make_mchdrdf(f)
    ismc = hdrdf.ismc.iloc[0]

    for plane in [0, 1, 2]:
        # Load track hits for this plane
        trkhitdf = make_trkhitdf(f, plane)

        # The hits are NOT FV-filtered here.  That filter used to run on this
        # line, and it was the last thing holding the recomputed chi2 away from
        # the CAF's: LArSoft applies no such cut, so filtering first guaranteed a
        # different hit set.  It was also the wrong object to cut on -- the FV
        # box is a 10 cm fiducial inset for choosing tracks, not a hit-quality
        # mask, and every hit it removed is a good hit inside the active volume.
        #
        # Two things it actively broke.  firsthit/lasthit are set in
        # make_trkhitdf from the unfiltered hit ordering, so deleting the hit
        # carrying lasthit left no survivor carrying it: ~lasthit then excluded
        # nothing and the new boundary hit, precisely the one LArSoft threw out,
        # was kept.  That fired on 76.4% of uncontained tracks.  And for a track
        # leaving the volume the cut deletes exactly the last 26 cm the chi2 sum
        # is built from, so 60.6% of those came back NaN in every varied column.
        # Dropping it gives 43.7% more tracks a defined chi2.
        #
        # Containment is still a selection.  It is applied downstream on the
        # track's start and end, which is where it belongs and is unaffected.
        #
        # "cv" is NOT skipped, and that is the point of this loop covering nine
        # entries rather than eight.  The eight varied chi2 are recomputed here
        # from hits, under chi2pid.chi2's rr < 26 / ~firsthit / ~lasthit cuts.
        # The nominal chi2 they used to be compared against is not: it is read
        # off the CAF, where LArSoft computed it at reco time.  So the ratio
        # nominal -> varied mixed a calorimetry effect with a hit-selection
        # difference, and the two are not separable after the fact.  Running cv
        # through this same path gives the variations a like-for-like
        # denominator.
        #
        # With the hit sets now identical and the constants matched, cv also
        # equals the CAF nominal to float precision on every track
        # (notebook_chi2pid C13).  chi2_{par}_I{plane} and its _cv are then the
        # same number computed twice, once by LArSoft and once here, and that is
        # the reason to keep both: any divergence is a regression in this chain
        # and can be nothing else.
        #
        # Costs one extra dE/dx recomputation per plane, i.e. 9/8 of the previous
        # work in this loop, and 12 columns on the track table.
        for var_name, calo_params in chi2pid.CALO_VARIATIONS.items():
            # charge="dqdx" reads the calorimetry's own dQ/dx off the CAF.
            # Rebuilding it as integral/pitch cannot reproduce it: Gnocchi runs
            # with ChargeMethod 3 and sums the integrals over the hit's snippet,
            # and only the primary hit's integral is in the file.
            dedx_redo = chi2pid.dedx(trkhitdf, gain=det, calibrate=det, plane=plane, isMC=ismc, new_calo_params=calo_params, charge="dqdx")
            trkhitdf["dedx_redo"] = dedx_redo

            # Recalculate Chi2 for Muon, Proton, Kaon and Pion
            for par in ['muon', 'proton', 'kaon', 'pion']:
                this_chi2_new, _ = chi2pid.chi2par(trkhitdf, dedxname="dedx_redo", par=par)
                pandora_df[_key("pfp", "trk", "chi2pid", f"I{plane}", f"chi2_{par}_{var_name}")] = this_chi2_new.fillna(0.)
    # -----------------------

    flat_df = pandora_df.loc[:, ~pandora_df.columns.duplicated()].reset_index(level=pfp_idx_col)

    # Rename complex awkward tuples to flat sensible names
    pfp_col_key = _key(pfp_idx_col)
    col_map = {
        _key("pfp", "trk", "start", "x"): "start_x",
        _key("pfp", "trk", "start", "y"): "start_y",
        _key("pfp", "trk", "start", "z"): "start_z",
        _key("pfp", "trk", "end", "x"): "end_x",
        _key("pfp", "trk", "end", "y"): "end_y",
        _key("pfp", "trk", "end", "z"): "end_z",
        _key("pfp", "trackScore"): "trackScore",
        _key("pfp", "trk", "len"): "trk_len",
        _key("pfp", "trk", "chi2pid", "I0", "chi2_kaon"): "chi2_kaon_I0",
        _key("pfp", "trk", "chi2pid", "I0", "chi2_muon"): "chi2_muon_I0",
        _key("pfp", "trk", "chi2pid", "I0", "chi2_proton"): "chi2_proton_I0",
        _key("pfp", "trk", "chi2pid", "I1", "chi2_kaon"): "chi2_kaon_I1",
        _key("pfp", "trk", "chi2pid", "I1", "chi2_muon"): "chi2_muon_I1",
        _key("pfp", "trk", "chi2pid", "I1", "chi2_proton"): "chi2_proton_I1",
        _key("pfp", "trk", "chi2pid", "I2", "chi2_kaon"): "chi2_kaon_I2",
        _key("pfp", "trk", "chi2pid", "I2", "chi2_muon"): "chi2_muon_I2",
        _key("pfp", "trk", "chi2pid", "I2", "chi2_proton"): "chi2_proton_I2",
        _key("pfp", "trk", "chi2pid", "I0", "chi2_pion"): "chi2_pion_I0",
        _key("pfp", "trk", "chi2pid", "I1", "chi2_pion"): "chi2_pion_I1",
        _key("pfp", "trk", "chi2pid", "I2", "chi2_pion"): "chi2_pion_I2",
        _key("pfp", "trk", "rangeP", "p_pion"): "trk_rangeP_pion",
        _key("pfp", "trk", "truth", "p", "pdg"): "truth_pdg",
        _key("pfp", "trk", "truth", "p", "G4ID"): "truth_G4ID",
        _key("pfp", "trk", "truth", "p", "parent"): "truth_parent",
        _key("pfp", "trk", "truth", "p", "interaction_id"): "truth_interaction_id",
        _key("pfp", "trk", "truth", "p", "genE"): "truth_genE",
        _key("pfp", "trk", "truth", "p", "start", "x"): "truth_start_x",
        _key("pfp", "trk", "truth", "p", "start", "y"): "truth_start_y",
        _key("pfp", "trk", "truth", "p", "start", "z"): "truth_start_z",
        _key("pfp", "trk", "truth", "p", "end", "x"): "truth_end_x",
        _key("pfp", "trk", "truth", "p", "end", "y"): "truth_end_y",
        _key("pfp", "trk", "truth", "p", "end", "z"): "truth_end_z",
        pfp_col_key: "pfp_index",
    }

    # Add systematic columns to col_map
    # Includes "cv", matching the loop above.  chi2_{par}_I{plane}_cv does not
    # collide with the CAF's chi2_{par}_I{plane}: both are kept, deliberately.
    for var_name in chi2pid.CALO_VARIATIONS.keys():
        for plane in [0, 1, 2]:
            col_map[_key("pfp", "trk", "chi2pid", f"I{plane}", f"chi2_muon_{var_name}")] = f"chi2_muon_I{plane}_{var_name}"
            col_map[_key("pfp", "trk", "chi2pid", f"I{plane}", f"chi2_proton_{var_name}")] = f"chi2_proton_I{plane}_{var_name}"
            col_map[_key("pfp", "trk", "chi2pid", f"I{plane}", f"chi2_kaon_{var_name}")] = f"chi2_kaon_I{plane}_{var_name}"
            col_map[_key("pfp", "trk", "chi2pid", f"I{plane}", f"chi2_pion_{var_name}")] = f"chi2_pion_I{plane}_{var_name}"

    cols_to_keep = {k: v for k, v in col_map.items() if k in flat_df.columns}
    clean_df = flat_df[list(cols_to_keep.keys())].copy()
    clean_df.columns = list(cols_to_keep.values())

    # A track with no backtracked particle carries INT_MIN in every truth
    # column, which is a number a histogram or a mean will happily consume.
    # Map it onto sentinels that cannot be mistaken for data:
    #   truth_pdg  0   no particle has pdg 0, so this is THE unmatched test
    #   truth_G4ID 0   G4 numbers particles from 1
    #   truth_interaction_id -1  the CAF already uses -1 for a cosmic-origin
    #                            particle, and "not from a neutrino
    #                            interaction" is the only distinction that
    #                            matters downstream; truth_pdg == 0 still
    #                            separates unmatched from genuinely cosmic
    #   truth_parent -1  NOT 0: G4 uses parent == 0 for a primary, so mapping
    #                    unmatched onto 0 would make an unmatched track read as
    #                    the child of a primary. -1 is already the CAF's
    #                    "not from a neutrino interaction" value.
    #   truth_genE NaN  already NaN out of the CAF, left alone
    #   truth_start_*, truth_end_*  float, and left alone for the same reason
    #                    as genE. If a build ever does put INT_MIN in them the
    #                    failure mode is safe: util.InFV(-2.1e9) is False, so
    #                    an unmatched track reads as uncontained and cannot be
    #                    labelled signal.
    _unmatched = np.iinfo(np.int32).min
    for _col, _fill in (("truth_pdg", 0), ("truth_G4ID", 0), ("truth_parent", -1),
                        ("truth_interaction_id", -1)):
        if _col in clean_df.columns:
            clean_df[_col] = clean_df[_col].replace(_unmatched, _fill).astype(int)

    # Range-momentum under the KAON hypothesis, which the CAF does not carry:
    # caf::SRTrkRange has p_muon/p_pion/p_proton and no kaon slot, and adding one
    # upstream would mean a coordinated sbnobj/sbnanaobj schema change plus a
    # full reco reprocessing. It is unnecessary -- sbncode's own producer takes
    # exactly one track-dependent input, recob::Track::Length(), which is trk_len
    # here. See makedf/range_momentum.py for the derivation and the fidelity
    # notes; NaN below ~3.29 cm, where the muon CSDA spline would be
    # extrapolated off the bottom of its table.
    if "trk_len" in clean_df.columns:
        clean_df["trk_rangeP_kaon"] = range_momentum.p_kaon(clean_df["trk_len"].to_numpy(float))

    # Restore index: entry and slice index are already in the index, so just append pfp_index
    clean_df = clean_df.set_index("pfp_index", append=True).sort_index()

    return clean_df, slice_idx_names


# --- shared base-track extraction -------------------------------------------
# make_track_df and make_pair_df both need the base track frame, and
# analysis_village/kaon/configs/kaon_config.py lists BOTH in DFS. Without a
# cache the whole extraction -- the hit load, 24 dedx computations and 96 chi2
# refits -- runs twice per file. Measured on 4 flatcafs: 8.47 s for both stages
# against 5.03 s sharing one extraction, i.e. ~41% of the track stages and ~31%
# of total df-making runtime.
#
# A single entry is enough: pyanalib.ntuple_glob._loaddf gives each worker
# process one file at a time and applies DFS in order, so make_track_df fills
# the cache and make_pair_df hits it. Module state is per-process under
# multiprocessing, so workers cannot collide.
#
# The cache holds a strong reference to the file object as its key. Keying on
# id() instead would risk a stale hit if the file were freed and a later object
# reused the address. Memory is bounded to one file's frame; call
# clear_track_cache() to release it early.
#
# NOTE: make_track_df returns the cached frame itself, not a copy, so callers
# must not mutate it in place -- make_pair_df would then see the mutation. It
# selects columns by name and copies, so merely *adding* columns is harmless.
_TRACK_CACHE = None  # (file_object, (track_df, slice_idx_names))


def _base_track_df(f):
    """_extract_base_track_df, memoised on the file object."""
    global _TRACK_CACHE
    if _TRACK_CACHE is not None and _TRACK_CACHE[0] is f:
        return _TRACK_CACHE[1]
    result = _extract_base_track_df(f)
    _TRACK_CACHE = (f, result)
    return result


def clear_track_cache():
    """Drop the cached base track frame, releasing it and the file reference."""
    global _TRACK_CACHE
    _TRACK_CACHE = None


def make_track_df(f: dict) -> pd.DataFrame:
    """
    Extracts Single-Track features for the Track BDT evaluation.
    Contains 1 row per PFP track, preserving all calorimetry systematic variations.

    Shares its extraction with make_pair_df; see _base_track_df. The returned
    frame is the cached object -- do not modify it in place.
    """
    track_df, _ = _base_track_df(f)
    return track_df


def make_pair_df(f: dict, proximity_cm: float = 5.0) -> pd.DataFrame:
    """
    Extracts PFP pair features from a flatcaf file for Pair BDT training and evaluation.

    proximity_cm is the widest separation the products will ever hold, not the
    analysis cut. Kept loose deliberately: dist_to_parent_end is exported per
    pair, so a downstream selection can tighten it, but it can never recover
    pairs the products never built. The pair BDT selection cuts at 2.0 cm.
    """
    track_df, slice_idx_names = _base_track_df(f)
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
        if c.startswith(("chi2", "truth_")):
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
        if c.startswith(("chi2", "truth_")):
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


# --- per-pfp direction and hierarchy ----------------------------------------
# The nine per-pfp branches the track table does not carry, for studies that need an
# angle or Pandora's parentage. Written as its own table and its own pass because it
# reads no hits and recomputes no chi2: the whole thing is a branch read, so it costs a
# small fraction of a reco production and does not have to wait for one.
#
# Motivating case is a proton control sample -- a nu_mu CC 1mu1p topology, the long track
# tagging the event and the short contained one at the same vertex being a proton, used to
# measure the chi2 response of a *known* particle without using chi2 to find it. Everything
# else that needs is already exported: `track` has the lengths, both end points,
# trackScore, all nine calorimetry universes and the backtracked truth, and `slice` has the
# reconstructed vertex. Measured on kcv, that reaches 84% proton purity on the short leg
# and 94% once a transverse-imbalance cut is added -- and the imbalance is what wants
# directions, because the chord between the two end points sits over 10 degrees off
# `trk.dir` for 13% of 20-50 cm tracks and 21% of tracks past a metre.
#
# `rangeP.p_muon` and `rangeP.p_proton` are deliberately NOT here: range_momentum.p_muon
# and p_proton reproduce the CAF's own columns from trk_len to 7e-08 in ratio, so storing
# them would store a function of a column already exported.
#
# No cut is applied beyond the clear-cosmic drop that `track` also applies. A gated
# variant was tried and measured first -- gating on the 1mu1p topology keeps 17-27% of
# track rows, because two or three primary tracks with a 30 cm tag is a common topology
# rather than a rare one -- so it would have bounded every later study of its own gate to
# save less than a factor of four. Nine narrow columns need no such bargain.

#: ``loadbranches`` output column -> exported name.  Keyed by the dotted name rather than
#: by a tuple: ``loadbranches`` sets column arity from the deepest branch requested and
#: pads the rest, so adding a branch one level deeper renumbers every tuple key. Joining
#: the non-empty parts of each tuple is arity-independent and needs no maintenance.
PFP_GEOM_COLUMNS = {
    "pfp.trk.dir.x": "dir_x",
    "pfp.trk.dir.y": "dir_y",
    "pfp.trk.dir.z": "dir_z",
    "pfp.trk.dir_end.x": "dir_end_x",
    "pfp.trk.dir_end.y": "dir_end_y",
    "pfp.trk.dir_end.z": "dir_end_z",
    "pfp.id": "pfp_id",
    "pfp.parent": "pfp_parent",
    "pfp.parent_is_primary": "parent_is_primary",
}

#: Stored narrow. The CAF holds the directions as float32 and the hierarchy as int32, so
#: float64 would double the product to store nothing -- and this table's whole argument
#: for being unfiltered is that it is cheap.
PFP_GEOM_DTYPES = {
    "dir_x": "float32", "dir_y": "float32", "dir_z": "float32",
    "dir_end_x": "float32", "dir_end_y": "float32", "dir_end_z": "float32",
    "pfp_id": "int32", "pfp_parent": "int32", "parent_is_primary": "int8",
}

#: No parent recorded. -1 rather than 0, which Pandora uses for a real id -- the same trap
#: ``truth_parent`` documents in ``_extract_base_track_df``.
PFP_NO_PARENT = -1


def make_pfp_geom_df(f: dict) -> pd.DataFrame:
    """One row per pfp: the track direction at both ends, and its place in the hierarchy.

    Keyed as ``make_track_df`` is -- ``(entry, rec.slc..index, pfp_index)``, with the loader
    adding ``__ntuple`` -- so within one processing ``track.join(geom)`` is the whole join.
    **Across** processings it is not: ``__ntuple`` is assigned from the order a job read its
    inputs, so a geom pass and a reco pass number the same file differently. That is why
    ``hdr`` and ``file`` ship alongside, exactly as they do for ``kaon_syst_config``: the key
    that survives is ``(file_key, entry)``.

    Clear cosmics are dropped, matching ``make_track_df`` -- a row ``track`` does not have
    would join to nothing anyway, and it is a large fraction of the pfps. A shower-like pfp
    is **kept**, with NaN directions, because what else Pandora put in the slice is part of
    how a topology cut gets checked.
    """
    branches = [f"rec.slc.reco.{name}" for name in PFP_GEOM_COLUMNS]
    frame = ph.loadbranches(f["recTree"], branches).rec.slc.reco
    if frame.empty:
        return pd.DataFrame()

    frame.columns = ['.'.join(n for n in c if n != '') for c in frame.columns]
    frame = frame[list(PFP_GEOM_COLUMNS)].rename(columns=PFP_GEOM_COLUMNS)

    slc_df = ph.loadbranches(f["recTree"], ['rec.slc.is_clear_cosmic']).rec.slc
    if isinstance(slc_df.columns, pd.MultiIndex):
        slc_df.columns = ["_".join([str(c) for c in col if c]).strip() for col in slc_df.columns.values]
    is_cosmic = (slc_df['is_clear_cosmic'] == 1).reindex(frame.index.droplevel(-1))
    frame = frame[~is_cosmic.to_numpy()]
    if frame.empty:
        return pd.DataFrame()

    # INT_MIN in an int column is a number a histogram or a mean would consume.
    unmatched = np.iinfo(np.int32).min
    for col, fill in (("pfp_id", PFP_NO_PARENT), ("pfp_parent", PFP_NO_PARENT),
                      ("parent_is_primary", 0)):
        frame[col] = frame[col].replace(unmatched, fill).fillna(fill)

    frame = frame.astype(PFP_GEOM_DTYPES)
    frame.index = frame.index.set_names([*frame.index.names[:-1], "pfp_index"])
    return frame.sort_index()
