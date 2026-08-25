from . import calo
import pandas as pd
import numpy as np
import sqlite3
import uproot

larsoft_data_v = "v1_02_02"
icarus_data_v = "v10_06_06"
sbnd_data_v = "v01_42_00" # the version sbndcode v10_14_02 reco2 read

rr_max_cut_chi2 = 26. ## for resolving MC's hit RR cut, after fixing the issue, put this value to 26.

#### == use pandora_df_calo_update to apply these changes
ICARUS_CALO_PARAMS = {
    "alpha_emb": 0.904,
    "beta_90": 0.204,
    "R_emb": 1.25,
    "gains": [0.016751, 0.012755, 0.012516],
    "c_cal_frac": [1., 1., 1.],
}

#### Default
#### == For each element, the first entry is for MC and the second entry is for the data
SBND_CALO_PARAMS = {
    "alpha_emb": [0.904, 0.9366111025888879],
    "beta_90": [0.204, 0.1835451204193374],
    "R_emb": [1.25, 1.567685536351266],
    "gains": [
        ## MC: sbndcode calorimetry_sbnd.fcl, sbnd_calorimetryalgmc.CalAreaConstants
        [0.02052, 0.02044, 0.02019], ## MC
        ## Data: the reco2 fcl's own gains, with NOTHING FITTED -- this is "setting F".
        ##
        ## It was [0.0211835, 0.0209689, 0.0205106], i.e. these times a fitted
        ## plane-independent 0.9753 (V3.51).  That remainder is REMOVED here, because the
        ## per-(itpc, plane) level in SBND_TPC_PLANE_LEVEL replaces it and was derived
        ## against setting F's charge.  Leaving the 0.9753 in would make the data charge
        ## entering sbnd_calo_chain 2.53% larger than the charge the level was measured
        ## on, so the level would under-correct data by 2.53% on every plane -- silently.
        ## Do not restore it without re-deriving the level.  CALO.87, CALO.105.
        [0.02172, 0.02150, 0.02103]], ## Data
    "c_cal_frac": [1., 1., 1.],
    "etau": [35., 35.], ## first value for MC and second value for data
}


def _vary(params, key, delta):
    """Offset a [MC, data] calo parameter by delta, preserving each centre.

    dedx() selects params[key][0] for MC and params[key][1] for data, and the
    two centres differ. Hardcoding the MC centre in both slots turns the data
    "variation" into a shift of the central value -- for beta_90 and R_emb it
    puts both +1 and -1 sigma on the same side of the data nominal, which
    breaks any symmetrised covariance built from them.
    """
    base = params[key]
    return {**params, key: [base[0] + delta, base[1] + delta]}


# calo variations
# variations on recombination parameters are taken from the ICARUS measurement uncertainties
# NB: c_cal_frac is indexed BY PLANE (3 entries), unlike alpha_emb/beta_90/R_emb
# which are indexed BY isMC (2 entries), so _vary() does not apply to it.
CALO_VARIATIONS = {
    "cv": SBND_CALO_PARAMS,
    "ccal_p": {**SBND_CALO_PARAMS, "c_cal_frac": [1.02, 1.02, 1.02]},
    "ccal_m": {**SBND_CALO_PARAMS, "c_cal_frac": [0.98, 0.98, 0.98]},
    "alpha_p": _vary(SBND_CALO_PARAMS, "alpha_emb", +0.008),
    "alpha_m": _vary(SBND_CALO_PARAMS, "alpha_emb", -0.008),
    "beta_p":  _vary(SBND_CALO_PARAMS, "beta_90",   +0.008),
    "beta_m":  _vary(SBND_CALO_PARAMS, "beta_90",   -0.008),
    "R_p":     _vary(SBND_CALO_PARAMS, "R_emb",     +0.02),
    "R_m":     _vary(SBND_CALO_PARAMS, "R_emb",     -0.02),
}


def chi2(hitdf, exprr, expdedx, experr, dedxname="dedx"):
    dedx_exp = pd.cut(hitdf.rr, exprr, labels=expdedx).astype(float)
    dedx_err = pd.cut(hitdf.rr, exprr, labels=experr).astype(float)

    # Evaluated at the EXPECTATION, not at the measurement.  A pull is a
    # standardised residual only if its denominator is the sampling spread given
    # the truth: (x-mu)/sigma(mu) is unit-normal, (x-mu)/sigma(x) is not --
    # conditioning on the realised value shrinks upward fluctuations and
    # magnifies downward ones.  Measured on the 1mu1p control protons with only
    # this argument changed, the pooled pull width falls 5.185 -> 4.056, the skew
    # halves, and the data/MC chi2 ratio falls 1.731 -> 1.465.  It moves a
    # data/MC ratio despite being applied to both samples because a denominator
    # conditioned on the observation is not common-mode -- it inherits each
    # sample's own fluctuation distribution.  notebook_v3 CALO.64-CALO.65.
    dedx_res = (0.04231 + 0.0001783*dedx_exp**2)*dedx_exp

    v_chi2 = (hitdf[dedxname] - dedx_exp)**2 / (dedx_err**2 + dedx_res**2)

    when_chi2 = (hitdf.rr < rr_max_cut_chi2) & ~hitdf.firsthit & ~hitdf.lasthit & (hitdf[dedxname] < 1000.)

    chi2_group = v_chi2[when_chi2].groupby(level=list(range(hitdf.index.nlevels-1)))

    return chi2_group.sum() / chi2_group.size(), chi2_group.size()

def chi2u(hitdf, dedxname="dedx"):
    return chi2(hitdf, muon_rr, muon_dedx, muon_yerr, dedxname)

def chi2p(hitdf, dedxname="dedx"):
    return chi2(hitdf, proton_rr, proton_dedx, proton_yerr, dedxname)

def chi2k(hitdf, dedxname="dedx"):
    return chi2(hitdf, kaon_rr, kaon_dedx, kaon_yerr, dedxname)

def chi2pi(hitdf, dedxname="dedx"):
    return chi2(hitdf, pion_rr, pion_dedx, pion_yerr, dedxname)

def chi2par(hitdf, dedxname="dedx", par=""):
    if par == "muon":
        return chi2u(hitdf, dedxname)
    elif par == "proton":
        return chi2p(hitdf, dedxname)
    elif par == "kaon":
        return chi2k(hitdf, dedxname)
    elif par == "pion":
        return chi2pi(hitdf, dedxname)
    else:
        raise ValueError(f"Invalid par={par!r}. Expected 'muon', 'proton', 'kaon', or 'pion'.")

def chi2_ndof(hitdf):
    when_chi2 = (hitdf.rr < rr_max_cut_chi2) & ~hitdf.firsthit & ~hitdf.lasthit & (hitdf.dedx < 1000.)
    chi2_group = v_chi2[when_chi2].groupby(level=list(range(hitdf.index.nlevels-1)))

    return chi2_group.size()

def dqdx(dqdxdf, gain=None, calibrate=None, isMC=False, charge="integral"):
    """charge selects where SBND's dQ/dx comes from.

    "integral"  rebuild it as integral/pitch and apply the yz map (the default)
    "dqdx"      take the calorimetry's own dQ/dx off the CAF, yz already in it
    """
    if calibrate == "ICARUS": 
        # get raw dqdx
        dqdx = dqdxdf.integral / dqdxdf.pitch

        # compute y-scale
        ybin = _yz_ybin(dqdxdf.y, yz_ybin)
        zbin = _yz_zbin(dqdxdf.z, yz_zbin)
        iov = _yz_iov(dqdxdf.run)
        itpc = dqdxdf.tpc // 2 + dqdxdf.cryo*2
        plane = dqdxdf.plane

        yzdf = pd.DataFrame({"ybin": ybin, "zbin": zbin, "itpc": itpc, "plane": plane, "iov": iov})
        yz_scale = yzdf.merge(IC_yz_cal_df, on=["iov", "itpc", "plane", "ybin", "zbin"], how="left", validate="many_to_one").scale
        yz_scale[yz_scale == -999.000000] = 1.
        yz_scale = np.clip(yz_scale, 0.7, 1.3).fillna(1)
        yz_scale.index = dqdxdf.index

        # compute lifetime correction
        iov = _etau_iov(dqdxdf.run)
        etaudf = pd.DataFrame({"itpc": itpc, "iov": iov})
        etau = etaudf.merge(IC_etau_cal_df, on=["iov", "itpc"], how="left", validate="many_to_one").etau
        etau = etau.fillna(np.inf)
        etau.index = dqdxdf.index
        etau[dqdxdf.run == 1] = 3.5e3 # set MC lifetime to MC default

        # compute TPC scale
        iov = _tpc_iov(dqdxdf.run)
        tpcdf = pd.DataFrame({"itpc": itpc, "plane": plane, "iov": iov})
        tpc_scale = tpcdf.merge(IC_tpc_cal_df, on=["iov", "itpc", "plane"], how="left", validate="many_to_one").scale
        tpc_scale.index = dqdxdf.index

        # apply the corrections
        t0 = 0 # assume in time
        tick_period = 0.4 # us
        tanode = 850 # ticks
        tdrift = dqdxdf.t*tick_period - t0 - tanode*tick_period

        dqdx = dqdx * tpc_scale * np.exp(tdrift / etau) / yz_scale
    elif calibrate == "SBND": # TODO: add calibrations?
        if charge == "dqdx":
            # The CAF's dqdx branch is the calorimetry's own dQ/dx, but it is
            # not integral/pitch: GnocchiCalorimetry runs with ChargeMethod 3,
            # which sums the hit integrals while the CAF stores only the
            # primary hit's integral, so the sum cannot be rebuilt from the
            # file.  NormalizeYZ has also already been applied to it.
            dqdx = dqdxdf.dqdx
        else:
            # get raw dqdx
            dqdx  = dqdxdf.integral / dqdxdf.pitch

        this_yz_cal_df = SBND_yz_cal_mc_df   if isMC else SBND_yz_cal_data_df
        this_yz_zbin = yz_zbin_sbnd_mc if isMC else yz_zbin_sbnd_data
        this_yz_ybin = yz_ybin_sbnd_mc if isMC else yz_ybin_sbnd_data
        this_etau_df   = SBND_etau_cal_mc_df if isMC else SBND_etau_cal_data_df

        # compute y-scale
        ybin = _yz_ybin(dqdxdf.y, this_yz_ybin)
        zbin = _yz_zbin(dqdxdf.z, this_yz_zbin)
        bin_test_df = pd.DataFrame({"y":dqdxdf.y, "z": dqdxdf.z, "ybin": ybin, "zbin": zbin, "dqdx": dqdx})
        if isMC:
            dqdxdf['iov'] = 0 ## FIXME: once SBND has time dep. calo, it should be updated
        else:
            dqdxdf['iov'] = _etau_iov_sbnd(dqdxdf.run)

        iov = dqdxdf.iov ## FIXME: once SBND has time dep. calo, it should be updated
        itpc = dqdxdf.tpc
        plane = dqdxdf.plane

        if charge == "dqdx":
            yz_scale = pd.Series(1., index=dqdxdf.index)
        else:
            yzdf = pd.DataFrame({"ybin": ybin, "zbin": zbin, "itpc": itpc, "plane": plane, "iov": iov})
            yzdf['iov'] = 0 ## yzdf iov ==0 for MC and data
            yz_scale = yzdf.merge(this_yz_cal_df, on=["iov", "itpc", "plane", "ybin", "zbin"], how="left", validate="many_to_one").scale
            yz_scale[yz_scale < 1e-3] = 1.
            yz_scale = yz_scale.fillna(1)
            yz_scale.index = dqdxdf.index
        #dqdxdf['yz_scale'] = yz_scale ## FIXME

        #yzdf['rr'] = dqdxdf.rr
        #yzdf['scale'] = yz_scale
        #print(yzdf[yzdf.rr < 26.].head(50))
        # compute lifetime correction
        iov = dqdxdf.iov ## FIXME: once SBND has time dep. calo, it should be updated
        if isMC:
            etaudf = pd.DataFrame({"itpc": itpc, "iov": iov})
            etau = etaudf.merge(this_etau_df, on=["iov", "itpc"], how="left", validate="many_to_one").etau
        else:
            etaudf = pd.DataFrame({"iov": iov})
            etau = etaudf.merge(SBND_etau_cal_data_df, on=["iov"], how="left", validate="many_to_one")
            etau['etau'] = np.where(dqdxdf.tpc == 0, etau.etau_E, etau.etau_W)
            etau = etau.etau

        etau = etau.fillna(np.inf)
        etau.index = dqdxdf.index

        # apply the corrections
        t0 = 0 # assume in time
        # 0.205 ms, see detectorclocks_sbnd.fcl TriggerOffsetTPC = -0.205e3 us,
        # which is what trigger_offset() returns in CalorimetryAlg::LifetimeCorrection
        tdrift = dqdxdf.t / 2000. - 0.205
        etau_corr = np.exp(tdrift / etau)
        #dqdxdf['etau_corr'] = etau_corr ## FIXME
        dqdx = dqdx * etau_corr * yz_scale

    else: # if not specified, rely on input calibration
        dqdx = dqdxdf.dqdx

    # apply gain
    if gain == "ICARUS":
        gains = ICARUS_CALO_PARAMS["gains"]
        gain_perhit = pd.Series(1.0, dqdxdf.index)
        for iplane, g in enumerate(gains):
            gain_perhit[dqdxdf.plane == iplane] = 1.0/g
    elif gain == "SBND": # TODO
        gains = SBND_CALO_PARAMS["gains"][0] if isMC else SBND_CALO_PARAMS["gains"][1]
        gain_perhit = pd.Series(1.0, dqdxdf.index)
        for iplane, g in enumerate(gains):
            gain_perhit[dqdxdf.plane == iplane] = 1.0/g
    else:
        gain_perhit = 1

    return dqdx*gain_perhit

##############################
# THE SBND CALORIMETRY CHAIN (derived by kaonana.calo, CALO.91-CALO.104)
#
# Order:  level -> smear -> turnon -> invert.  NOT FREE.
#   `absorb` is RETIRED (CALO.143) -- see sbnd_calo_chain.
#   * `level` is a data-side gain, so it acts first, before anything charge-dependent.
#   * `reco` is a reconstruction defect and applies to BOTH samples.
#   * `saturation` and `smear` are MC-ONLY.  Applying them to data is not a
#     bug, it is a wrong measurement.
#   * the turn-on is a factor on the CALIBRATED charge, so applying it to
#     the raw charge evaluates the knee in the wrong variable.
#   * everything acts on CHARGE; recombination is inverted once, at the end.
#
# Two of the five carry no fitted numbers at all -- the absorbing factor and the
# correlated field -- so a patch with only the constant blocks below is wrong by
# ~5% in dE/dx and will look plausible.
#
# See docs/patches/cafpyana_sbnd_calo_chain.patch for what each rung is and why.
##############################

#: The charge the reconstruction lost, per PLANE (TPC-independent to 0.2%).  BOTH SAMPLES.
#: Fitted on Geant4 truth: closes MC to its own truth 0.955 -> 0.998, and the closure holds
#: for protons and muons alike (0.93-0.94), which is what makes it a chain effect.
#:
#: These are the POOLED per-plane fits, not the mean of the two per-TPC fits.  A mean of two fits
#: is not a fit -- it lands ~0.2% off in the correction and, because the saturation's knee is a
#: kink, up to tens of percent on individual near-knee hits.  Re-derive with
#: `--planes 0 1 2` (no `--tpcs`) if these are ever refreshed.  Track-weighted (CALO.107).
SBND_RECO_CORRECTION = {
    0: dict(q0=130000, a0=0.0466985792, a1=-0.0651132686, a2=-0.0278776686),
    1: dict(q0=130000, a0=0.0468439409, a1=-0.0878337296, a2=-0.0281752486),
    2: dict(q0=130000, a0=0.0289619024, a1=-0.040410619, a2=-0.0427837696),
}

#: A DATA-side dQ/dx scale per (itpc, plane).  MC is 1.0 by construction: this is a
#: calibration, not a correction to the simulation.  It cannot live in `gains` or
#: `c_cal_frac`, which are indexed by plane alone.
#:
#: Almost all of it is (1, 0) at 4.8%, where the proton and stopping-muon legs agree to
#: 0.07%.  *** (0, 1) is deliberately ~1: *** its two legs disagree at chi2 5.4 on one
#: degree of freedom (protons +2.6%, muons -2.0%), so no correction is applied there.  Do
#: not replace it with the proton-only number -- that was withdrawn at CALO.102.
SBND_TPC_PLANE_LEVEL = {
    (0, 0): 0.988313,
    (1, 0): 0.963561,
    (0, 1): 0.995309,  # species chi2 5.4: no correction, deliberately.  See the header.
    (1, 1): 0.998391,
    (0, 2): 0.999731,
    (1, 2): 0.990689,
}

#: The above-knee charge factor on MC, per (itpc, plane).
#:
#: *** FROZEN above q_max, and that is not optional. ***  Unbounded, this rung makes the
#: per-track chi2 shape WORSE than no correction at all -- delta 0.125 against an
#: uncorrected 0.104 -- and 1.5% of hits decide it.  Freeze, do not clip to 1: above the
#: last fitted cell the correction is real but unmeasured, and holding the last measured
#: value is the weaker of the two assumptions.
#:
#: b1 is -0.14 on both induction planes and -0.35 on the collection plane, a factor 2.5 at
#: 6.4 sigma.  The least trustworthy rung in the chain: above 12 MeV/cm on a stopping proton
#: is inside ~0.5 cm of the track end, where MC carries an end-placement error, and a charge
#: saturation and a misplaced end bias the same ratio in the same hits.  Nothing in the
#: control sample separates them.
SBND_SATURATION = {
    (0, 0): dict(knee=165052, b0=0.0135411, b1=-0.162002, b2=-0.05443,
                 p0=0.42, q_max=254981),
    (1, 0): dict(knee=157153, b0=0.0042521, b1=-0.123429, b2=-0.0486317,
                 p0=0.42, q_max=245650),
    (0, 1): dict(knee=153346, b0=0.0202639, b1=-0.168899, b2=-0.0339018,
                 p0=0.42, q_max=258987),
    (1, 1): dict(knee=153346, b0=0.00634117, b1=-0.101189, b2=-0.0432164,
                 p0=0.42, q_max=246092),
    (0, 2): dict(knee=191213, b0=-0.00638476, b1=-0.328385, b2=-0.000802373,
                 p0=0.42, q_max=278890),
    (1, 2): dict(knee=195959, b0=0.008889, b1=-0.400159, b2=-0.077716,
                 p0=0.42, q_max=281817),
}

#: The correlated smear per (itpc, plane).  MC only, and STOCHASTIC.
#:
#: THREE BASES ARE SUPPORTED AND THE KEY NAMES SELECT ONE:
#:   `amplitude`      -- a fractional CHARGE wobble, used as it stands.        <-- SHIPPED
#:   `amplitude_dedx` -- a fractional dE/dx wobble, divided by each hit's own
#:                       `sbnd_amplification` before it is applied.
#:   `amplitude_dedx` + `power` + `rr0` + `rr_range`
#:                    -- the same, times (rr/rr0)**-power frozen outside the range.
#: Give exactly one amplitude key.  Switching basis is a constant-block edit, not a code change.
#:
#: WHY THE SHAPE.  kaonana CALO.119/120.  The dE/dx basis fixes the SPECIES dependence -- the control
#: sample's two legs disagree on the amplitude by +37.4% (9.9 sigma) in charge and +4.0% (1.4 sigma) in
#: dE/dx, the difference being the amplification -- and supplies no rr dependence.  The charge basis
#: supplies an rr dependence by ACCIDENT, because the amplification rises toward the track end, and
#: overshoots it: an effective power of ~0.135 against the ~0.083 the data wants.  So neither pure basis
#: is right.  The fitted power is +0.0639 +- 0.0098, shared by all six cells, and the shape it gives
#: rises 12.5% from rr 19 to 3 cm against the +12.3% CALO.113 measured by an unrelated route.
#:
#: *** BOTH dE/dx VARIANTS HAVE BEEN TRIED ON THE ANALYSIS AND BOTH LOST.  FLAT: REVERTED AT CALO.113.
#: WITH AN rr SHAPE: REVERTED AT CALO.125.  Median `delta` against the uncorrected baseline --
#: charge -37%/-33%/-54% on kmu/kpi/off-ramp, flat dE/dx -19%/-27%/unchanged, dE/dx+shape
#: -16%/-11%/-50%.  CALO.113 blamed the missing rr dependence; CALO.120 fitted it explicitly and
#: CALO.125 found it no better.  That diagnosis was wrong and the cause is still unknown. ***
#: Flat dE/dx is better on the statistic it is fitted to (the control sample's per-hit width closes 25%
#: better on the held-out muon leg) and WORSE on the analysis: the kmu BDT inputs went -32% -> -19%
#: against the uncorrected baseline and kpi -42% -> -27%, with 9 of 16 and 13 of 19 features degrading.
#: CALO.120 diagnosed the cause as the missing rr dependence, which `power` restores.
#:
#: KNOWN OPEN, and it is a systematic rather than a defect in this code: the muon leg agrees with the
#: proton-derived shape pooled (a0 ratio 0.991) and not per cell, and the residual is structured by TPC
#: -- 0.906 in TPC 0 against 1.094 in TPC 1, a 3.7 sigma split, unexplained (CALO.122).  Booked as an
#: antisymmetric +-9.5% on the amplitude, which is 1.3-1.6% of per-hit width within a TPC and 0.03-0.3%
#: pooled over them (CALO.123).
#:
#: The reason is a shape the charge basis gets right by accident.  The amplification RISES toward the
#: track end (1.39 -> 1.70 over rr 20 -> 4.5 cm), so a flat charge amplitude delivers a dE/dx wobble
#: that rises with it, and that is the shape the data wants -- needed [1, 1.089, 1.123] against
#: charge's [1, 1.083, 1.218] and a flat dE/dx's [1, 1, 1].  The charge basis is a working proxy for an
#: rr dependence nobody has fitted.
#: THE CONSTANTS THE KERNEL KEYS ADDED BY cafpyana_sbnd_calo_student.patch NEED.  That patch taught
#: this function to READ `nu` and `knee`; without them here the chain silently ran the Gaussian with
#: no knee -- which is the "a patch with only the code" mirror of the defect section 3 warns about.
#:
#: `nu` = 5 and `knee` = 5.6e4 from kaonana CALO.169 (`NOISE_V7`), amplitudes from the same fit.  Six
#: independent planes; the fitted amplitude over the previously shipped one runs 0.962 to 1.044, three
#: cells inside 0.3%, at pull-shape chi2/bin 0.70-1.06.  So the 11-28% by which kaonana's GAUSSIAN
#: amplitudes exceeded these was a Gaussian artefact -- a Gaussian must over-widen the core to reach a
#: given total width -- and not a deficit in what shipped.
#:
#: `length_cm` IS DELIBERATELY LEFT AT THE PREVIOUSLY MEASURED PER-CELL VALUES, not replaced by the
#: 0.679 the amplitudes were fitted at.  Those are a measurement and this fit is not a reason to
#: discard them.  The induced inconsistency is bounded and small: across the full 0.5629-0.7534 spread
#: the amplitude a fixed achieved width needs moves by 0.85-0.89% (CALO.167), because the within-track
#: projection removes only 4-12% of the field over a 5x length range.  That is inside the amplitude's
#: own 1-sigma step.  Re-fitting per length would remove it and has not been done.
SBND_MC_NOISE = {
    (0, 0): dict(amplitude=0.04719, length_cm=0.7534, nu=5.0, knee=5.6e4),
    (1, 0): dict(amplitude=0.04772, length_cm=0.5696, nu=5.0, knee=5.6e4),
    (0, 1): dict(amplitude=0.05820, length_cm=0.6700, nu=5.0, knee=5.6e4),
    (1, 1): dict(amplitude=0.05825, length_cm=0.5629, nu=5.0, knee=5.6e4),
    (0, 2): dict(amplitude=0.03900, length_cm=0.7063, nu=5.0, knee=5.6e4),
    (1, 2): dict(amplitude=0.03970, length_cm=0.5786, nu=5.0, knee=5.6e4),
}

#: What the amplitudes above replace, kept so the Gaussian configuration stays reproducible.
SBND_MC_NOISE_GAUSSIAN = {
    (0, 0): dict(amplitude=0.0490709, length_cm=0.7534),
    (1, 0): dict(amplitude=0.0457291, length_cm=0.5696),
    (0, 1): dict(amplitude=0.0580706, length_cm=0.6700),
    (1, 1): dict(amplitude=0.0583289, length_cm=0.5629),
    (0, 2): dict(amplitude=0.0389311, length_cm=0.7063),
    (1, 2): dict(amplitude=0.0384605, length_cm=0.5786),
}

#: The smallest fraction of its own charge a smeared hit may keep.  At these amplitudes it
#: needs a 20-sigma excursion and never fires; it exists because a non-positive charge maps
#: to NaN through the inversion and then poisons every quantile of the band it sits in.
SBND_CHARGE_FLOOR = 0.05


def sbnd_level_scale(itpc, plane, isMC=False):
    """The data-side per-(itpc, plane) dQ/dx scale.  1.0 everywhere for MC."""
    itpc = np.asarray(itpc, dtype=int)
    out = np.ones(len(itpc))
    if isMC:
        return out
    for tpc in np.unique(itpc):
        scale = SBND_TPC_PLANE_LEVEL.get((int(tpc), int(plane)))
        if scale is not None:
            out[itpc == tpc] = scale
    return out


def sbnd_reco_factor(charge, phi, plane):
    """The reco charge correction: ln c = a0 + a1 ln(q/q0) + a2 (cos^2 phi - 0.25)."""
    block = SBND_RECO_CORRECTION.get(int(plane))
    charge = np.asarray(charge, dtype=float)
    if block is None:
        return np.ones(len(charge))
    angular = np.cos(np.asarray(phi, dtype=float)) ** 2 - 0.25
    return np.exp(block["a0"] + block["a1"] * np.log(np.clip(charge, 1.0, None) / block["q0"])
                  + block["a2"] * angular)


def sbnd_absorbing_factor(charge, phi, efield, density, calo_params):
    """MC's charge scaled so that MC's inversion reports what DATA's would.

    Parameter-free: the closed-form ratio of the two samples' own recombination inversions,
    worth -7% to +25% in dE/dx and sign-changing with angle.  Uses `calo_params`, so in a
    varied universe it follows that universe -- there is no fit to redo.
    """
    charge = np.asarray(charge, dtype=float)
    energy = calo.recombination_cor(charge, phi, efield, density,
                                    calo_params["alpha_emb"][1], calo_params["beta_90"][1],
                                    calo_params["R_emb"][1])
    wanted = calo.recombination(np.asarray(energy, dtype=float), phi, efield, density,
                                calo_params["alpha_emb"][0], calo_params["beta_90"][0],
                                calo_params["R_emb"][0])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(charge > 0, np.asarray(wanted, dtype=float) / charge, 1.0)


def sbnd_saturation_factor(charge, pitch, itpc, plane):
    """The above-knee factor on MC's charge, FROZEN above q_max.  See SBND_SATURATION."""
    charge = np.asarray(charge, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    itpc = np.asarray(itpc, dtype=int)
    out = np.ones(len(charge))
    for tpc in np.unique(itpc):
        block = SBND_SATURATION.get((int(tpc), int(plane)))
        if block is None:
            continue
        rows = itpc == tpc
        # `b3` is the pitch tilt's own deposition dependence (CALO.132).  It multiplies an
        # UNCLIPPED ln(q/knee), which runs to -inf as the charge falls, so a ramped block is
        # frozen at the BOTTOM as well -- q_max's reason in the other direction.  The two keys
        # travel together; b3 without q_min would apply an unbounded tilt to a small hit.
        ramp = block.get("b3", 0.0)
        floor = float(block.get("q_min", 0.0))
        if ramp and not floor > 0.0:
            raise ValueError("a saturation block with b3 needs a positive q_min (CALO.133)")
        bounded = np.clip(np.where(charge[rows] > 0.0, charge[rows], floor),
                          floor, block["q_max"])
        with np.errstate(divide="ignore"):
            ratio = np.log(bounded / block["knee"])
        above = np.maximum(0.0, ratio)
        slope = block["b2"] + ramp * ratio
        out[rows] = np.exp(block["b0"] + block["b1"] * above
                           + slope * np.log(pitch[rows] / block["p0"]))
    return out


def sbnd_correlated_field(rr, track_id, length_cm, seed):
    """A unit-variance field per hit, correlated along each track as exp(-|ds| / L).

    An Ornstein-Uhlenbeck recursion in DISTANCE, not in hit index: the pitch varies
    0.35-0.50 cm across this sample, so a hit-index correlation would be a different
    physical length on every track.  Reset at every track boundary.

    A loop rather than a cumulative product on purpose -- the product form underflows,
    since exp(-0.42/0.68) to the fortieth hit is 1e-11.
    """
    rr = np.asarray(rr, dtype=float)
    track = np.asarray(track_id)
    order = np.lexsort((rr, track))
    same = np.zeros(len(order), dtype=bool)
    if len(order) > 1:
        same[1:] = track[order][1:] == track[order][:-1]
    step = np.zeros(len(order))
    step[1:] = np.abs(np.diff(rr[order]))
    weight = np.where(same, np.exp(-step / max(length_cm, 1e-9)), 0.0)
    noise = np.random.default_rng(seed).standard_normal(len(order))
    field = np.empty(len(order))
    previous = 0.0
    for index in range(len(order)):
        w = weight[index]
        previous = w * previous + np.sqrt(max(1.0 - w * w, 0.0)) * noise[index]
        field[index] = previous
    out = np.empty(len(order))
    out[order] = field
    return out



def sbnd_amplification(charge, phi, efield, density, calo_params, step=0.01):
    """`d ln(dE/dx) / d ln(q)` per hit, by a central difference through the REAL inversion.

    This is the factor a fractional charge wobble is multiplied by on its way to dE/dx, and it
    runs ~1.13 on MIP-like hits to ~1.93 at the top of the stopping-proton range.  Dividing the
    dE/dx amplitude by it per hit is what makes `SBND_MC_NOISE` deliver a CONSTANT fractional
    dE/dx width -- see the comment on that block.

    USES INDEX 0, WHICH IS MC.  Slot 0 is the MC ModBox set and slot 1 is data's.  `dedx()` below
    now uses slot 0 for BOTH samples -- one inversion, MC's ModBox, CALO.143 -- so this function
    agreeing with it is no longer a per-sample question.  This function shipped with `[1]`
    while its docstring claimed index 1 was MC -- so an MC-only correction was being scaled by
    DATA's recombination, ~1% out in the amplification and straight through to the applied
    amplitude.  It never reached a product because the dE/dx basis it was written for was
    reverted at CALO.113; the cross-check that caught it is in kaonana's test suite (CALO.124).
    Note `sbnd_absorbing_factor` above genuinely does use both slots, and its 1-then-0 order is
    deliberate: it inverts with data's constants and re-forwards with MC's.
    """
    charge = np.asarray(charge, dtype=float)
    alpha, beta, R = (calo_params["alpha_emb"][0], calo_params["beta_90"][0],
                      calo_params["R_emb"][0])
    up = calo.recombination_cor(charge * (1.0 + step), phi, efield, density, alpha, beta, R)
    down = calo.recombination_cor(charge * (1.0 - step), phi, efield, density, alpha, beta, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.asarray(up, dtype=float) / np.asarray(down, dtype=float)) / np.log(
            (1.0 + step) / (1.0 - step))
    # FLOORED AT 1.0, WHICH IS PHYSICS.  Recombination quenches more at higher deposition, so a
    # fractional charge increase always buys a LARGER fractional dE/dx increase: this derivative
    # cannot be below 1 and tends to 1 from above as the deposition falls.  5.3% of MC control hits
    # come back below 1 and 0.01% below 0.16, all at dE/dx ~1.7 MeV/cm -- below the MIP minimum,
    # where the inversion is being asked about charge no track deposits.  A 0.1 floor instead lets a
    # 6% dE/dx amplitude become a 40% charge smear on such a hit.  See kaonana CALO.124.
    return np.where(np.isfinite(out), np.maximum(out, 1.0), 1.0)


def sbnd_smear_factor(rr, track_id, itpc, plane, seed, charge, phi, efield, density,
                      calo_params):
    """The stochastic factor on MC's charge.

    Reads whichever amplitude key `SBND_MC_NOISE` carries -- `amplitude` (charge) or `amplitude_dedx`
    (divided per hit by `sbnd_amplification`), optionally with `power`/`rr0` for the shipped rr shape.
    See the comment on that block for why, and for what happened when flat dE/dx was tried.

    Two OPTIONAL keys select the heavy-tailed kernel (kaonana CALO.169, `NOISE_V7`):

      `nu`    -- degrees of freedom of a Student-t field.  A per-TRACK inverse-chi2 scale on the
                 correlated Gaussian, which is the only construction giving a Student marginal while
                 leaving the within-track correlation intact: a per-hit draw destroys the correlation,
                 one draw for the sample merely rescales it.
      `knee`  -- charge below which a hit is NOT smeared, e/cm.  Absent, every hit is smeared, which
                 is the shipped behaviour.

    A block carrying `nu` is applied as `exp(amplitude * field)`, not `1 + amplitude * field`.  That is
    REQUIRED: at nu = 5 the field has occasional large excursions and the additive form would drive
    them into SBND_CHARGE_FLOOR or negative, while the exponential is positive by construction and
    median-preserving -- which is what stops the rung injecting a level and colliding with the turn-on
    (kaonana CALO.149).  Blocks WITHOUT `nu` keep the additive form exactly, so nothing shipped moves.

    TWO THINGS TO CHECK BEFORE TRUSTING A REPROCESSING THAT USES THESE KEYS.

    (1) THE KNEE IS EVALUATED ON THE CHARGE THIS FUNCTION IS GIVEN.  `sbnd_calo_chain` calls the smear
        last, so that is the POST-SATURATION charge, while kaonana fitted `v7` in its own smear-first
        order where the knee sees the calibrated charge before the turn-on.  The two differ by the reco
        and saturation factors, so a knee ported as a bare number is evaluated in the wrong variable --
        the same defect class as section 3(c) of docs/calo_reprocessing.md.

    (2) THE SEED CONVENTIONS DIFFER ON PURPOSE.  This function mixes the cell into the seed, kaonana
        uses one seed for every cell.  So the two produce the same DISTRIBUTION and different
        REALISATIONS, and a hit-for-hit comparison only means something with the conventions aligned.
        sandbox_lowvar/port_student.py aligns them and agrees to 2e-16 on all six cells; that validates
        the construction -- field ordering, per-track scale, gate, exponential form -- and nothing about
        the streams.

    THE SEED MUST BE DERIVED FROM THE DATA, NOT FROM A STREAM.  A running counter would make
    a product depend on the order its inputs were read: two passes over the same flatcafs in
    a different order would disagree, and nothing would complain.  The caller passes a hash
    of the file's track keys; this mixes in the cell.
    """
    itpc = np.asarray(itpc, dtype=int)
    out = np.ones(len(np.asarray(rr)))
    amplification = None
    for tpc in np.unique(itpc):
        block = SBND_MC_NOISE.get((int(tpc), int(plane)))
        if block is None:
            continue
        rows = itpc == tpc
        cell_seed = (int(seed) ^ (int(tpc) << 8) ^ (int(plane) << 16)) & 0x7FFFFFFF
        field = sbnd_correlated_field(np.asarray(rr)[rows], np.asarray(track_id)[rows],
                                      block["length_cm"], cell_seed)
        nu = block.get("nu")
        if nu is not None:
            # One draw per TRACK, indexed by FIRST APPEARANCE -- not by the sorted order the field
            # recursion uses.  Getting this ordering wrong is how the port first disagreed with
            # kaonana by a factor 2.3: the recursion consumes one normal per hit in sorted order, so
            # any permutation of the sort permutes the noise.  Vectorised; a dict lookup per hit is
            # not affordable at reprocessing scale.
            keys = np.asarray(track_id)[rows]
            _, first_index, inverse = np.unique(keys, return_index=True, return_inverse=True)
            rank = np.empty(len(first_index), dtype=int)
            rank[np.argsort(first_index)] = np.arange(len(first_index))
            codes = rank[inverse]
            draws = np.random.default_rng(cell_seed + 77).chisquare(nu, size=len(first_index))
            field = field * np.sqrt(nu / draws)[codes]
        if "amplitude" in block:
            amplitude = block["amplitude"]
        elif "amplitude_dedx" in block:
            if amplification is None:
                amplification = sbnd_amplification(charge, phi, efield, density, calo_params)
            amplitude = block["amplitude_dedx"] / amplification[rows]
            if "power" in block:
                # The rr shape, FROZEN outside the range it was fitted in.  A power law in rr is an
                # extrapolation below the lowest fitted band and (rr/rr0)**-power diverges as rr -> 0;
                # the analysis smears every hit, including the rr < 1 cm ones the derivation's floor
                # excluded, and unfrozen that reached a 1.81 factor at rr ~ 1e-3 cm against 1.11 at the
                # lowest fitted band.  Same freeze as SBND_SATURATION's pitch term, same reason.
                low, high = block.get("rr_range", (2.0, 26.0))
                clipped = np.clip(np.asarray(rr, dtype=float)[rows], low, high)
                amplitude = amplitude * (clipped / block["rr0"]) ** -block["power"]
        else:
            raise KeyError(f"SBND_MC_NOISE[{int(tpc)}, {int(plane)}] has neither "
                           "`amplitude` (charge) nor `amplitude_dedx`")
        knee = block.get("knee")
        if knee is not None:
            # Gated on the charge as it arrives; see (1) in the docstring on which charge that is.
            amplitude = np.where(np.asarray(charge, dtype=float)[rows] >= float(knee), amplitude, 0.0)
        if nu is not None:
            out[rows] = np.exp(amplitude * field)
        else:
            out[rows] = np.maximum(1.0 + amplitude * field, SBND_CHARGE_FLOOR)
    return out


#: The ladder's charge correction, kaonana CALO.145 rebuilt and CALO.163 reparameterised.  REPLACES
#: `SBND_SATURATION` in `sbnd_calo_chain`; that block stays defined for the figures that cite it.
#:
#:     ln factor = ref_level + ln(depth) * (S(q) - S(q_ref)) / (S(q_hi) - S(q_lo))
#:                 + b2 * ln(pitch / p0)
#:     S(q)      = s * logaddexp(0, ln(q / knee) / s)
#:
#: `ref_level` is the log factor at Q_REF; `depth` is factor(Q_HIGH) / factor(Q_LOW).  THE THREE
#: ANCHORS ARE FIXED AND SHARED -- 6.0e4, 1.2e5, 2.4e5 -- and are NOT each block's [q_min, q_max].
#: Anchoring on the window instead would make `depth` mean a different thing in every cell and
#: silently rescale the correction; nothing downstream would flag it.
#:
#: `knee` and `s` are SCANNED AND HELD, not fitted: they define the basis.  `(b0, b1)` are absent by
#: design -- `b1` is degenerate with the knee and floats 37-123% across the region the chi2 cannot
#: distinguish, while the curve it describes holds to 0.5%.  Quoting it would be quoting noise.
#:
#: Frozen outside [q_min, q_max] AND outside [pitch_min, pitch_max].  Both matter: `b2` is a pitch
#: coefficient and plane 2's window is only 0.30-0.75, so an unfrozen tilt extrapolates hard.
#:
#: Six independent fits, nothing pooled across TPCs or planes.  `depth` is consistent with one number
#: (0.9399 +- 0.0039); `ref_level` is decisively per-cell.  A log-linear alternative -- which is what
#: `sbnd_reco_factor` is -- loses in ALL SIX at chi2/dof up to 3.353 against 1.272, with the
#: curvature significant at 2.4-8.1 sigma.  That is why `reco` is not a substitute for this rung.
SBND_CHARGE_TURNON = {
    (0, 0): dict(ref_level=0.014, depth=0.94684, b2=-0.03498,
                 knee=227670.5, s=0.09487, q_min=50939.0, q_max=266493.0,
                 pitch_min=0.3, pitch_max=1.6),
    (0, 1): dict(ref_level=0.00803, depth=0.93795, b2=-0.01225,
                 knee=171238.2, s=0.05335, q_min=51504.0, q_max=260261.0,
                 pitch_min=0.3, pitch_max=1.7),
    (0, 2): dict(ref_level=-0.00047, depth=0.93043, b2=-0.02415,
                 knee=207591.1, s=0.08215, q_min=50089.0, q_max=265293.0,
                 pitch_min=0.3, pitch_max=0.75),
    (1, 0): dict(ref_level=0.03653, depth=0.94348, b2=-0.0276,
                 knee=227670.5, s=0.1687, q_min=50908.0, q_max=269680.0,
                 pitch_min=0.3, pitch_max=1.6),
    (1, 1): dict(ref_level=0.0118, depth=0.94529, b2=-0.01456,
                 knee=231843.2, s=0.25979, q_min=51362.0, q_max=269954.0,
                 pitch_min=0.3, pitch_max=1.7),
    (1, 2): dict(ref_level=0.01164, depth=0.95079, b2=-0.03231,
                 knee=249334.8, s=0.08215, q_min=50218.0, q_max=261276.0,
                 pitch_min=0.3, pitch_max=0.75),
}

#: The anchors `depth` is defined on.  Constants, not tunables -- see SBND_CHARGE_TURNON.
SBND_TURNON_ANCHORS = dict(q_low=60000.0, q_ref=120000.0, q_high=240000.0, p0=0.42)


def sbnd_charge_turnon_factor(charge, pitch, itpc, plane):
    """The charge turn-on on MC's charge, frozen outside its fitted charge AND pitch windows.

    Verified against kaonana's `softplus_turnon` on all six cells to 2.2e-16
    (`sandbox_lowvar/port_turnon.py`).  Every cell is checked separately on purpose: this block is
    keyed `(itpc, plane)` while kaonana's registry is keyed `"t0p2"`, built the other way round, and a
    transposition is nearly invisible on plane 1 -- where the two cells differ least -- and wrong
    everywhere else.  Three of the five defects CALO.124 found were transport bugs of that kind.
    """
    charge = np.asarray(charge, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    itpc = np.asarray(itpc, dtype=int)
    out = np.ones(len(charge))
    anchors = SBND_TURNON_ANCHORS
    for tpc in np.unique(itpc):
        block = SBND_CHARGE_TURNON.get((int(tpc), int(plane)))
        if block is None:
            continue
        rows = itpc == tpc
        q = np.clip(charge[rows], block["q_min"], block["q_max"])
        p = np.clip(pitch[rows], block["pitch_min"], block["pitch_max"])
        s, knee = block["s"], block["knee"]

        def softplus(value, s=s, knee=knee):
            return s * np.logaddexp(0.0, np.log(value / knee) / s)

        low = softplus(anchors["q_low"])
        high = softplus(anchors["q_high"])
        reference = softplus(anchors["q_ref"])
        shape = (softplus(q) - reference) / (high - low)
        out[rows] = np.exp(block["ref_level"] + np.log(block["depth"]) * shape
                           + block["b2"] * np.log(p / anchors["p0"]))
    return out


def sbnd_calo_chain(dqdxdf, charge, plane, isMC, calo_params, seed=0, smear=True):
    """The five rungs, in the only order that is right.  Returns CORRECTED CHARGE.

    `charge` is the CALIBRATED dQ/dx -- gain, lifetime and YZ already in.  The recombination
    inversion happens after this returns, once.

    THE SMEAR RUNS FIRST, and that is the whole reason this docstring changed.  Two arguments, one
    physical and one arithmetic:

      physical    the noise is a fluctuation of the CHARGE THAT WAS COLLECTED, while `reco`,
                  `absorb` and `saturation` all model something that happened to it downstream.
                  MC's missing noise therefore belongs before them (kaonana CALO.148, which settled
                  the same question against the turn-on and measured the order at 4.5-5.6% on the
                  high half of the two innermost proton bands, under 0.3% elsewhere).
      arithmetic  `sbnd_smear_factor`'s `knee` is a threshold in CHARGE, and kaonana fitted it on the
                  calibrated charge.  Called last, this function received the post-saturation charge,
                  so a knee of 5.6e4 gated at ~5.3e4 calibrated -- inside the MIP core rather than at
                  its top edge, which is exactly the region the knee exists to keep the kernel out of.
                  Re-expressing the knee in the received basis does NOT fix it: the pre-smear factor
                  at fixed charge spans +4.5%/-7.66% across real hits (varying with phi through
                  `reco`, with pitch through `saturation`, and with field and density through
                  `absorb`), so one adjusted number leaves a per-hit misplacement the size of the
                  original error.  Running first, the smear sees the calibrated charge and the fitted
                  knee is already in the right basis.  `sbnd_level_scale` is 1.0 for MC, so nothing
                  stands between the input and the gate.

    ONE CONSEQUENCE FOR THE dE/dx AMPLITUDE BASIS.  `amplitude_dedx` blocks divide by
    `sbnd_amplification` evaluated at the hit's charge, which is now the calibrated charge rather
    than the post-saturation one.  `SBND_MC_NOISE` ships `amplitude` (charge basis), so nothing in
    use is affected -- but that path exists and both dE/dx bases have been tried and lost (CALO.113,
    CALO.125), so re-deriving before using one is required anyway.

    The FITTED blocks are held at their nominal values in every calo universe, so each
    variation is a variation about the corrected central value, which is what a covariance
    built from them assumes.

    `absorb` IS RETIRED AND THIS FUNCTION NO LONGER CALLS IT (kaonana CALO.143).  It was never a
    correction: `absorbing_factor` is `R_MC(Rinv_data(q, phi), phi) / q`, so feeding its output
    through MC's own inversion gives `Rinv_data(q)` exactly -- proven to 1.3e-15.  Applying it and
    then inverting with MC's ModBox therefore delivered DATA's ModBox for MC, which is the opposite
    of the decision (MC's ModBox, one inversion).  It is a CHOICE OF INVERSION, not a rung, and
    keeping both was double-counting worth -7% to +25% in dE/dx and sign-changing with angle.

    `sbnd_absorbing_factor` is deliberately LEFT DEFINED.  It is the closed form of the alternative
    choice, so anyone wanting data's ModBox should call it here and invert with MC's -- or invert with
    data's slot and not call it.  What is wrong is doing both, which is what this patch removes.
    Its own two-slot use is correct and is checked (see the note on `sbnd_amplification`).

    A side effect worth having: the retired rung was the only source of NON-POSITIVE charge in the
    chain.  It returned values down to -161.7 for 70 of 87,626 plane-2 MC hits, all at calibrated
    charge 15-3761 e/cm against a MIP's ~54,000, and `sbnd_saturation_factor` then turned those into
    NaN -- for a block without `b3`, `ramp` is 0.0, the clipped charge is 0, `log(0)` is -inf, and
    `0.0 * -inf` is NaN rather than 0.  NaN charge becomes NaN dE/dx, NaN chi2, and a NaN BDT feature
    that xgboost absorbs silently.  With `absorb` gone, no hit is sent non-positive.  The `0.0 * -inf`
    trap in the saturation is still latent and still worth guarding; nothing feeds it now.
    """
    charge = np.asarray(charge, dtype=float)
    itpc = np.asarray(dqdxdf.tpc)
    phi = np.asarray(dqdxdf.phi, dtype=float)
    charge = charge * sbnd_level_scale(itpc, plane, isMC=isMC)
    # Everything past here is MC-only, so data leaves now.  The early return moved up from below
    # `reco` when `reco` was dropped, which is what makes the guard on the smear unnecessary.
    if not isMC:
        return charge
    # The smear FIRST, on the calibrated charge -- see the docstring.  With `reco` gone there is
    # nothing at all between the input and the gate, so the fitted knee is exactly in its own basis.
    if smear:
        levels = list(range(dqdxdf.index.nlevels - 1))
        track = (np.asarray(dqdxdf.index.droplevel(-1).to_numpy()) if levels
                 else np.zeros(len(charge), dtype=int))
        charge = charge * sbnd_smear_factor(np.asarray(dqdxdf.rr, dtype=float), track,
                                            itpc, plane, seed, charge, phi,
                                            np.asarray(dqdxdf.efield, dtype=float),
                                            np.asarray(dqdxdf.rho, dtype=float), calo_params)
    charge = charge * sbnd_charge_turnon_factor(charge, np.asarray(dqdxdf.pitch, dtype=float),
                                                itpc, plane)
    return charge


def dedx(dqdxdf, gain=None, calibrate=None, plane=2, isMC=False, smear=-1, scale=1, new_calo_params=None, charge="integral", calo_chain=False, calo_seed=0, calo_smear=True):
    dqdx_v = dqdx(dqdxdf, gain=gain, calibrate=calibrate, isMC=isMC, charge=charge)
    if gain == "SBND":

        if new_calo_params is None:
            calo_params = SBND_CALO_PARAMS
        else:
            calo_params = new_calo_params

        scalegain = calo_params['c_cal_frac'][plane]
        # ONE INVERSION, MC's ModBox, FOR BOTH SAMPLES -- the second half of kaonana CALO.143,
        # which decided "MC's ModBox, one inversion, `absorb` retired".  Only the retirement was
        # ported; this is the rest.  Slot 0 is MC's set (see `sbnd_amplification`).
        #
        # `absorbing_factor` is `R_MC(Rinv_data(q)) / q`, so `Rinv_MC(absorb * q) == Rinv_data(q)`
        # identically -- verified to 1.33e-15 on 97,415 real hits.  So while `absorb` was applied MC
        # inverted *effectively* with DATA's ModBox and data inverted with data's: both against the
        # decision, equally, and it cancelled in every data/MC ratio.  Retiring `absorb` removed
        # MC's half of that accident and left the two samples on different recombination models,
        # with nothing left to cancel.
        #
        # Cost of leaving it, measured on control hits: data's dE/dx sits a median 6.0% from where
        # the decision puts it, and the factor runs 0.96 to 1.09 across angle and charge, CROSSING
        # 1 -- 0.9992 at phi < 0.5 against 1.0642 at phi > 1.2, and 0.9615 for high-charge hits in
        # the first band.  NO CHARGE RUNG CAN ABSORB A SIGN-CHANGING ANGULAR FACTOR, which is why
        # the symptom must not be treated by restoring `reco`: that would pull the median back and
        # leave the angular residual in place, looking fixed.
        #
        # This is also what `plot_calo_chi2pid.py --inversion common` has always done.  With this
        # patch the chain production runs is the chain the control figures were judged on; without
        # it they were two different ladders, which is exactly why control-sample agreement did not
        # transfer to the BDT inputs and how this was found.
        this_alpha_emb = calo_params["alpha_emb"][0]
        this_beta_90 = calo_params["beta_90"][0]
        this_R_emb = calo_params["R_emb"][0]
        this_dqdx = scale*dqdx_v/scalegain
        if calo_chain:
            # The five derived rungs, on the CALIBRATED charge, before the inversion.
            this_dqdx = sbnd_calo_chain(dqdxdf, this_dqdx, plane, isMC, calo_params,
                                        seed=calo_seed, smear=calo_smear)
        dedx = calo.recombination_cor(this_dqdx, dqdxdf.phi, dqdxdf.efield, dqdxdf.rho, this_alpha_emb, this_beta_90, this_R_emb)

    elif gain == "ICARUS":
        scalegain = ICARUS_CALO_PARAMS['c_cal_frac'][plane]
        dedx = calo.recombination_cor(scale*dqdx_v/scalegain, dqdxdf.phi, dqdxdf.efield, dqdxdf.rho)

    else:
        scalegain = 1.
        dedx = calo.recombination_cor(scale*dqdx_v/scalegain, dqdxdf.phi, dqdxdf.efield, dqdxdf.rho)

    if smear > 0:
        dedx = dedx*np.random.normal(loc=1., scale=smear, size=dedx.size)

    return dedx

def _yz_ybin(y, yz_ybin):
    return np.searchsorted(yz_ybin, y) - 1

def _yz_zbin(z, yz_zbin):
    return np.searchsorted(yz_zbin, z) - 1

def _yz_iov(run): 
    iov = __iov(run, IC_yz_cal_iovdf)
    iov[run == 1] = 5 # non-Overlay MC default to Run 4
    return iov

def _etau_iov(run):
    iov = __iov(run, IC_etau_cal_iovdf)
    iov[run == 1] = -1 # non-Overlay MC default to no run
    return iov

def _tpc_iov(run):
    iov = __iov(run, IC_tpc_cal_iovdf)
    iov[run == 1] = 4 # non-Overlay MC default to Run 4
    return iov

def __iov(run, df):
    return pd.cut(run, list(df.run) + [np.inf], labels=df.iov).astype(float).fillna(-1).astype(int)

def _etau_iov_sbnd(run):
    return __iov_sbnd(run, SBND_etau_cal_iovdf)

def __iov_sbnd(run, df):
    return pd.cut(run, list(df.run) + [np.inf], labels=df.iov, right=False)

##############################
# EXPECTED dE/dx FILES
##############################
datadir = "/cvmfs/larsoft.opensciencegrid.org/products/larsoft_data/" + larsoft_data_v + "/ParticleIdentification/"
fhist = datadir + "dEdxrestemplates.root"

profp = uproot.open(fhist)["dedx_range_pro"]
profmu = uproot.open(fhist)["dedx_range_mu"]
profka = uproot.open(fhist)["dedx_range_ka"]
profpi = uproot.open(fhist)["dedx_range_pi"]

proton_dedx = profp.values()
proton_rr = profp.axis().edges()
proton_yerr = profp.errors(error_mode="s")
for i in range(len(proton_yerr)):
    if proton_yerr[i] < 1e-6:
        proton_yerr[i] = (proton_yerr[i-1] + proton_yerr[i+1]) / 2
    if proton_dedx[i] < 1e-6:
        proton_dedx[i] = (proton_dedx[i-1] + proton_dedx[i+1]) / 2

kaon_dedx = profka.values()
kaon_rr = profka.axis().edges()
kaon_yerr = profka.errors(error_mode="s")
for i in range(len(kaon_yerr)):
    if kaon_yerr[i] < 1e-6:
        kaon_yerr[i] = (kaon_yerr[i-1] + kaon_yerr[i+1]) / 2
    if kaon_dedx[i] < 1e-6:
        kaon_dedx[i] = (kaon_dedx[i-1] + kaon_dedx[i+1]) / 2

muon_dedx = profmu.values()
muon_rr = profmu.axis().edges()
muon_rr_center = profmu.axis().centers()
muon_yerr = profmu.errors(error_mode="s")

pion_dedx = profpi.values()
pion_rr = profpi.axis().edges()
pion_yerr = profpi.errors(error_mode="s")
for i in range(len(pion_yerr)):
    if pion_yerr[i] < 1e-6:
        pion_yerr[i] = (pion_yerr[i-1] + pion_yerr[i+1]) / 2
    if pion_dedx[i] < 1e-6:
        pion_dedx[i] = (pion_dedx[i-1] + pion_dedx[i+1]) / 2

##############################
# ICARUS TPC calo files
##############################
# ICARUS CALIBRATION DATABASES
IC_yz_cal_f = "/cvmfs/icarus.opensciencegrid.org/products/icarus/icarus_data/" + icarus_data_v + "/icarus_data/database/tpc_yz_correction_allplanes_data.db"
IC_yz_cal_db = "tpc_yz_correction_allplanes_data_data"
IC_yz_cal_iov = "tpc_yz_correction_allplanes_data_iovs"

IC_etau_cal_f = "/cvmfs/icarus.opensciencegrid.org/products/icarus/icarus_data/" + icarus_data_v + "/icarus_data/database/tpc_elifetime_data.db"
IC_etau_cal_db = "tpc_elifetime_data_data"
IC_etau_cal_iov = "tpc_elifetime_data_iovs"

IC_tpc_cal_f = "/cvmfs/icarus.opensciencegrid.org/products/icarus/icarus_data/" + icarus_data_v + "/icarus_data/database/tpc_dqdxcalibration_allplanes_data.db"
IC_tpc_cal_db = "tpc_dqdxcalibration_allplanes_data_data"
IC_tpc_cal_iov = "tpc_dqdxcalibration_allplanes_data_iovs"

# LOAD THE YZ CALIBRATION
yz_ybin = np.linspace(-180, 130, 32)
yz_ylos = yz_ybin[:-1]
yz_yhis = yz_ybin[1:]
yz_ys = (yz_ylos + yz_yhis) / 2.

yz_zbin = np.linspace(-900, 900, 181)
yz_zlos = yz_zbin[:-1]
yz_zhis = yz_zbin[1:]
yz_zs = (yz_zlos + yz_zhis) / 2.

conn = sqlite3.connect(IC_yz_cal_f)
cursor = conn.cursor()
cursor.execute("SELECT * FROM %s" % IC_yz_cal_db)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_yz_cal_df = pd.DataFrame({
  "iov": data[0],
  "plane": data[2],
  "tpc": data[3],
  "ybin": data[4],
  "zbin": data[5],
  "scale": data[6]
})
IC_yz_cal_df["itpc"] = 0 
IC_yz_cal_df.loc[IC_yz_cal_df.tpc == "EE", "itpc"] = 0
IC_yz_cal_df.loc[IC_yz_cal_df.tpc == "EW", "itpc"] = 1
IC_yz_cal_df.loc[IC_yz_cal_df.tpc == "WE", "itpc"] = 2
IC_yz_cal_df.loc[IC_yz_cal_df.tpc == "WW", "itpc"] = 3
del IC_yz_cal_df["tpc"]

cursor.execute("SELECT * FROM %s WHERE ACTIVE=1" % IC_yz_cal_iov)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_yz_cal_iovdf = pd.DataFrame({
  "iov": data[0],
  "begin_time": data[1],
})
IC_yz_cal_iovdf["run"] = IC_yz_cal_iovdf.begin_time % 1000000000
IC_yz_cal_iovdf.sort_values(by="run", inplace=True)
conn.close()

# LOAD THE LIFETIME CALIBRATION
conn = sqlite3.connect(IC_etau_cal_f)
cursor = conn.cursor()
cursor.execute("SELECT * FROM %s" % IC_etau_cal_db)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_etau_cal_df = pd.DataFrame({
  "iov": data[0],
  "itpc": data[1],
  "etau": data[2]
})

cursor.execute("SELECT * FROM %s WHERE ACTIVE=1" % IC_etau_cal_iov)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_etau_cal_iovdf = pd.DataFrame({
  "iov": data[0],
  "begin_time": data[1],
})
IC_etau_cal_iovdf["run"] = IC_etau_cal_iovdf.begin_time % 1000000000
IC_etau_cal_iovdf.sort_values(by="run", inplace=True)
conn.close()

# LOAD THE TPC SCALE
conn = sqlite3.connect(IC_tpc_cal_f)
cursor = conn.cursor()
cursor.execute("SELECT * FROM %s" % IC_tpc_cal_db)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_tpc_cal_df = pd.DataFrame({
  "iov": data[0],
  "plane": data[2],
  "tpc": data[3],
  "scale": data[4]
})
IC_tpc_cal_df["itpc"] = 0 
IC_tpc_cal_df.loc[IC_tpc_cal_df.tpc == "EE", "itpc"] = 0
IC_tpc_cal_df.loc[IC_tpc_cal_df.tpc == "EW", "itpc"] = 1
IC_tpc_cal_df.loc[IC_tpc_cal_df.tpc == "WE", "itpc"] = 2
IC_tpc_cal_df.loc[IC_tpc_cal_df.tpc == "WW", "itpc"] = 3
del IC_tpc_cal_df["tpc"]

cursor.execute("SELECT * FROM %s WHERE ACTIVE=1" % IC_tpc_cal_iov)
rows = cursor.fetchall()
data = list(zip(*rows))
IC_tpc_cal_iovdf = pd.DataFrame({
  "iov": data[0],
  "begin_time": data[1],
})
IC_tpc_cal_iovdf["run"] = IC_tpc_cal_iovdf.begin_time % 1000000000
IC_tpc_cal_iovdf.sort_values(by="run", inplace=True)
conn.close()

##############################
# SBND TPC calo files
##############################

# load SBND YZ unif maps
# the maps normtools_sbnd.fcl points NormalizeYZ at, so the correction cafpyana
# applies to integral/pitch is the one the reco applied to the same hits
SBND_yz_cal_mc_f = "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data/" + sbnd_data_v + "/YZmaps/yz_mc2025_v10_14_02.root"
SBND_yz_cal_data_f = "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data/" + sbnd_data_v + "/YZmaps/yz_data2025_v10_14_02.root"

yz_zbin_sbnd_mc = []
yz_ybin_sbnd_mc = []
yz_zbin_sbnd_data = []
yz_ybin_sbnd_data = []

def call_sbnd_yz_corr(map_f):
    maps = []
    z_edges = y_edges = None
    for tpc, plane in [(t, p) for t in range(2) for p in range(3)]:
        this_hist = uproot.open(map_f)["CzyHist_" + str(plane) + "_" + str(tpc)]
        this_yz_zbin = this_hist.axis(0).edges()
        this_yz_ybin = this_hist.axis(1).edges()

        if z_edges is None:
            z_edges, y_edges = this_yz_zbin, this_yz_ybin

        corr = this_hist.values()

        corr_mi = pd.MultiIndex.from_product(
            [range(corr.shape[0]), range(corr.shape[1])],
            names=["zbin", "ybin"]
        )

        corr_df = pd.Series(corr.ravel(), index=corr_mi, name="scale").to_frame()
        corr_df = corr_df.reset_index()

        corr_df['itpc'] = tpc
        corr_df['plane'] = plane
        corr_df['iov'] = 0
        new_order = ["iov", "plane", "ybin", "zbin", "scale", 'itpc']
        corr_df = corr_df[new_order]
        maps.append(corr_df)

    out_df = pd.concat(maps, ignore_index=True)
    return out_df, z_edges, y_edges

SBND_yz_cal_mc_df, yz_zbin_sbnd_mc, yz_ybin_sbnd_mc = call_sbnd_yz_corr(SBND_yz_cal_mc_f)
SBND_yz_cal_data_df, yz_zbin_sbnd_data, yz_ybin_sbnd_data = call_sbnd_yz_corr(SBND_yz_cal_data_f)

# load SBND etau DB
SBND_etau_cal_f = "/cvmfs/sbnd.opensciencegrid.org/products/sbnd/sbnd_data/" + sbnd_data_v + "/CalibrationDatabase/tpc_elifetime.db"
SBND_etau_cal_db = "tpc_elifetime_data"
SBND_etau_cal_iov = "tpc_elifetime_iovs"

conn = sqlite3.connect(SBND_etau_cal_f)
cursor = conn.cursor()
cursor.execute("SELECT * FROM %s" % SBND_etau_cal_db)
rows = cursor.fetchall()
data = list(zip(*rows))
SBND_etau_cal_data_df = pd.DataFrame({
  "iov": data[0],
  "etau_E": data[5],
  "etau_W": data[8],
})

cursor.execute("SELECT * FROM %s WHERE ACTIVE=1" % SBND_etau_cal_iov)
rows = cursor.fetchall()
data = list(zip(*rows))
SBND_etau_cal_iovdf = pd.DataFrame({
  "iov": data[0],
  "begin_time": data[1],
})
SBND_etau_cal_iovdf["run"] = SBND_etau_cal_iovdf.begin_time % 1000000000
SBND_etau_cal_iovdf.sort_values(by="run", inplace=True)
conn.close()

SBND_etau_cal_mc_df = pd.DataFrame( {'iov': [0, 0], 'itpc': [0, 1], 'etau': [SBND_CALO_PARAMS["etau"][0], SBND_CALO_PARAMS["etau"][0]]})
