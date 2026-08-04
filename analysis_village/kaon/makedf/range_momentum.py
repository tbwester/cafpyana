"""
Range-momentum for a kaon hypothesis, reproducing sbncode offline.

Why this exists
---------------
``caf::SRTrkRange`` carries only ``p_muon``, ``p_pion`` and ``p_proton``; there
is no kaon slot.  Adding one upstream means a coordinated sbnobj/sbnanaobj schema
change *and* a full reco reprocessing of every file we already have.  Neither is
necessary.  The producer that fills those branches
(``sbncode/LArRecoProducer/RangePAllPID_module.cc``) takes exactly one
track-dependent input -- ``recob::Track::Length()`` -- and that is already in the
CAF as ``pfp.trk.len``.  Every hypothesis, kaon included, is a one-argument
function of a column we already have.

What sbncode actually does
--------------------------
``trkf::TrackMomentumCalculator::GetTrackMomentum`` (the *sbncode* fork under
``sbncode/LArRecoProducer/LArReco/``, not the larreco one) knows only two
particles:

* pdg 13   -- a ``TSpline3`` through the 29-point PDG muon CSDA table
* pdg 2212 -- two PSTAR polynomial fits

Anything else, kaons included, returns a sentinel.  There is no pion table and
never was: the pion hypothesis is produced by *mass-scaling the muon table*, in
``RangePAllPID_module.cc:89-95``::

    // Rescale the input and output as described by
    // https://inspirehep.net/literature/1766384 (eq. 6.2)
    if (PIDs[i] == 211) {
      rangep.range_p =
        fRangeCalculator.GetTrackMomentum(track->Length() * (fMuonMass / fPionMass), 13)
        * (fPionMass / fMuonMass);
    }

Electronic stopping power depends on beta*gamma = p/m alone, so R/m is a
universal function of p/m for particles of the same charge, giving

    p_X(L) = (m_X / m_mu) * p_mu(L * m_mu / m_X)

The kaon substitutes into that identity on exactly the same footing as the pion.

Validation
----------
* Mass-scaling the muon table into a *proton* reproduces sbncode's independent
  PSTAR parameterisation to better than 1% above 10 cm (worst 1.4% at 5 cm).
  That is the justification for the kaon: the identity is checkable against
  ground truth inside the same C++ function, at a larger mass ratio (8.9x) than
  the kaon needs (4.7x).
* Closure against flat CAFs: recomputing ``p_muon``/``p_pion``/``p_proton`` from
  ``trk.len`` alone reproduces what CAFMaker wrote to worst-case relative
  differences of 8.7e-8 / 8.2e-7 / 7.7e-8 over 5440 tracks -- the float32
  storage precision of the branches.  ``p_kaon`` comes off the same code path.

Both live in ``sandbox_rangep/`` in the kaonana repo.

Fidelity notes -- do not "fix" these
------------------------------------
This reproduces sbncode bit-for-bit rather than "correctly", so that ``p_pion``
computed here can be closure-tested against the ``p_pion`` already in the CAF.
Three quirks are inherited deliberately:

* the g/cm^2 -> cm conversion uses **1.396**, though the comment above the table
  in the C++ says 1.4;
* ``GetTrackMomentum`` hard-codes ``Muon_M = 105.7`` MeV while the *rescale* in
  ``RangePAllPID`` uses ``TDatabasePDG`` masses -- inconsistent upstream, kept;
* the table is ``std::array<float>`` divided in float and then widened to double
  by the ``TGraph`` constructor, so the knots carry float rounding.

If these are fixed, closure against the CAF is lost, and with it the only
argument that the kaon number is what a patched sbncode would have written.

numpy only, no scipy and no ROOT: the spline is a translation of
``TSpline3::BuildCoeff``, so this runs inside the df workers with no new
dependency.
"""

import numpy as np

__all__ = [
    "KAON_MIN_LENGTH_CM",
    "MASS_KAON",
    "MASS_MUON",
    "MASS_PION",
    "MASS_PROTON",
    "momentum_from_range",
    "p_kaon",
    "p_muon",
    "p_pion",
    "p_proton",
    "track_momentum",
]

# ---------------------------------------------------------------------------
# The muon CSDA table, verbatim from
# sbncode/LArRecoProducer/LArReco/TrackMomentumCalculator.cxx:21-36
#
# Source: PDG 2012 muon energy-loss tables for liquid argon,
# http://pdg.lbl.gov/2012/AtomicNuclearProperties/MUON_ELOSS_TABLES/muonloss_289.pdf
#
# Laid out 8-per-row to match the C++ source line for line so the two can be
# diffed by eye; the point of this block is that it is checkable against
# upstream, not that it is pretty.
# ---------------------------------------------------------------------------

_RANGE_GPERCM2 = np.array([
    9.833e-1, 1.786e0, 3.321e0, 6.598e0, 1.058e1, 3.084e1, 4.250e1, 6.732e1,
    1.063e2,  1.725e2, 2.385e2, 4.934e2, 6.163e2, 8.552e2, 1.202e3, 1.758e3,
    2.297e3,  4.359e3, 5.354e3, 7.298e3, 1.013e4, 1.469e4, 1.910e4, 3.558e4,
    4.326e4,  5.768e4, 7.734e4, 1.060e5, 1.307e5,
], dtype=np.float32)

_KE_MEV = np.array([
    10,    14,    20,    30,    40,     80,     100,    140,    200,   300,
    400,   800,   1000,  1400,  2000,   3000,   4000,   8000,   10000, 14000,
    20000, 30000, 40000, 80000, 100000, 140000, 200000, 300000, 400000,
], dtype=np.float32)

#: Argon density used by the C++ (note: the comment there says 1.4).
_ARGON_DENSITY = 1.396

# ``value /= 1.396`` on a float array: promote to double, divide, truncate back
# to float.  Then ROOT's TGraph(Int_t, const Float_t*, const Float_t*) widens to
# double.  Replicated so the knots land on the same doubles ROOT used.
_RANGE_CM = (_RANGE_GPERCM2.astype(np.float64) / _ARGON_DENSITY).astype(np.float32)
_RANGE_CM = _RANGE_CM.astype(np.float64)
_KE = _KE_MEV.astype(np.float64)

# ---------------------------------------------------------------------------
# Masses.  GetTrackMomentum hard-codes these two internally (MeV):
# ---------------------------------------------------------------------------

_MUON_M_INTERNAL = 105.7
_PROTON_M_INTERNAL = 938.272

# ...while RangePAllPID builds the rescale factors from TDatabasePDG (GeV).
# These are the ROOT pdg_table.txt values.
MASS_MUON = 0.1056584
MASS_PION = 0.13957039
MASS_KAON = 0.493677
MASS_PROTON = 0.9382720

#: Track length below which the kaon hypothesis extrapolates off the bottom of
#: the muon table (0.704 cm muon-equivalent / (m_mu/m_K)).  The muon branch of
#: GetTrackMomentum has *no* validity check -- unlike the proton branch it
#: silently extrapolates the first cubic instead of returning a sentinel -- so
#: this bound has to be enforced here.
KAON_MIN_LENGTH_CM = float(_RANGE_CM[0] / (MASS_MUON / MASS_KAON))


# ---------------------------------------------------------------------------
# TSpline3, not-a-knot
# ---------------------------------------------------------------------------


def _build_tspline3_coeffs(x, y):
    """Line-by-line translation of ``TSpline3::BuildCoeff`` (ROOT TSpline.cxx:1040).

    ROOT's ``TSpline3`` is constructed by sbncode with ``opt=""``, leaving
    ``fBegCond == fEndCond == 0``; with 29 > 3 knots that selects **not-a-knot**
    at both ends (TSpline.cxx:1064, :1096).  scipy's ``CubicSpline`` default is
    the same boundary condition and agrees to 4.3e-16, but this is a direct
    translation so the arithmetic is identical rather than merely mathematically
    equivalent -- and so this module needs neither scipy nor ROOT.

    Returns per-knot cubic coefficients ``(Y, B, C, D)`` such that on
    ``[x[k], x[k+1]]``::

        f(t) = Y[k] + dx*(B[k] + dx*(C[k] + dx*D[k])),   dx = t - x[k]
    """
    n = len(x)
    if n < 4:
        raise ValueError("this translation only covers the fNp > 3 not-a-knot branch")

    Y = np.asarray(y, dtype=np.float64).copy()
    B = np.zeros(n)
    C = np.zeros(n)
    D = np.zeros(n)

    # First differences of x in C, first divided differences of the data in D.
    for m in range(1, n):
        C[m] = x[m] - x[m - 1]
        D[m] = (Y[m] - Y[m - 1]) / C[m]

    # Not-a-knot condition at the left end (fBegCond == 0, fNp > 2).
    D[0] = C[2]
    C[0] = C[1] + C[2]
    B[0] = ((C[1] + 2.0 * C[0]) * D[1] * C[2] + C[1] * C[1] * D[2]) / C[0]

    # Interior knots: forward pass of Gauss elimination.
    g = 0.0
    for m in range(1, n - 1):
        g = -C[m + 1] / D[m - 1]
        B[m] = g * B[m - 1] + 3.0 * (C[m] * D[m + 1] + C[m + 1] * D[m])
        D[m] = g * C[m - 1] + 2.0 * (C[m] + C[m + 1])

    # Not-a-knot condition at the right end (fEndCond == 0, fNp > 3).
    g = C[n - 2] + C[n - 1]
    B[n - 1] = (
        (C[n - 1] + 2.0 * g) * D[n - 1] * C[n - 2]
        + C[n - 1] * C[n - 1] * (Y[n - 2] - Y[n - 3]) / C[n - 2]
    ) / g
    g = -g / D[n - 2]
    D[n - 1] = C[n - 2]

    # Complete the forward pass.
    D[n - 1] = g * C[n - 2] + D[n - 1]
    B[n - 1] = (g * B[n - 2] + B[n - 1]) / D[n - 1]

    # Back substitution; slopes end up in B.
    for j in range(n - 2, -1, -1):
        B[j] = (B[j] - C[j] * B[j + 1]) / D[j]

    # Cubic coefficients per interval.  In-place, and the read of C[i] must
    # happen before the write of C[i-1] on the next iteration -- preserved by
    # keeping this sequential, exactly as the C++ does.
    for i in range(1, n):
        dtau = C[i]
        divdf1 = (Y[i] - Y[i - 1]) / dtau
        divdf3 = B[i - 1] + B[i] - 2.0 * divdf1
        C[i - 1] = (divdf1 - B[i - 1] - divdf3) / dtau
        D[i - 1] = (divdf3 / dtau) / dtau

    return Y, B, C, D


_SPLINE_Y, _SPLINE_B, _SPLINE_C, _SPLINE_D = _build_tspline3_coeffs(_RANGE_CM, _KE)


def _spline_eval(xq):
    """Vectorised ``TSpline3::Eval``, including ROOT's extrapolation behaviour.

    ``TSpline3::FindX`` (TSpline.cxx:738) clamps below ``fXmin`` to the first
    interval and above ``fXmax`` to the last, so out-of-range queries extrapolate
    the end cubic rather than returning a sentinel.  ``searchsorted`` with a clip
    reproduces that index choice exactly.
    """
    xq = np.asarray(xq, dtype=np.float64)
    n = len(_RANGE_CM)
    k = np.clip(np.searchsorted(_RANGE_CM, xq, side="right") - 1, 0, n - 2)
    dx = xq - _RANGE_CM[k]
    return _SPLINE_Y[k] + dx * (_SPLINE_B[k] + dx * (_SPLINE_C[k] + dx * _SPLINE_D[k]))


# ---------------------------------------------------------------------------
# GetTrackMomentum
# ---------------------------------------------------------------------------


def track_momentum(trkrange, pdg):
    """Reproduce ``trkf::TrackMomentumCalculator::GetTrackMomentum``.

    Parameters
    ----------
    trkrange:
        Track length in cm.  Scalar, array, or pandas Series.
    pdg:
        Only 13 (muon) and 2212 (proton) are supported *by sbncode*; anything
        else returns the sentinel, as the C++ does.  Use `momentum_from_range`
        for the scaled hypotheses.

    Returns
    -------
    Momentum in GeV/c, or the sbncode sentinel where the C++ would produce one:
    -1 for a negative/NaN length, -0.999 otherwise.  (Note -0.999, not -999: the
    C++ sets ``Momentum = -999`` then divides by 1000 unconditionally.)
    """
    r = np.asarray(trkrange, dtype=np.float64)

    # if (trkrange < 0 || std::isnan(trkrange)) return -1.;
    invalid = np.isnan(r) | (r < 0)

    if abs(pdg) == 13:
        mass = _MUON_M_INTERNAL
        ke = _spline_eval(np.where(invalid, 1.0, r))
    elif abs(pdg) == 2212:
        mass = _PROTON_M_INTERNAL
        safe = np.where(invalid, 1.0, r)
        ke = np.where(
            (safe > 0) & (safe <= 80),
            29.9317 * np.power(np.clip(safe, 1e-30, None), 0.586304),
            np.where(
                (safe > 80) & (safe <= 3.022e3),
                149.904
                + 3.34146 * safe
                - 0.00318856 * safe**2
                + 4.34587e-6 * safe**3
                - 3.18146e-9 * safe**4
                + 1.17854e-12 * safe**5
                - 1.71763e-16 * safe**6,
                -999.0,
            ),
        )
    else:
        # KE = -999 for every other pdg -> Momentum = -999/1000.
        return np.full(r.shape, -0.999) if r.ndim else np.float64(-0.999)

    # np.where evaluates both branches, so clamp before the sqrt rather than
    # letting the KE<0 sentinel path emit an invalid-value warning.
    ke_safe = np.where(ke < 0, 0.0, ke)
    momentum = np.where(ke < 0, -999.0, np.sqrt(ke_safe * ke_safe + 2.0 * mass * ke_safe)) / 1000.0
    return np.where(invalid, -1.0, momentum)


def momentum_from_range(trkrange, mass):
    """Range-momentum for an arbitrary singly-charged mass, by muon-table scaling.

    Implements ``p_X(L) = (m_X/m_mu) * p_mu(L * m_mu/m_X)`` -- the same
    transformation ``RangePAllPID_module.cc:89-95`` applies for the pion, which
    is why feeding this `MASS_PION` reproduces the CAF's ``p_pion``.

    Parameters
    ----------
    trkrange:
        Track length in cm.
    mass:
        Particle mass in **GeV**, on the same scale as `MASS_MUON`.
    """
    scale = mass / MASS_MUON
    return track_momentum(np.asarray(trkrange, dtype=np.float64) / scale, 13) * scale


# ---------------------------------------------------------------------------
# The four hypotheses
# ---------------------------------------------------------------------------


def p_muon(trkrange):
    """Muon range-momentum [GeV/c].  Equals the CAF's ``rangeP.p_muon``.

    Caveat when closure-testing: ``CAFMaker::BlindEnergyParameters``
    (CAFMaker_module.cc:469-487) overwrites ``rangeP.p_muon`` with NaN above
    0.6 GeV for fiducial-start tracks -- but only in the *blinded* CAF stream,
    and it leaves ``p_pion``/``p_proton`` alone.  Prefer `p_pion` as the closure
    target: never blinded, and it exercises the scaling path too.
    """
    return track_momentum(trkrange, 13)


def p_pion(trkrange):
    """Pion range-momentum [GeV/c].  Equals the CAF's ``rangeP.p_pion``."""
    return momentum_from_range(trkrange, MASS_PION)


def p_proton(trkrange):
    """Proton range-momentum [GeV/c].  Equals the CAF's ``rangeP.p_proton``.

    Uses the PSTAR polynomials, *not* the scaling identity, because that is what
    sbncode does.  `momentum_from_range` with `MASS_PROTON` is the independent
    cross-check, not the reproduction.
    """
    return track_momentum(trkrange, 2212)


def p_kaon(trkrange, min_length_cm=KAON_MIN_LENGTH_CM):
    """Kaon range-momentum [GeV/c] -- the hypothesis the CAF does not carry.

    Parameters
    ----------
    trkrange:
        Track length in cm.
    min_length_cm:
        Return NaN below this length.  Defaults to `KAON_MIN_LENGTH_CM`
        (~3.29 cm), the point where the muon spline starts being extrapolated off
        the bottom of its table; sbncode itself would silently extrapolate and
        hand back a plausible-looking number.  Pass a smaller value to accept
        controlled extrapolation -- comparing not-a-knot against natural
        boundary conditions puts the extrapolation ambiguity at ~0.2% at 3.0 cm,
        ~1.7% at 2.0 cm and ~5.7% at 1.0 cm, so 2.0 is defensible and 1.0 is not.
        Pass 0 to disable the guard entirely.

    Physics caveat, and it dominates the numerics: range-momentum assumes the
    particle stopped by ionisation.  K+ decay-in-flight (c*tau = 3.7 m, a real
    effect at these momenta) and hadronic interaction both truncate the track,
    and both bias p_K low with a *tail* rather than a symmetric resolution.
    That is a selection problem, and it is not improved by computing this number
    upstream instead of here.
    """
    r = np.asarray(trkrange, dtype=np.float64)
    p = momentum_from_range(r, MASS_KAON)

    # The C++ sentinels get *multiplied by the mass ratio* on the way out of the
    # rescale, so an invalid length surfaces as -1 * m_K/m_mu = -4.672 rather
    # than anything recognisable.  (sbncode has the same behaviour for p_pion,
    # where it appears as -1.321 -- faithful, but useless downstream.)  This is a
    # new column with no parity obligation, so map every non-physical result to
    # NaN and let pandas propagate it.
    p = np.where(p <= 0, np.nan, p)
    p = np.where(np.isnan(r) | (r < 0), np.nan, p)

    if min_length_cm > 0:
        p = np.where(r < min_length_cm, np.nan, p)
    return p
