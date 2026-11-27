#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Folder of this script, and the default data folder next to it. Using the script location
# (not the current working directory) means the script finds ./data no matter where it is run from.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# Sweep direction codes. They are the sign of the setpoint step, so +1 = field going up.
RISE, FALL = 1, -1
NAME = {RISE: "Rising field", FALL: "Falling field"}

# Plot colours: blue = rising, orange = falling (a colour-blind-safe pair).
COLOR = {RISE: "#2a78d6", FALL: "#eb6834"}
# Neutral colours for text, secondary text, muted labels, grid lines and axes.
INK, INK2, MUTED, GRIDC, AXISC = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Columns that the sweep script always writes and that are NOT sensor signals.
# Everything else (except *_std columns) is a measured channel and can be the signal.
FIXED = {"t_s", "loop", "point", "setpoint", "polarity", "psu_V", "psu_I", "psu_mode",
         "I_coil_A", "B_mT", "B_set_mT", "I_shunt_A", "B_shunt_mT"}
# Field columns per --field choice, and how they are described in the output.
FIELD_COL = {"shunt": "B_shunt_mT", "set": "B_set_mT", "read": "B_mT"}
FIELD_DESC = {"shunt": "shunt current (B_shunt_mT)", "set": "setpoint (B_set_mT)",
              "read": "PSU read-back (B_mT)"}


# =====================================================================================
#  Loading
# =====================================================================================
def load(csv_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a sweep CSV into {column name: numpy array} plus the JSON metadata (if any).

    Only numeric columns are kept; text columns such as psu_mode ("CC"/"CV") are skipped.
    The JSON sidecar (same name, .json) holds the sweep settings, e.g. how many samples
    were averaged per point. It is optional: an empty dict is returned if it is missing.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))  # one dict per row, keyed by the header
    if not rows:
        raise ValueError("file has no data rows")
    cols = {}
    for k in rows[0]:
        try:
            # Convert the whole column to floats. If any value is not a number the
            # column is treated as text and left out.
            cols[k] = np.array([float(r[k]) for r in rows])
        except (TypeError, ValueError):
            pass  # text columns such as psu_mode
    # These four columns are needed by every step below; fail early with a clear message.
    for need in ("t_s", "loop", "setpoint", "B_mT"):
        if need not in cols:
            raise ValueError(f"column '{need}' missing")
    meta_path = csv_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return cols, meta


def pick_signal(cols: dict, name: str | None, supply: str, shunt: str = "") -> str:
    """Choose which column is the sensor signal.

    Order of preference: the column given with --signal, then V_bridge_out, then the first
    column that is not a fixed sweep column, not a *_std column, not the supply and not the
    coil-current shunt voltage. Runs that used only the Keithley therefore fall back to the
    Keithley column.
    """
    if name:
        if name not in cols:
            raise ValueError(f"signal column '{name}' not found; have {sorted(cols)}")
        return name
    if "V_bridge_out" in cols:
        return "V_bridge_out"
    cands = [c for c in cols if c not in FIXED and not c.endswith("_std") and c not in (supply, shunt)]
    if not cands:
        raise ValueError("no signal column found")
    return cands[0]


def n_averaged(meta: dict, signal: str) -> int:
    """How many raw readings were averaged into each point of `signal`.

    Needed to turn the stored standard deviation of the raw readings into the standard
    error of the mean: SEM = std / sqrt(N). For the myDAQ that is daq_samples (e.g. 1000),
    for the Keithley keithley_n_avg (e.g. 3). Without metadata we assume 1 (no averaging),
    which gives the most cautious (largest) error.
    """
    cfg = meta.get("config", {})
    # Set/reset "flip" runs add <label>_reset and <label>_sr columns; both average the same
    # number of readings as <label> itself.
    for suffix in ("_sr", "_reset"):
        if signal.endswith(suffix):
            signal = signal[: -len(suffix)]
    if signal in cfg.get("daq_labels", []):
        return int(cfg.get("daq_samples", 1))
    if signal == cfg.get("keithley_label"):
        return int(cfg.get("keithley_n_avg", 1))
    return 1


# =====================================================================================
#  Building blocks
# =====================================================================================
@dataclass
class Branch:
    """One monotonic part of the sweep (field only going up, or only going down).

    B and y hold one value per distinct setpoint (repeated setpoints are averaged), sorted
    by B so that np.interp can be used on them directly.
    """
    direction: int          # RISE / FALL
    idx: np.ndarray         # row indices in the file (time order)
    B: np.ndarray           # per unique setpoint, sorted by B
    y: np.ndarray           # signal at those fields (drift-corrected if --drift is used)
    noise: float            # per-point noise estimate (signal units)
    sp_lo: float            # lowest setpoint in the branch (A or V)
    sp_hi: float            # highest setpoint in the branch


def drift_model(t, y, loop, sp, mode: str) -> tuple[np.ndarray, str]:
    """Return an additive drift model (zero at t[0]) and a description.

    Idea: pick points that were measured at the same setpoint AND with the same magnetic
    history. Ideally they would all give the same signal, so any change between them is
    drift. A straight line through them (signal vs time) is the drift model, which the
    caller subtracts from the signal.

    mode
      "none"  : no correction.
      "ends"  : first and last point of the file. Only valid if both have the same setpoint
                and the sensor is in the same magnetic state at both (true for a unipolar
                sweep; NOT for a bipolar loop that starts from an unknown state).
      "loops" : the last point of every loop. All of them come at the end of a complete
                loop, so they share the same history. Needs at least 2 complete loops.
      "auto"  : "loops" if there are >= 2 loops, otherwise "none".

    The model is shifted so that it is 0 at t[0]: the correction keeps the signal level of
    the first point and only removes the change over time.
    """
    zero = np.zeros_like(y)
    loops = np.unique(loop)
    if mode == "auto":
        mode = "loops" if len(loops) >= 2 else "none"
    if mode == "none":
        return zero, "none"
    if mode == "ends":
        if sp[0] != sp[-1]:
            # Different setpoints at the start and end: their difference is not drift.
            return zero, "none (ends: first and last setpoints differ)"
        idx = np.array([0, len(y) - 1])
    elif mode == "loops":
        # Last row of each loop number.
        ends = [np.flatnonzero(loop == L)[-1] for L in loops]
        # Keep only loop ends at the same setpoint as the first loop's end. An aborted
        # (incomplete) last loop ends somewhere else and must not be used.
        ref = sp[ends[0]]
        idx = np.array([i for i in ends if sp[i] == ref])  # skip an aborted last loop
        if len(idx) < 2:
            return zero, "none (loops: need >= 2 complete loops)"
    else:
        raise ValueError(mode)
    # Straight line y = p[0] * t + p[1] through the reference points.
    p = np.polyfit(t[idx], y[idx], 1)
    return np.polyval(p, t) - np.polyval(p, t[0]), f"{mode} ({len(idx)} ref. points)"


def step_directions(sp: np.ndarray) -> np.ndarray:
    """Direction of each step i -> i+1; zero steps inherit the previous direction.

    The setpoint is used (not the measured B) because it is noise-free: the measured field
    wobbles slightly and could make a single step look like it goes the wrong way.

    Result has len(sp) - 1 entries, each +1 (RISE) or -1 (FALL). A step of 0 happens at the
    boundary between loops, where the setpoint 0 is written twice (end of one loop, start of
    the next). Such a step belongs to whatever direction was going on before it.
    """
    step = np.sign(np.diff(sp)).astype(int)
    nz = np.flatnonzero(step)          # indices of the non-zero steps
    if len(nz) == 0:
        return step                    # setpoint never changes: no direction at all
    step[:nz[0]] = step[nz[0]]         # leading zero steps take the first real direction
    for i in range(1, len(step)):      # every later zero step takes the previous direction
        if step[i] == 0:
            step[i] = step[i - 1]
    return step


def split_branches(sp: np.ndarray) -> list[tuple[int, int, int]]:
    """(direction, first row, last row) for every monotonic run; turning points are shared.

    Example, unipolar loop 0 -> 2 -> 0 A with rows 0..80 (row 40 is at 2 A):
        [(RISE, 0, 40), (FALL, 40, 80)]
    Row 40, the turning point, is the last point of the rising branch AND the first point of
    the falling branch, so both branches reach the maximum field.
    """
    step = step_directions(sp)
    if len(step) == 0 or not step.any():
        return []
    out, start = [], 0
    for i in range(1, len(step)):
        if step[i] != step[i - 1]:
            # Step i-1 and step i point in different directions, so row i is a turning
            # point. The branch so far covers rows start..i (steps start..i-1).
            out.append((int(step[i - 1]), start, i))
            start = i                  # the next branch starts at the turning point
    out.append((int(step[-1]), start, len(sp) - 1))  # the last branch runs to the end
    return out


def d2_noise(y: np.ndarray) -> float:
    """Robust point-to-point scatter from second differences (insensitive to smooth curvature).

    For a smooth curve with independent noise of standard deviation sigma on each point,
        d2_i = y[i-1] - 2*y[i] + y[i+1]
    is almost free of the curve itself (a straight line gives exactly 0) and has standard
    deviation sigma * sqrt(1 + 4 + 1) = sigma * sqrt(6).

    The spread of d2 is measured with the median absolute deviation (MAD) instead of the
    standard deviation, so a few big jumps (e.g. at a switching field) do not inflate it.
    For normally distributed data, std = 1.4826 * MAD.

    This estimate catches noise that the per-point standard error misses, such as slow
    fluctuations or mains pickup that does not average out within one point.
    """
    if len(y) < 5:
        return math.nan  # too few points for a meaningful estimate
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / math.sqrt(6))


def make_branch(direction, rows, sp, B, y, sem) -> Branch | None:
    """Build a Branch from the given rows of the file.

    rows      : row indices belonging to this branch (from split_branches)
    sp, B, y  : setpoint, field and signal for ALL rows of the file
    sem       : standard error of the mean of each point (std / sqrt(N averaged))

    Points with the same setpoint (e.g. the repeated 0 at a loop boundary) are averaged into
    one, so every branch has strictly one value per setpoint. Branches with fewer than
    3 distinct setpoints are ignored (returns None): too short to interpolate or to estimate
    noise from.
    """
    keys = np.unique(sp[rows])         # distinct setpoints in this branch, sorted
    if len(keys) < 3:
        return None
    Bu, yu, su = [], [], []            # averaged field, signal and SEM, one per setpoint
    for k in keys:  # average repeated setpoints (e.g. the duplicate 0 between loops)
        m = rows[sp[rows] == k]
        Bu.append(B[m].mean())
        yu.append(y[m].mean())
        # SEM of the mean of len(m) points: sqrt(sum of squares) / count
        su.append(math.sqrt(np.sum(sem[m] ** 2)) / len(m))
    # Sort by measured field, as np.interp needs increasing x values.
    o = np.argsort(Bu)
    Bu, yu, su = np.array(Bu)[o], np.array(yu)[o], np.array(su)[o]
    # d2 on setpoint order (uniform steps), not on B order
    y_sp = yu[np.argsort(keys[o])]
    # Per-point noise of the branch: the larger of the averaging error and the actual
    # scatter seen in the data. Taking the larger one is the cautious choice.
    noise = np.nanmax([np.nanmedian(su), d2_noise(y_sp)])
    return Branch(direction, rows, Bu, yu, float(noise), float(keys.min()), float(keys.max()))


def combine(branches: list[Branch], grid: np.ndarray):
    """Average same-direction branches on the grid. Returns mean, sigma, n, matrix.

    Every branch is linearly interpolated onto the common field grid. Outside the field range
    of a branch the value is NaN, so branches that only cover part of the range (e.g. -max..0)
    simply do not contribute there.

    Returned per grid point:
      mean   : average over the branches that cover that point
      sigma  : uncertainty of that mean (see below)
      n      : number of branches that cover that point
      M      : the interpolated branches themselves, shape (n_branches, len(grid))

    sigma is the larger of two estimates:
      sig_pt   = typical per-point noise / sqrt(n)            (always available)
      sig_loop = std between the branches / sqrt(n)           (only with n >= 2)
    sig_loop includes everything that changes from loop to loop (drift, temperature,
    history) and is the more honest error when there are several loops.
    """
    M = np.array([np.interp(grid, b.B, b.y, left=np.nan, right=np.nan) for b in branches])
    # Matching matrix of the per-point noise of each branch (NaN where the branch is absent).
    S = np.array([np.where(np.isnan(m), np.nan, b.noise) for m, b in zip(M, branches)])
    n = np.sum(~np.isnan(M), axis=0)
    with warnings.catch_warnings():
        # nanmean/nanstd warn on grid points that no branch covers (all NaN). That is
        # expected; the result is NaN there and is filtered out later.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(M, axis=0)
        sig_pt = np.sqrt(np.nanmean(S ** 2, axis=0)) / np.sqrt(np.maximum(n, 1))
        sig_loop = np.where(n >= 2, np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
    # np.fmax ignores NaN: where sig_loop is NaN (only one branch) sig_pt is used.
    return mean, np.fmax(sig_pt, sig_loop), n, M


def trapz_weights(x: np.ndarray) -> np.ndarray:
    """Weights w so that sum(w * f) is the trapezoid-rule integral of f over x.

    Having explicit weights (instead of calling np.trapezoid) lets us propagate the error:
    the integral of independent values f_j with errors s_j has error sqrt(sum((w_j*s_j)^2)).
    Each inner point gets half the distance between its two neighbours; the end points get
    half the distance to their single neighbour.
    """
    w = np.zeros_like(x)
    if len(x) > 1:
        w[1:-1] = (x[2:] - x[:-2]) / 2
        w[0] = (x[1] - x[0]) / 2
        w[-1] = (x[-1] - x[-2]) / 2
    return w


def slope3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dy/dx, 3-point smoothed; the two end points (one-sided, noisy) are left NaN.

    np.gradient uses central differences inside and one-sided differences at the ends. The
    one-sided ones amplify noise, and at a turning point the end value is often off, so the
    first and last two values are left out. A 3-point moving average then smooths the rest,
    because differentiating noisy data makes the noise much worse.
    """
    out = np.full_like(y, np.nan)
    if len(y) < 5:
        return out
    g = np.gradient(y, x)
    out[2:-2] = (g[1:-3] + g[2:-2] + g[3:-1]) / 3
    return out


def branch_shift(grid, rise, fall):
    """dB such that falling(B - dB) ~ rising(B). Positive = falling branch displaced to
    negative field (classic ferromagnetic loop; dB ~ 2 Hc). Returns (dB, at_search_edge).

    Method: brute-force search. For each trial shift dB, move the falling curve sideways by
    dB, compare it with the rising curve where both exist, and compute the mean squared
    difference ("cost"). The shift with the lowest cost wins.

    Search range: +/- a quarter of the field range, in 801 steps. A trial shift only counts
    if the shifted curves still overlap on at least half the rising points (and >= 5),
    otherwise a large shift could "win" just by comparing very few points.

    at_search_edge is True when the best shift lies at (or next to) the end of the searched
    range or of the valid region. Then the true minimum may lie outside the range and the
    value should not be trusted.
    """
    # Remove grid points where a curve is NaN (no data there).
    okr, okf = ~np.isnan(rise), ~np.isnan(fall)
    gr, r, gf, f = grid[okr], rise[okr], grid[okf], fall[okf]
    if len(gr) < 5 or len(gf) < 5:
        return math.nan, False
    R = (grid[-1] - grid[0]) / 4
    deltas = np.linspace(-R, R, 801)
    cost = np.full_like(deltas, np.nan)
    for i, dl in enumerate(deltas):
        # Value of the falling curve at B - dl, for every rising grid point B.
        fs = np.interp(gr - dl, gf, f, left=np.nan, right=np.nan)
        m = ~np.isnan(fs)
        if m.sum() >= max(5, len(gr) // 2):
            cost[i] = np.mean((fs[m] - r[m]) ** 2)
    if np.all(np.isnan(cost)):
        return math.nan, False
    i = int(np.nanargmin(cost))
    edge = i < 5 or i > len(deltas) - 6 or np.isnan(cost[max(i - 5, 0)]) or np.isnan(cost[min(i + 5, len(cost) - 1)])
    return float(deltas[i]), bool(edge)


# =====================================================================================
#  Analysis
# =====================================================================================
def analyse(csv_path: Path, args) -> dict:
    """Run the full analysis for one CSV file.

    Returns (results, plot_data): `results` is a flat dict of numbers and text that is
    printed and saved as JSON; `plot_data` holds the arrays make_plot() needs.
    Raises ValueError if the file cannot be analysed (the caller then skips it).
    """
    cols, meta = load(csv_path)
    cfg = meta.get("config", {})
    shunt_lab = cfg.get("shunt_label", "") if "B_shunt_mT" in cols else ""
    sig = pick_signal(cols, args.signal, args.supply, shunt_lab)
    t, loop, sp = cols["t_s"], cols["loop"], cols["setpoint"]
    notes = []                                       # remarks shown with the result
    # Field axis. "shunt" uses B_shunt_mT, the field of the coil current measured over the
    # shunt resistor: the real current, averaged over many samples. "set" uses B_set_mT, the
    # field the current SETPOINT asks for. The EA-PS current read-back ("read", B_mT) is
    # unusable for this: it reads 0 A below ~0.05 A and scatters by ~10 mA above, while the
    # real current follows the setpoint smoothly (the sensor shows it). A fixed setpoint error
    # is the same on both branches, so it cannot fake hysteresis; read-back scatter can.
    # "auto" (default) = shunt if the file has it, else set. Voltage-mode runs have no
    # B_set_mT (all NaN) and fall back to the read-back.
    field = args.field
    if field == "auto":
        field = "shunt" if "B_shunt_mT" in cols else "set"
    if field == "shunt" and "B_shunt_mT" not in cols:
        raise ValueError("no B_shunt_mT column in this file (run without the shunt): use --field set")
    if field == "set" and not np.all(np.isfinite(cols.get("B_set_mT", [np.nan]))):
        field = "read"
        notes.append("no setpoint field in this file (voltage mode or old file): using read-back B_mT")
    # --b-offset adds a constant, e.g. the Earth-field component along the coil axis, which
    # the sweep script does not know about.
    B = cols[FIELD_COL[field]] + args.b_offset
    V = cols[sig]
    Vstd = cols.get(sig + "_std", np.zeros_like(V))   # std of the raw readings per point
    n_avg = n_averaged(meta, sig)

    # ---- signal: ratiometric if possible ---------------------------------------------
    # A Wheatstone bridge output is proportional to its supply voltage. Dividing by the
    # supply (output in mV per V of supply) removes the effect of supply changes.
    have_supply = args.supply in cols and args.supply != sig
    if have_supply:
        S = cols[args.supply]
        Sstd = cols.get(args.supply + "_std", np.zeros_like(S))
        if np.median(np.abs(S)) < np.median(np.abs(V)):  # a bridge output cannot exceed its supply
            # The two myDAQ inputs were wired the other way round in this run.
            V, S, Vstd = S, V, Sstd
            notes.append(f"'{sig}' and '{args.supply}' look swapped (output > supply): swapped back")
        if np.median(np.abs(S)) < 0.5:
            # Supply not connected or not measured: dividing by ~0 would be meaningless.
            have_supply = False
            notes.append(f"'{args.supply}' is ~0 V: using the raw signal, not the ratio")

    ratio = not args.no_ratio and have_supply
    if ratio:
        # V / S is in V/V; * 1e3 gives mV/V. The supply's own noise is ignored in the error.
        y_raw, ystd, unit = V / S * 1e3, Vstd / S * 1e3, "mV/V"
        ylabel = f"{sig} / {args.supply} (mV/V)"
    else:
        y_raw, ystd, unit = V * 1e3, Vstd * 1e3, "mV"   # volts -> millivolts
        ylabel = f"{sig} (mV)"
    # Standard error of each point: std of the raw readings / sqrt(number averaged).
    sem = ystd / math.sqrt(max(n_avg, 1))

    # ---- drift correction ---------------------------------------------------------------
    # y_raw is kept for the time plot and the start/end drift figure; y is used from here on.
    model, drift_desc = drift_model(t, y_raw, loop, sp, args.drift)
    y = y_raw - model

    # ---- branches ------------------------------------------------------------------
    raw_br = split_branches(sp)        # (direction, first row, last row) for every branch
    branches = []
    for d, a, b in raw_br:
        br = make_branch(d, np.arange(a, b + 1), sp, B, y, sem)
        if br is not None:
            branches.append(br)
    # The first branch starts from whatever magnetic state the sensor was left in ("virgin"
    # curve), so it can differ from all later ones. Drop it if a later branch of the same
    # direction covers the same setpoints; otherwise keep it, as it is the only data there.
    if branches and not args.keep_first:
        first = branches[0]
        if any(b.direction == first.direction and b.sp_lo <= first.sp_lo and b.sp_hi >= first.sp_hi
               for b in branches[1:]):
            branches = branches[1:]
            notes.append("first branch dropped (unknown magnetic history); --keep-first to keep it")
    # Group the branches by direction.
    by_dir = {d: [b for b in branches if b.direction == d] for d in (RISE, FALL)}
    if not by_dir[RISE] or not by_dir[FALL]:
        raise ValueError("need at least one rising and one falling branch")

    # Warn about measurements that cannot show (or cannot judge) hysteresis well.
    if sp.min() >= 0 or sp.max() <= 0:
        notes.append("unipolar sweep: the field never crosses zero, so the magnetization is never "
                     "reversed; switching hysteresis cannot appear (use bipolar=True)")
    if len(by_dir[RISE]) < 2 or len(by_dir[FALL]) < 2:
        notes.append("only one branch per direction: errors come from the point-to-point scatter, "
                     "not loop-to-loop repeatability (use loops >= 3)")

    # ---- common grid ---------------------------------------------------------------
    # Rising and falling points are not at exactly the same fields (the measured current
    # differs slightly), so both are interpolated onto one shared field grid.
    # Grid spacing = the typical field step of the data, so the grid points are roughly
    # independent (a finer grid would only add interpolated, correlated points).
    dB = np.abs(np.diff(np.concatenate([b.B for b in branches])))
    step = float(np.median(dB[dB > 1e-9]))
    # Grid range = the field range covered by BOTH directions, so the difference exists.
    lo = max(min(b.B[0] for b in by_dir[d]) for d in (RISE, FALL))
    hi = min(max(b.B[-1] for b in by_dir[d]) for d in (RISE, FALL))
    grid = np.linspace(lo, hi, max(int(round((hi - lo) / step)) + 1, 2))

    # Average curve, its error, branch count and the individual branches, per direction.
    mean, sigma, nbr, mats = {}, {}, {}, {}
    for d in (RISE, FALL):
        mean[d], sigma[d], nbr[d], mats[d] = combine(by_dir[d], grid)

    # ---- metrics -------------------------------------------------------------------
    # Difference falling - rising at each grid point, and its error (errors of the two
    # curves added in quadrature: sqrt(a^2 + b^2)).
    diff = mean[FALL] - mean[RISE]
    sd = np.hypot(sigma[FALL], sigma[RISE])
    # Keep only grid points where both curves and the error exist.
    ok = ~np.isnan(diff) & ~np.isnan(sd) & (sd > 0)
    g, dv, s = grid[ok], diff[ok], sd[ok]
    n = len(g)
    if n < 3:
        raise ValueError("rising and falling branches barely overlap")

    # Loop area = integral of (falling - rising) over B, with propagated error.
    # Positive area: the falling branch lies above the rising branch on average.
    w = trapz_weights(g)
    area, area_err = float(np.sum(w * dv)), float(math.sqrt(np.sum((w * s) ** 2)))
    # Weighted mean of the difference (weights 1/sigma^2): a constant offset between the
    # branches, and its standard error.
    wt = 1 / s ** 2
    off = float(np.sum(wt * dv) / np.sum(wt))
    off_err = float(1 / math.sqrt(np.sum(wt)))
    # Reduced chi-squared of the difference:
    #   chi2       : tests "difference = 0 everywhere"           (n degrees of freedom)
    #   chi2_shape : tests "difference = constant offset"        (n - 1 degrees of freedom)
    # About 1 means "consistent with noise". chi2_shape separates real field-dependent
    # hysteresis from a plain offset caused by drift.
    chi2 = float(np.sum((dv / s) ** 2) / n)
    chi2_shape = float(np.sum(((dv - off) / s) ** 2) / (n - 1))
    imax = int(np.argmax(np.abs(dv)))              # grid point with the largest |difference|
    span = float(np.nanmax(y) - np.nanmin(y))      # total signal range, for scale

    # Noise-equivalent field: the field change that produces a signal change equal to the
    # typical error of the difference. Uses the slope on the steep part of the curve
    # (90th percentile of |slope|), since a saturating curve is flat almost everywhere.
    avg = (mean[RISE] + mean[FALL]) / 2
    slope_avg = np.gradient(avg, grid)
    steep = float(np.nanpercentile(np.abs(slope_avg), 90))  # slope on the steep part of the curve
    nef = float(np.median(s) / steep) if steep > 0 else math.nan

    # Sideways shift between the branches (about 2 x coercive field for a switching loop).
    shift, shift_edge = branch_shift(grid, mean[RISE], mean[FALL])

    # Field of the steepest slope on each branch. For a switching loop these are the
    # switching fields (about +Hc on the rising and -Hc on the falling branch).
    slope = {d: slope3(grid, mean[d]) for d in (RISE, FALL)}
    peak_B = {}
    for d in (RISE, FALL):
        m = ~np.isnan(slope[d])
        peak_B[d] = float(grid[m][np.argmax(np.abs(slope[d][m]))]) if m.any() else math.nan

    # Signal at B = 0 on both branches (remanence), only if the grid spans zero.
    rem = None
    if g[0] < 0 < g[-1]:
        r0 = float(np.interp(0, g, mean[RISE][ok]))
        f0 = float(np.interp(0, g, mean[FALL][ok]))
        rem = {"rising": r0, "falling": f0, "diff": f0 - r0, "diff_err": float(np.interp(0, g, s))}

    # Raw change between the first and last point when both are at the same setpoint:
    # a direct, model-free look at drift over the whole run.
    end_drift = None
    if sp[0] == sp[-1]:
        end_drift = {"delta": float(y_raw[-1] - y_raw[0]), "dt_s": float(t[-1] - t[0]),
                     "sem": float(math.hypot(sem[0], sem[-1]))}

    # Repeatability ("null test"): RMS difference between consecutive branches of the SAME
    # direction. Without hysteresis, rising vs falling should differ by about this much.
    # Only available with >= 2 branches per direction.
    repeat = {}
    for d in (RISE, FALL):
        M = mats[d]
        rms = []
        for i in range(len(M) - 1):
            m = ~np.isnan(M[i]) & ~np.isnan(M[i + 1])   # where both branches have data
            if m.sum() >= 3:
                rms.append(float(np.sqrt(np.mean((M[i + 1][m] - M[i][m]) ** 2))))
        if rms:
            repeat[NAME[d].lower()] = float(np.mean(rms))
    rms_rf = float(np.sqrt(np.mean(dv ** 2)))       # the same measure for rising vs falling

    # ---- verdict -------------------------------------------------------------------
    # For pure noise, chi2/dof has mean 1 and standard deviation sqrt(2 / dof). The threshold
    # is therefore roughly the 3-sigma upper limit of what noise alone produces.
    shape_thr = 1 + 3 * math.sqrt(2 / max(n - 1, 1))  # ~3 sigma for chi2/dof
    z_area, z_off = area / area_err, off / off_err    # significance in units of sigma
    if chi2_shape > shape_thr:
        # The difference changes with field more than noise allows: real hysteresis, or a
        # lag (e.g. the signal not settled yet when read).
        verdict = (f"field-dependent difference between branches exceeds the noise "
                   f"(shape chi2/dof {chi2_shape:.2f} > {shape_thr:.2f}): hysteresis or lag detected")
    elif abs(z_off) > 3:
        # Only a constant offset: the falling branch is shifted up or down as a whole.
        verdict = (f"uniform offset between branches ({z_off:+.1f} sigma) with no field-dependent "
                   f"shape: typical of drift or thermal lag, not magnetic switching")
    else:
        verdict = "no hysteresis distinguishable from noise"

    # Everything that is printed and saved to JSON.
    res = {
        "file": csv_path.name, "signal": ylabel, "unit": unit, "ratiometric": ratio,
        "samples_averaged": n_avg, "drift_correction": drift_desc, "b_offset_mT": args.b_offset,
        "field_source": FIELD_DESC[field],
        "branches": {NAME[d].lower(): len(by_dir[d]) for d in (RISE, FALL)},
        "grid_mT": [float(grid[0]), float(grid[-1]), len(grid)],
        "span": span,
        "noise_per_point_median": float(np.median([b.noise for b in branches])),
        "diff_sigma_median": float(np.median(s)),
        "mean_offset": off, "mean_offset_err": off_err,
        "max_abs_diff": float(dv[imax]), "max_abs_diff_B_mT": float(g[imax]),
        "max_abs_diff_z": float(dv[imax] / s[imax]),
        "loop_area": area, "loop_area_err": area_err,
        # Loop area as a percentage of the box "signal span x field range": a unit-free size.
        "loop_area_pct_of_box": 100 * area / (span * (g[-1] - g[0])) if span > 0 else math.nan,
        "chi2_per_dof": chi2, "chi2_shape_per_dof": chi2_shape, "chi2_shape_threshold": shape_thr,
        "n_points": n, "branch_shift_mT": shift, "branch_shift_at_search_edge": shift_edge,
        "noise_equivalent_field_mT": nef,
        "B_max_slope_mT": {NAME[d].lower(): peak_B[d] for d in (RISE, FALL)},
        "remanence_at_B0": rem, "start_end_drift": end_drift,
        "rms_rising_vs_falling": rms_rf, "rms_same_direction_repeat": repeat or None,
        "verdict": verdict, "notes": notes,
    }
    # Arrays for the figure (not saved to JSON).
    plot_data = dict(t=t, y_raw=y_raw, model=model, B=B, y=y, branches=branches, grid=grid,
                     mean=mean, sigma=sigma, slope=slope, g=g, dv=dv, s=s, ylabel=ylabel,
                     unit=unit, raw_br=raw_br)
    return res, plot_data


# =====================================================================================
#  Report & plot
# =====================================================================================
def fmt(x, p=4):
    """Format a number with sign and p significant digits; 'n/a' for None or NaN."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{p}g}"


def print_report(r: dict):
    """Print the results of one file as an aligned text block (one quantity per line)."""
    u = r["unit"]
    print(f"\n=== {r['file']} ===")
    print(f"signal      : {r['signal']}  (n_avg {r['samples_averaged']}, drift: {r['drift_correction']})")
    print(f"branches    : {r['branches']}   grid {r['grid_mT'][0]:.3f} .. {r['grid_mT'][1]:.3f} mT "
          f"({r['grid_mT'][2]} pts)")
    print(f"span        : {r['span']:.4g} {u}   per-point noise ~ {r['noise_per_point_median']:.3g} {u}")
    print(f"fall - rise : mean {fmt(r['mean_offset'])} +/- {r['mean_offset_err']:.2g} {u}, "
          f"max {fmt(r['max_abs_diff'])} {u} at {r['max_abs_diff_B_mT']:.3f} mT "
          f"({r['max_abs_diff_z']:+.1f} sigma)")
    print(f"loop area   : {fmt(r['loop_area'])} +/- {r['loop_area_err']:.2g} {u}*mT "
          f"({fmt(r['loop_area_pct_of_box'], 3)} % of span x field range)")
    print(f"chi2/dof    : {r['chi2_per_dof']:.2f} total, {r['chi2_shape_per_dof']:.2f} with offset removed "
          f"(3-sigma threshold {r['chi2_shape_threshold']:.2f}, n={r['n_points']})")
    edge = "  [at search edge: unreliable]" if r["branch_shift_at_search_edge"] else ""
    print(f"field shift : {fmt(r['branch_shift_mT'])} mT (falling vs rising){edge}; "
          f"noise-equivalent field (steep part) {fmt(r['noise_equivalent_field_mT'], 2)} mT")
    pb = r["B_max_slope_mT"]
    print(f"max |dV/dB| : rising at {fmt(pb['rising field'], 3)} mT, falling at {fmt(pb['falling field'], 3)} mT")
    # The next lines only appear when the quantity could be computed.
    if r["remanence_at_B0"]:
        m = r["remanence_at_B0"]
        print(f"at B = 0    : rising {m['rising']:.4g}, falling {m['falling']:.4g}, "
              f"diff {fmt(m['diff'])} +/- {m['diff_err']:.2g} {u}")
    if r["start_end_drift"]:
        e = r["start_end_drift"]
        print(f"start->end  : {fmt(e['delta'])} {u} over {e['dt_s']:.0f} s (raw, same setpoint)")
    rep = r["rms_same_direction_repeat"]
    print(f"rms diff    : rising vs falling {r['rms_rising_vs_falling']:.3g} {u}"
          + (f"; same direction (null) {', '.join(f'{k} {v:.3g}' for k, v in rep.items())} {u}" if rep else ""))
    for n_ in r["notes"]:
        print(f"note        : {n_}")
    print(f"VERDICT     : {r['verdict']}")


def make_plot(res: dict, p: dict, png: Path, show: bool):
    """Save a 2 x 2 figure for one file (and optionally show it).

      (a) top left     signal vs field: raw points, branch averages, +/-1 sigma bands
      (b) top right    falling - rising with error bars, mean offset, key numbers
      (c) bottom left  slope dV/dB of both branches (peaks = switching fields)
      (d) bottom right raw signal vs time, coloured by direction, with the drift model
    """
    try:
        import matplotlib
        if not show:
            # "Agg" renders to files only, without opening a window (works without a display).
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot)")
        return
    # Quiet styling: light grid, no top/right frame, grey axis text, left-aligned titles.
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
    u, grid = p["unit"], p["grid"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.8))
    (a, b), (c, d) = ax

    # (a) loop
    # Faint dots: every branch as measured. Lines and bands: the per-direction average +/- 1 sigma.
    for br in p["branches"]:
        a.plot(br.B, br.y, "o", ms=3, color=COLOR[br.direction], alpha=0.35, mec="none")
    for k in (RISE, FALL):
        m, s = p["mean"][k], p["sigma"][k]
        a.fill_between(grid, m - s, m + s, color=COLOR[k], alpha=0.18, lw=0)
        a.plot(grid, m, color=COLOR[k], lw=1.6, label=NAME[k])
    a.set(title="Signal vs field", xlabel="B (mT)", ylabel=p["ylabel"])
    a.legend(loc="best")

    # (b) difference
    # Zero line = no hysteresis. Dashed line = weighted mean offset.
    b.axhline(0, color=AXISC, lw=1)
    b.errorbar(p["g"], p["dv"], yerr=p["s"], fmt="o", ms=3.5, color=INK2, ecolor=MUTED,
               elinewidth=0.8, capsize=0)
    b.axhline(res["mean_offset"], color=INK2, lw=1, ls="--")
    txt = (f"mean (dashed) {res['mean_offset']:+.3g} ± {res['mean_offset_err']:.2g} {u}\n"
           f"loop area {res['loop_area']:+.3g} ± {res['loop_area_err']:.2g} {u}·mT\n"
           f"χ²/dof {res['chi2_per_dof']:.2f}  (shape {res['chi2_shape_per_dof']:.2f}, "
           f"3σ limit {res['chi2_shape_threshold']:.2f})\n"
           f"field shift {fmt(res['branch_shift_mT'], 3)} mT")
    b.text(0.02, 0.98, txt, transform=b.transAxes, va="top", ha="left", fontsize=8, color=INK2,
           bbox=dict(boxstyle="round,pad=0.4", fc="#fcfcfb", ec=GRIDC), zorder=5)
    # Extend the y axis upwards so the text box does not cover data points.
    lo_, hi_ = b.get_ylim()
    b.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))  # head-room for the text box
    b.set(title="Falling − rising  (±1σ)", xlabel="B (mT)", ylabel=f"Δ ({u})")

    # (c) slope
    c.axhline(0, color=AXISC, lw=1)
    for k in (RISE, FALL):
        c.plot(grid, p["slope"][k], color=COLOR[k], lw=1.6, label=NAME[k])
    c.set(title="Slope dV/dB (3-point smoothed, end points omitted)", xlabel="B (mT)",
          ylabel=f"dV/dB ({u}/mT)")
    c.legend(loc="best")

    # (d) time trace with direction colour
    # Uses the raw (uncorrected) signal, so the drift model can be judged against it.
    t, yr = p["t"], p["y_raw"]
    d.plot(t, yr, color=AXISC, lw=0.8, zorder=1)               # thin grey line joining the points
    for dirn, a0, a1 in p["raw_br"]:
        d.plot(t[a0:a1 + 1], yr[a0:a1 + 1], "o", ms=3, color=COLOR[dirn], mec="none", zorder=2)
    if np.any(p["model"]):
        # The model is 0 at t[0]; adding the first value puts it on top of the data.
        d.plot(t, yr[0] + p["model"], color=INK2, lw=1, ls="--", label="drift model")
        d.legend(loc="best")
    d.set(title="Raw signal vs time", xlabel="t (s)", ylabel=p["ylabel"])

    # Two-line heading: file name and drift setting, then the verdict.
    fig.text(0.01, 0.985, f"{res['file']}   ·   drift correction: {res['drift_correction']}",
             fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.955, res["verdict"], fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))   # leave the top 6 % free for the heading
    fig.savefig(png, dpi=150)
    print(f"plot        : {png}")
    if show:
        plt.show()
    plt.close(fig)   # free memory when looping over many files


def _json_clean(o):
    """Replace NaN and infinity with None (JSON null), recursively.

    Python's json module would otherwise write NaN, which is not valid JSON and breaks
    other readers.
    """
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    return o


# =====================================================================================
#  CLI
# =====================================================================================
def main(argv=None):
    """Command-line entry point: choose the files, analyse each one, write the outputs."""
    ap = argparse.ArgumentParser(description="Hysteresis analysis of AMR Helmholtz-coil sweeps")
    ap.add_argument("csv", nargs="*", help="CSV file(s); default: newest in ./data")
    ap.add_argument("--all", action="store_true", help="analyse every CSV in ./data")
    ap.add_argument("--signal", help="signal column (default V_bridge_out)")
    ap.add_argument("--supply", default="V_bridge_supply", help="supply column for ratiometric signal")
    ap.add_argument("--no-ratio", action="store_true", help="use the raw signal, not signal/supply")
    ap.add_argument("--drift", choices=["auto", "none", "ends", "loops"], default="auto",
                    help="linear-in-time drift correction: 'ends' = first & last point, "
                         "'loops' = end point of every loop, 'auto' = loops if >= 2 loops else none")
    ap.add_argument("--b-offset", type=float, default=0.0,
                    help="mT added to the field (e.g. Earth-field component along the coil axis)")
    ap.add_argument("--field", choices=["auto", "shunt", "set", "read"], default="auto",
                    help="field from the shunt current (B_shunt_mT), the current setpoint "
                         "(B_set_mT) or the PSU read-back current (B_mT, reads 0 below ~0.05 A "
                         "and scatters ~10 mA); 'auto' (default) = shunt if measured, else set")
    ap.add_argument("--keep-first", action="store_true", help="keep the first (virgin) branch")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: next to each CSV)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--show", action="store_true", help="open the plot window")
    args = ap.parse_args(argv)

    # Which files to analyse.
    if args.all:
        files = sorted(DATA_DIR.glob("*.csv"))
    elif args.csv:
        files = []
        for f in args.csv:
            # Accept paths relative to the current folder or to the script folder.
            pth = Path(f)
            files.append(pth if pth.exists() else HERE / f)
    else:
        # No file given: take the most recently modified CSV in ./data.
        files = sorted(DATA_DIR.glob("*.csv"), key=lambda q: q.stat().st_mtime)[-1:]
    if not files:
        ap.error(f"no CSV files found (looked in {DATA_DIR})")

    for f in files:
        try:
            res, pdata = analyse(f, args)
        except Exception as e:
            # One bad file (e.g. no signal column) must not stop the others.
            print(f"\n=== {f.name} ===\nskipped: {e}")
            continue
        print_report(res)
        out = args.out_dir or f.parent
        out.mkdir(parents=True, exist_ok=True)
        js = out / f"{f.stem}_hysteresis.json"
        js.write_text(json.dumps(_json_clean(res), indent=2))
        print(f"results     : {js}")
        if not args.no_plot:
            make_plot(res, pdata, out / f"{f.stem}_hysteresis.png", args.show)


if __name__ == "__main__":
    main()
