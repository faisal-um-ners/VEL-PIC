"""Cold-test measurement and calibration of the loaded quality factor Q.

A lossy object (material 'lossy', loss_spec 'Q') gets the conductivity
sigma = 2 pi f eps0 / (Q F), where F ('fill') is the fraction of the mode's electric
energy inside the lossy region.  F depends on the geometry, so ``calibrate_q`` runs
short field-only ring-down simulations (no particles): a narrow-band tone burst at the
target frequency excites the structure, and the ring-down of the probe signals is
decomposed into damped modes (harmonic inversion, matrix-pencil method).  The mode
nearest the target frequency gives the loaded Q; 'fill' is rescaled until it equals
the requested Q.

Why harmonic inversion: neighbouring modes can be closer than the loaded line width
(6-cavity planar magnetron: pi mode at 9.05 GHz, 2pi/3 pair at 8.98 GHz, line width
f/Q = 36 MHz at Q = 250).  A spectral-peak pick or the decay of the total field energy
then measures the wrong mode or a mixture.

Why the update is bounded: the loss is not monotonic in sigma.  For sigma << w eps0 the
field crosses the load unperturbed, P ~ sigma and Q ~ fill.  For sigma >> w eps0 the load
behaves like a metal and expels the field (skin depth below the load size), P falls and Q
rises again.  The loaded Q therefore has a minimum near sigma ~ w eps0; every step of
fill is limited to a factor 5, and the calibration stops if Q moves the wrong way.
"""
from __future__ import annotations

import copy

import numpy as np

from ..model.model import complete, evaluator


def _cold_model(model, periods, f0):
    m = copy.deepcopy(complete(model))
    m["species"] = []
    m["loaders"] = []
    m["emitters"] = []
    m["sources"] = []
    m["simulation"]["solver"] = "electromagnetic"
    m["simulation"]["t_end"] = repr(periods / f0)
    m["simulation"]["n_steps"] = ""
    m["simulation"]["smoothing"] = 0
    m["sweep"]["enabled"] = False
    m["diagnostics"].update(history_every="2", field_every="100000000", particle_every="100000000",
                            live_every="100000000")
    m["power_lines"] = []
    m["external_fields"] = []
    for b in m["boundaries"].values():          # cold test: no static electrode fields
        b["voltage"] = "0"
    for o in m["objects"]:
        if o.get("material") == "conductor":
            o["voltage"] = "0"
    return m


def _seed(sim, rng, smooth=12):
    """Smooth random initial field inside the vacuum region (excites the low-order modes)."""
    from ..fields.fdtd import binomial_filter
    y = sim.yee
    for name, mask in (("E0", "m0"), ("E1", "m1"), ("E2", "m2")):
        A = getattr(y, name)
        if A.ndim == 0:
            continue
        R = binomial_filter(rng.standard_normal(A.shape), smooth)
        setattr(y, name, R * getattr(y, mask))


def harmonic_inversion(x, dt, fmin, fmax, max_modes=12):
    """Damped modes of a real signal x(t) in the band [fmin, fmax] (matrix-pencil method).

    The signal is mixed down to the band centre, low-pass filtered (windowed-sinc FIR, valid part),
    decimated to about 4 samples per band width and fitted by a sum of complex exponentials.
    Returns a list of dicts (f [Hz], Q, gamma [1/s, amplitude decay], amp), strongest first."""
    x = np.asarray(x, float)
    N = len(x)
    fc, bw = 0.5 * (fmin + fmax), (fmax - fmin)
    t = np.arange(N) * dt
    z = (x - x.mean()) * np.exp(-2j * np.pi * fc * t)
    # FIR low-pass (windowed sinc, cut-off 0.6 bw) applied in 'valid' mode: an LTI filter maps every
    # damped exponential onto itself, and 'valid' avoids the edge/wrap-around errors of FFT filtering
    fcut = 0.6 * bw * dt
    ntap = int(min(max(8.0 / (bw * dt), 31), N // 3)) | 1
    kk = np.arange(ntap) - ntap // 2
    h = 2 * fcut * np.sinc(2 * fcut * kk) * np.blackman(ntap)
    z = np.convolve(z, h, mode="valid")
    step = max(1, int(1.0 / (4.0 * bw * dt)))
    y = z[::step]
    dts = dt * step
    n = len(y)
    if n < 20:
        return []
    L = n // 3
    H = np.lib.stride_tricks.sliding_window_view(y, L + 1)          # Hankel matrix (n-L) x (L+1)
    _, S, Vh = np.linalg.svd(H, full_matrices=False)
    M = int(min(max_modes, np.sum(S > 1e-4 * S[0])))
    if M < 1:
        return []
    V = Vh[:M].T                     # rows of Vh span the signal vectors (z_k^j)
    lam = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    lam = lam[np.abs(lam) > 1e-12]
    sk = np.log(lam) / dts
    # amplitudes (least squares)
    E = np.exp(np.outer(np.arange(n) * dts, sk))
    amp, *_ = np.linalg.lstsq(E, y, rcond=None)
    modes = []
    for s_, a_ in zip(sk, amp):
        f = fc + s_.imag / (2 * np.pi)
        gam = -s_.real
        if not (fmin <= f <= fmax):
            continue
        Q = np.pi * f / gam if gam > 0 else np.inf
        modes.append({"f": float(f), "Q": float(Q), "gamma": float(gam), "amp": float(abs(a_))})
    modes.sort(key=lambda m_: -m_["amp"])
    return modes


def measure_q(model, f0, band=0.05, drive_periods=60, decay_periods=None, log=print):
    """Return (Q, f) of the mode nearest f0 from a cold-test ring-down.

    The structure is excited by a narrow-band soft source (Gaussian tone burst at f0) on the field
    component of the first probe.  After the burst the signals of all point probes are decomposed
    into damped modes in [f0 (1 - band), f0 (1 + band)]; the mode nearest f0 (among those carrying at
    least 10 % of the strongest amplitude) is taken.  All modes found are kept in measure_q.last_modes."""
    from .simulation import Simulation
    m = complete(model)
    ev = evaluator(m)
    lossy = [o for o in m["objects"] if o.get("material") == "lossy" and o.get("loss_spec", "Q") == "Q"]
    Qt = ev.num(lossy[0]["Q"]) if lossy else 200.0
    if decay_periods is None:
        decay_periods = int(min(max(1.5 * Qt / np.pi, 60), 2000))
    periods = drive_periods + decay_periods
    mc = _cold_model(m, periods, f0)
    if not mc["probes"]:
        raise ValueError("Add at least one probe inside the resonator (e.g. in a cavity) for the Q measurement")
    sim = Simulation(mc, write_output=False, out_dir="/tmp", log=lambda x: None)
    g, y = sim.grid, sim.yee
    pr = mc["probes"][0]
    zq, yq = ev.num(pr["z"]), ev.num(pr["y"])
    comp = {"Ez": "E0", "Ey": "E1", "Ex": "E2"}.get(pr["quantity"], "E1")
    A = getattr(y, comp)
    j = int(round((yq - g.y_min) / g.dy)) if g.dim == 2 else 0
    if g.dim == 2:
        j = min(max(j, 0), A.shape[1] - 1)
    # drive points: the first probe, plus its periodic images with the mode phase (pure-mode excitation)
    obj = lossy[0] if lossy else {}
    per = str(obj.get("cal_period", "") or "").strip()
    Lz = g.z_max - g.z_min
    pts = [(zq, 1.0)]
    if per:
        P_ = ev.num(per)
        ph = ev.num(str(obj.get("cal_phase", "pi") or "pi"))
        nper = int(round(Lz / P_))
        pts = []
        for k in range(nper):
            zk = zq + k * P_
            if zk > g.z_max + 1e-12 * Lz:
                zk -= Lz                       # periodic domain: wrap the image
            if g.z_min - 1e-12 * Lz <= zk <= g.z_max + 1e-12 * Lz:
                pts.append((zk, float(np.cos(k * ph))))
        log(f"  pure-mode drive: {len(pts)} points, period {P_:.4g} m, phase {ph:.4g} rad per period")
    drive_idx = []
    for zk, sg in pts:
        ik = int(round((zk - g.z_min) / g.dz))
        drive_idx.append((min(max(ik, 0), A.shape[0] - 1), sg))
    Td = drive_periods / f0
    t0, tau = 0.5 * Td, Td / 5
    n = sim.n_steps
    log(f"  cold test: {n} steps ({periods} periods at {f0 / 1e9:.4g} GHz)")
    names = {"Ez": 0, "Ey": 1, "Ex": 2, "Bz": 3, "By": 4, "Bx": 5}
    probes = [(names.get(p_["quantity"], 1), ev.num(p_["z"]), ev.num(p_.get("y", "0"))) for p_ in mc["probes"]]
    stride = max(1, int(1.0 / (40.0 * f0 * sim.dt)))      # ~40 samples per period are plenty
    t = np.empty(n // stride + 1)
    sig = np.empty((len(probes), len(t)))
    Wt, Wv = [], []
    ns = 0
    for k in range(n):
        sim.step()
        tk = sim.t
        if tk < Td:
            drive = np.exp(-((tk - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * tk)
            A = getattr(y, comp)
            for ik, sg in drive_idx:
                if g.dim == 2:
                    A[ik, j] += sg * drive
                else:
                    A[ik] += sg * drive
        if k % stride == 0 and ns < len(t):
            t[ns] = tk
            for q_, (ci, zz, yy) in enumerate(probes):
                sig[q_, ns] = sim._interp(sim.F[ci], np.array([zz]), np.array([yy]))[0]
            ns += 1
            if tk > Td:
                Wt.append(tk)
                Wv.append(y.energy())
    t, sig = t[:ns], sig[:, :ns]
    keep = t > Td + 5.0 / f0
    measure_q.last_signals = (t, sig, [p_["name"] for p_ in mc["probes"]], Td)
    dt = float(np.median(np.diff(t)))
    fmin, fmax = f0 * (1 - band), f0 * (1 + band)
    # pool the modes of all probes: same mode = frequency within a quarter line width
    pool = []
    for q_ in range(len(probes)):
        for md in harmonic_inversion(sig[q_, keep], dt, fmin, fmax):
            pool.append(md)
    modes = []
    if pool:
        amax = max(md["amp"] for md in pool)
        strong = [md for md in pool if md["amp"] > 0.10 * amax and np.isfinite(md["Q"]) and md["Q"] > 0]
        groups = []
        for md in sorted(strong, key=lambda m_: m_["f"]):
            if groups and abs(md["f"] - groups[-1][-1]["f"]) < 0.25 * md["f"] / max(md["Q"], 1.0):
                groups[-1].append(md)
            else:
                groups.append([md])
        for gr in groups:
            w = np.array([md["amp"] for md in gr])
            modes.append({"f": float(np.average([md["f"] for md in gr], weights=w)),
                          "Q": float(np.average([md["Q"] for md in gr], weights=w)),
                          "amp": float(w.max() / amax)})
    measure_q.last_modes = modes
    for md in modes:
        log(f"    mode {md['f'] / 1e9:.5g} GHz  Q = {md['Q']:.4g}  (relative amplitude {md['amp']:.2f})")
    # spectral peak of the probe signal (frequency of the excited mode)
    ss = sig[0, keep] - sig[0, keep].mean()
    N = len(ss)
    F = np.abs(np.fft.rfft(ss * np.hanning(N), 16 * N))
    fr = np.fft.rfftfreq(16 * N, dt)
    sel = (fr > fmin) & (fr < fmax)
    fpk = float(fr[sel][np.argmax(F[sel])]) if sel.any() else float(f0)
    if per:
        # pure-mode excitation: the stored energy decays as exp(-w t / Q)
        Wt, Wv = np.array(Wt), np.array(Wv)
        a, b = int(0.15 * len(Wt)), int(0.95 * len(Wt))
        slope, _ = np.polyfit(Wt[a:b], np.log(np.maximum(Wv[a:b], 1e-300)), 1)
        Q = 2 * np.pi * fpk / (-slope) if slope < 0 else np.inf
        log(f"    pure-mode ring-down: stored-energy decay gives Q = {Q:.4g} at {fpk / 1e9:.5g} GHz")
        if modes:                     # the mode-resolved value is preferred (residual modes excluded)
            best = min(modes, key=lambda m_: abs(m_["f"] - fpk))
            return float(best["Q"]), float(best["f"])
        return float(Q), fpk
    if not modes:
        return float("inf"), fpk
    best = min(modes, key=lambda m_: abs(m_["f"] - f0))
    return float(best["Q"]), float(best["f"])


def calibrate_q(model, obj_name=None, tol=0.05, max_iter=6, log=print):
    """Adjust 'fill' of the lossy object so that the measured loaded Q of the mode nearest its
    'frequency' matches its 'Q'.  Returns (updated model, history list of (fill, Q_measured, f_peak)).

    In the perturbative regime (sigma << w eps0, i.e. Q * fill >> 1) the loaded Q is proportional to
    fill, so fill <- fill * Q_target / Q_measured.  Each step is limited to a factor 5.  If Q moves
    the wrong way (fill reduced but Q increased, or the reverse) the load is past its loss maximum
    (it expels the field like a metal): the calibration stops and keeps the best value found."""
    m = copy.deepcopy(complete(model))
    ev = evaluator(m)
    idx = [i for i, o in enumerate(m["objects"]) if o.get("material") == "lossy"
           and o.get("loss_spec", "Q") == "Q" and (obj_name in (None, "", o.get("name")))]
    if not idx:
        raise ValueError("No lossy object with loss_spec = 'Q' found")
    i = idx[0]
    obj = m["objects"][i]
    # several loss ports with the same Q specification act as one load: they share 'fill' (the total
    # electric-energy fraction in all of them) and are calibrated together
    group = [k for k in idx if obj_name not in (None, "") or
             (m["objects"][k].get("Q") == obj.get("Q") and m["objects"][k].get("frequency") == obj.get("frequency"))]
    Qt = ev.num(obj["Q"])
    f0 = ev.num(obj["frequency"])
    fill = ev.num(obj.get("fill", 1))
    hist = []
    for it in range(max_iter):
        for k in group:
            m["objects"][k]["fill"] = repr(float(fill))
        ratio = 1.0 / (Qt * fill)
        if ratio > 1.0:
            log(f"  note: sigma / (w eps0) = 1/(Q fill) = {ratio:.3g} > 1 - the load is a good conductor at this "
                f"fill and partly expels the field; Q is no longer proportional to fill")
        Qm, fpk = measure_q(m, f0, log=log)
        hist.append((fill, Qm, fpk))
        log(f"  iteration {it + 1}: fill = {fill:.4g}  ->  Q = {Qm:.4g} at f = {fpk / 1e9:.5g} GHz "
            f"(target {Qt:.4g})")
        if np.isfinite(Qm) and abs(Qm / Qt - 1) < tol:
            break
        if len(hist) >= 2:
            (f1, q1, _), (f2, q2, _) = hist[-2], hist[-1]
            if np.isfinite(q1) and np.isfinite(q2) and (f2 - f1) * (q2 - q1) < 0:
                log("  stopped: Q changed in the opposite direction to fill - the load is past its loss "
                    "maximum (over-conductive). A loaded Q this low cannot be reached with this load size; "
                    "enlarge the lossy region or raise the target Q.")
                break
        if not np.isfinite(Qm) or Qm <= 0:
            fill *= 0.2
            continue
        fill *= float(np.clip(Qt / Qm, 0.2, 5.0))
    # keep the value whose measured Q is closest to the target
    good = [h for h in hist if np.isfinite(h[1]) and h[1] > 0]
    if good:
        best = min(good, key=lambda h: abs(np.log(h[1] / Qt)))
        if best is not hist[-1]:
            hist.append(best)
        fill = best[0]
    for k in group:
        m["objects"][k]["fill"] = repr(float(fill))
    return m, hist
