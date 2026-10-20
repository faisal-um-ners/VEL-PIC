"""Cold-test measurement and calibration of the loaded quality factor Q.

A lossy object (material 'lossy', loss_spec 'Q') gets the conductivity
sigma = 2 pi f eps0 / (Q F), where F is the fraction of the mode's electric
energy inside the lossy region.  F depends on the geometry, so
``calibrate_q`` runs short field-only ring-down simulations (no particles):
it excites the structure, band-pass filters the probe signals around the mode
frequency, fits the exponential decay of the envelope
(amplitude ~ exp(-w t / 2Q)) and rescales F until the measured loaded Q
equals the requested Q.
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


def measure_q(model, f0, band=0.006, drive_periods=60, decay_periods=None, log=print):
    """Return (Q, f_peak) of the mode closest to f0 from a cold-test ring-down.

    The mode is excited by a narrow-band soft source (Gaussian tone burst at f0) on the field
    component of the first probe; after the burst the band-passed probe envelope decays as
    exp(-pi f t / Q)."""
    from scipy.signal import hilbert
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
    i = int(round((zq - g.z_min) / g.dz))
    j = int(round((yq - g.y_min) / g.dy)) if g.dim == 2 else 0
    A = getattr(y, comp)
    i = min(max(i, 0), A.shape[0] - 1)
    if g.dim == 2:
        j = min(max(j, 0), A.shape[1] - 1)
    Td = drive_periods / f0
    t0, tau = 0.5 * Td, Td / 5
    n = sim.n_steps
    log(f"  cold test: {n} steps ({periods} periods at {f0 / 1e9:.4g} GHz)")
    t = np.empty(n)
    sig = np.empty(n)
    names = {"Ez": 0, "Ey": 1, "Ex": 2, "Bz": 3, "By": 4, "Bx": 5}
    ci = names.get(pr["quantity"], 1)
    # node mask of lossy cells, for the perturbative electric-energy fill factor
    lossy_cell = sim.mats.sigma_cell > 0
    if g.dim == 2:
        lp = np.pad(lossy_cell.astype(float), 1)
        lossy_node = (lp[:-1, :-1] + lp[1:, :-1] + lp[:-1, 1:] + lp[1:, 1:]) > 0
    else:
        lp = np.pad(lossy_cell.astype(float), 1)
        lossy_node = (lp[:-1] + lp[1:]) > 0
    vol = g.node_volume
    acc_in = acc_all = 0.0
    t_acc = Td + 20.0 / f0
    Wt, Wv = [], []
    for k in range(n):
        sim.step()
        tk = sim.t
        if tk < Td:
            drive = np.exp(-((tk - t0) / tau) ** 2) * np.sin(2 * np.pi * f0 * tk)
            A = getattr(y, comp)
            if g.dim == 2:
                A[i, j] += drive
            else:
                A[i] += drive
        t[k] = tk
        sig[k] = sim._interp(sim.F[ci], np.array([zq]), np.array([yq]))[0]
        if k % 20 == 0 and tk > Td:
            Wt.append(tk)
            Wv.append(y.energy())
        if Td < tk < t_acc:
            E2 = (sim.F[0] ** 2 + sim.F[1] ** 2 + sim.F[2] ** 2) * vol
            acc_in += E2[lossy_node].sum()
            acc_all += E2.sum()
    keep = t > Td
    tt, ss = t[keep], sig[keep] - sig[keep].mean()
    N = len(ss)
    F = np.fft.rfft(ss * np.hanning(N), 8 * N)
    f = np.fft.rfftfreq(8 * N, tt[1] - tt[0])
    sel = (f > f0 * 0.9) & (f < f0 * 1.1)
    fpk = f[sel][np.argmax(np.abs(F[sel]))]
    # loaded Q from the decay of the stored field energy after the narrow-band burst:
    # W ~ exp(-w t / Q)  ->  Q = 2 pi f / (-d ln W / dt)
    Wt, Wv = np.array(Wt), np.array(Wv)
    a, b = int(0.1 * len(Wt)), int(0.95 * len(Wt))
    slope, _ = np.polyfit(Wt[a:b], np.log(np.maximum(Wv[a:b], 1e-300)), 1)
    Q = 2 * np.pi * fpk / (-slope) if slope < 0 else np.inf
    F_est = acc_in / acc_all if acc_all > 0 else np.nan
    measure_q.last_fill = float(F_est)
    return float(Q), float(fpk)


def calibrate_q(model, obj_name=None, tol=0.05, max_iter=6, log=print):
    """Adjust 'fill' of the lossy object so that the measured loaded Q matches its 'Q'.
    Returns (updated model, history list of (fill, Q_measured, f_peak))."""
    m = copy.deepcopy(complete(model))
    ev = evaluator(m)
    idx = [i for i, o in enumerate(m["objects"]) if o.get("material") == "lossy"
           and o.get("loss_spec", "Q") == "Q" and (obj_name in (None, "", o.get("name")))]
    if not idx:
        raise ValueError("No lossy object with loss_spec = 'Q' found")
    i = idx[0]
    obj = m["objects"][i]
    Qt = ev.num(obj["Q"])
    f0 = ev.num(obj["frequency"])
    fill = ev.num(obj.get("fill", 1))
    hist = []
    for it in range(max_iter):
        obj["fill"] = repr(fill)
        Qm, fpk = measure_q(m, f0, log=log)
        hist.append((fill, Qm, fpk))
        Fe = getattr(measure_q, "last_fill", np.nan)
        log(f"  iteration {it + 1}: fill = {fill:.4g}  ->  Q = {Qm:.4g} at f = {fpk / 1e9:.4g} GHz "
            f"(target {Qt:.4g}); electric-energy fraction in load = {Fe:.4g}")
        if abs(Qm / Qt - 1) < tol:
            break
        # Q_measured ~ Q_target * fill / F_true (load-dominated) -> F_true = Q_target * fill / Q_measured.
        # Far from the target (or no measurable decay) use the perturbative energy fraction instead.
        if np.isfinite(Qm) and 0.2 < Qm / Qt < 5:
            fill = fill * Qt / Qm
        elif np.isfinite(Fe) and Fe > 0:
            fill = Fe if it == 0 else min(Fe, 0.5 * fill)
        else:
            fill *= 0.1
    obj["fill"] = repr(float(fill))
    obj["frequency"] = obj["frequency"]
    return m, hist
