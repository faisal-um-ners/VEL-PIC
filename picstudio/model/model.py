"""Model (project) object: load/save JSON, apply defaults, validate."""
from __future__ import annotations

import copy
import json
import os

from . import schema as S
from ..expr import Evaluator, ExpressionError

FILE_VERSION = 1


def default_boundaries(geometry="2D-ZY"):
    b = {side: S.defaults(S.BOUNDARY) for side in S.SIDES}
    if geometry == "2D-RZ":
        b["y_min"]["field"] = "axis"
        b["y_min"]["particles"] = "reflect"
    return b


def new_model(geometry="2D-ZY", solver="electrostatic", name="Untitled"):
    m = {
        "version": FILE_VERSION,
        "name": name,
        "description": "",
        "parameters": {},
        "boundaries": default_boundaries(geometry),
    }
    for sec, fields in S.SINGLE_SECTIONS.items():
        m[sec] = S.defaults(fields)
    for sec in S.LIST_SECTIONS:
        m[sec] = []
    m["simulation"]["geometry"] = geometry
    m["simulation"]["solver"] = solver
    if geometry == "2D-RZ":
        m["domain"]["y_min"] = "0"
    return m


def complete(model: dict) -> dict:
    """Return a copy of ``model`` with every missing entry filled by defaults."""
    m = copy.deepcopy(model)
    m.setdefault("version", FILE_VERSION)
    m.setdefault("name", "Untitled")
    m.setdefault("description", "")
    m.setdefault("parameters", {})
    for sec, fields in S.SINGLE_SECTIONS.items():
        d = S.defaults(fields)
        d.update(m.get(sec, {}) or {})
        m[sec] = d
    geom = m["simulation"]["geometry"]
    b = default_boundaries(geom)
    for side in S.SIDES:
        b[side].update((m.get("boundaries", {}) or {}).get(side, {}) or {})
    m["boundaries"] = b
    _migrate_line_probes(m)
    for sec, fields in S.LIST_SECTIONS.items():
        items = []
        for it in m.get(sec, []) or []:
            d = S.defaults(fields)
            d.update(it)
            items.append(d)
        m[sec] = items
    return m


def _migrate_line_probes(m):
    """Files from 1.1.0 kept line-extremum probes in 'probes' (kind = line minimum/maximum)."""
    keep, lines = [], list(m.get("line_probes", []) or [])
    for p in m.get("probes", []) or []:
        kind = p.get("kind", "point")
        if kind == "point":
            q = {k: v for k, v in p.items() if k not in ("kind", "z_end", "y_end")}
            keep.append(q)
            continue
        z0, y0 = p.get("z", "0"), p.get("y", "0")
        z1, y1 = p.get("z_end", z0), p.get("y_end", y0)
        along_y = str(z1).strip() == str(z0).strip() and str(y1).strip() != str(y0).strip()
        lines.append({"name": p.get("name", "line"), "quantity": p.get("quantity", "phi"),
                      "extremum": "max" if kind == "line maximum" else "min",
                      "along": "y" if along_y else "z", "y": y0, "z": z0,
                      "start": y0 if along_y else z0, "end": y1 if along_y else z1})
    if lines or "probes" in m:
        m["probes"] = keep
        m["line_probes"] = lines


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        m = json.load(fh)
    m = complete(m)
    m["_path"] = os.path.abspath(path)
    return m


def save(model: dict, path: str):
    m = {k: v for k, v in model.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)


def evaluator(model: dict) -> Evaluator:
    return Evaluator(model.get("parameters", {}))


def validate(model: dict) -> list[tuple[str, str]]:
    """Return a list of (level, message); level in {'error', 'warning', 'info'}."""
    msgs: list[tuple[str, str]] = []
    m = complete(model)
    try:
        ev = evaluator(m)
    except ExpressionError as exc:
        return [("error", f"Parameters: {exc}")]
    sim, dom = m["simulation"], m["domain"]
    geom, solver = sim["geometry"], sim["solver"]

    def num(sec, key, val):
        try:
            return ev.num(val)
        except Exception as exc:  # noqa: BLE001
            msgs.append(("error", f"{sec}.{key}: {exc}"))
            return None

    zmin, zmax = num("domain", "z_min", dom["z_min"]), num("domain", "z_max", dom["z_max"])
    nz = num("domain", "nz", dom["nz"])
    if None not in (zmin, zmax) and zmax <= zmin:
        msgs.append(("error", "Domain: z_max must exceed z_min"))
    if nz is not None and nz < 4:
        msgs.append(("error", "Domain: need at least 4 cells in z"))
    if geom == "1D-Z":
        for s_ in m.get("species", []):
            if s_.get("motion", "full") == "y_only":
                msgs.append(("error", f"Species '{s_.get('name')}': motion y_only needs a 2D geometry "
                                      f"(in 1D-Z particles move along z; use z_only)"))
    if geom != "1D-Z":
        ymin, ymax = num("domain", "y_min", dom["y_min"]), num("domain", "y_max", dom["y_max"])
        ny = num("domain", "ny", dom["ny"])
        if None not in (ymin, ymax) and ymax <= ymin:
            msgs.append(("error", "Domain: y_max must exceed y_min"))
        if geom == "2D-RZ" and ymin is not None and ymin < 0:
            msgs.append(("error", "RZ: r_min must be >= 0"))
        if ny is not None and ny < 2:
            msgs.append(("error", "Domain: need at least 2 cells in y"))
        if nz and ny and nz * ny > 4e6:
            msgs.append(("warning", f"Large mesh: {int(nz*ny)} cells"))
    b = m["boundaries"]
    for a, c in (("z_min", "z_max"), ("y_min", "y_max")):
        if geom == "1D-Z" and a == "y_min":
            continue
        pa, pc = b[a]["field"] == "periodic", b[c]["field"] == "periodic"
        if pa != pc:
            msgs.append(("error", f"Boundaries: {a} and {c} must both be periodic"))
    if geom == "2D-RZ" and ev.num(dom["y_min"]) == 0 and b["y_min"]["field"] != "axis":
        msgs.append(("warning", "RZ with r_min = 0: y_min boundary treated as axis"))
    names = [s["name"] for s in m["species"]]
    if len(set(names)) != len(names):
        msgs.append(("error", "Species names must be unique"))
    for sec in ("loaders", "emitters"):
        for it in m[sec]:
            if it["species"] not in names:
                msgs.append(("error", f"{sec} '{it['name']}': unknown species '{it['species']}'"))
    if solver == "electrostatic" and m["sources"]:
        msgs.append(("warning", "EM sources are ignored by the electrostatic solver"))
    if not m["species"]:
        msgs.append(("info", "No particle species defined (field-only simulation)"))
    for sec, fields in S.LIST_SECTIONS.items():
        for it in m[sec]:
            for f in fields:
                if f.kind == "expr" and S.is_visible(f, it):
                    v = it.get(f.name)
                    if v in (None, ""):
                        continue
                    try:
                        if not ev.is_time_dependent(v):
                            ev.value(v, z=0.0, y=0.0, r=0.0, t=0.0)
                    except Exception as exc:  # noqa: BLE001
                        msgs.append(("error", f"{sec} '{it.get('name', '')}'.{f.name}: {exc}"))
    try:
        msgs += _beam_limit_warnings(m, ev)
    except Exception:  # noqa: BLE001
        pass
    try:
        msgs += _refinement_warnings(m, ev)
    except Exception:  # noqa: BLE001
        pass
    try:
        msgs += _magnetization_info(m, ev)
    except Exception:  # noqa: BLE001
        pass
    try:
        jf = int(m["simulation"].get("current_filter", 0) or 0)
        if jf > 0 and m["simulation"]["solver"] == "electromagnetic" and \
                any(e.get("model") == "scl" for e in m.get("emitters", [])):
            msgs.append(("warning", "Current filter with space-charge-limited emission: the filter spreads the "
                                    "emitted current into the cathode cells and breaks local charge conservation "
                                    "at the emitting surface, which can drive the Gauss-law emitter unstable. "
                                    "Use 0 filter passes (or the smoothing option) for SCL problems."))
    except Exception:  # noqa: BLE001
        pass
    if not any(l == "error" for l, _ in msgs):
        msgs.append(("info", "Model check passed"))
    return msgs


def _magnetization_info(m, ev):
    """Strong applied B with full motion: dt must resolve the cyclotron motion (w_c dt <= 0.25), which
    can make the run slow; if the gyro-radius is far below the cell size, z_only (infinite-B limit) gives
    the same confined flow at a much larger time step."""
    import math
    out = []
    Bmax = 0.0
    for f in m.get("external_fields", []):
        if f.get("type", "uniform") == "uniform":
            try:
                Bmax = max(Bmax, math.sqrt(sum(ev.num(x) ** 2 for x in f.get("B", ["0", "0", "0"]))) *
                           abs(ev.num(f.get("time_factor", "1"))))
            except Exception:  # noqa: BLE001
                pass
    if Bmax <= 0:
        return out
    full = [s for s in m.get("species", []) if s.get("motion", "full") not in ("z_only", "y_only")]
    if not full:
        return out
    V = 0.0
    for side in m["boundaries"].values():
        if side.get("field") == "conductor":
            try:
                V = max(V, abs(ev.num(side.get("voltage", "0"))))
            except Exception:  # noqa: BLE001
                pass
    me, e, c = 9.1093837015e-31, 1.602176634e-19, 2.99792458e8
    wc = e * Bmax / me
    g = 1 + e * V / (me * c * c)
    v = c * math.sqrt(1 - 1 / (g * g)) if V > 0 else 0.0
    rL = g * me * v / (e * Bmax) if v > 0 else 0.0
    dom = m["domain"]
    try:
        h = ev.num(dom["z_max"]) - ev.num(dom["z_min"])
        h /= max(1.0, ev.num(dom["nz"]))
    except Exception:  # noqa: BLE001
        h = 0.0
    msg = (f"Applied B = {Bmax:g} T with motion 'full': the time step must resolve the electron cyclotron "
           f"motion (w_c = {wc:.3g} rad/s, dt <= 0.25/w_c = {0.25 / wc:.3g} s), so the run takes more steps.")
    if rL > 0 and h > 0 and rL < 0.2 * h:
        msg += (f" The gyro-radius ({rL:.2g} m) is far below the cell size ({h:.2g} m): the flow is effectively "
                f"confined, and Species > Motion = z_only or y_only (infinite-B limit along z or y) gives the same physics without "
                f"this time-step limit.")
    out.append(("info", msg))
    return out


def _refinement_warnings(m, ev):
    """Mesh refinement checks: EM needs a uniform Yee grid; gentle grading (growth ratio <= 1.3)."""
    out = []
    refs = [r for r in m.get("mesh_refinements", []) if r.get("enabled", True)]
    if not refs:
        return out
    if m["simulation"]["solver"] == "electromagnetic":
        out.append(("error", "Mesh Refinement is for the electrostatic solver only: the Yee FDTD solver "
                             "needs a uniform grid (disable the refinement regions)."))
        return out
    for r in refs:
        try:
            gr = ev.num(r.get("growth", 1.2))
            if gr > 1.3:
                out.append(("warning", f"Mesh refinement '{r.get('name', '')}': growth ratio {gr:g} > 1.3 - "
                                       "abrupt cell-size changes lower the field accuracy and add self-forces "
                                       "on particles; 1.05-1.2 is recommended."))
        except Exception:  # noqa: BLE001
            pass
    return out


def _beam_limit_warnings(m, ev):
    """Warn when a prescribed beam current exceeds the space-charge limit of the gap
    (planar Child-Langmuir law with injection energy, Jaffe 1944), which gives a virtual cathode
    at the emitter (nearly all current reflected)."""
    import math
    from .. import constants as C
    out = []
    V = []
    for bd in m["boundaries"].values():
        if bd.get("field") == "conductor":
            V.append(ev.value(bd.get("voltage", "0"), t=0.0))
    for o in m["objects"]:
        if o.get("material") == "conductor":
            V.append(ev.value(o.get("voltage", "0"), t=0.0))
    if len(V) < 2:
        return out
    dV = max(V) - min(V)
    dom = m["domain"]
    species = {s_["name"]: s_ for s_ in m["species"]}
    for e in m["emitters"]:
        if e.get("model") != "beam" or dV <= 0:
            continue
        sp = species.get(e["species"], {})
        q = abs(ev.num(sp.get("charge", "-1"))) * C.e
        mass = ev.num(sp.get("mass", "me"))
        horizontal = e.get("normal", "auto") in ("+y", "-y") or (
            e.get("surface", "segment") == "segment" and ev.num(e.get("z0", "0")) != ev.num(e.get("z1", "0"))
            and m["simulation"]["geometry"] != "1D-Z")
        d = (ev.num(dom["y_max"]) - ev.num(dom["y_min"])) if horizontal else (ev.num(dom["z_max"]) - ev.num(dom["z_min"]))
        Ei = max(ev.value(e.get("energy", "0"), t=0.0), 0.0)
        Jlim = 4 * C.eps0 / 9 * math.sqrt(2 * q / mass) * ((dV + Ei) ** 0.75 + Ei ** 0.75) ** 2 / d ** 2
        # small emitting areas carry more (Luginsland 1996, Lau 2001, Lau et al. 2026): use the
        # first-order upper estimates so that the warning only appears when the limit is surely exceeded
        geom = m["simulation"]["geometry"]
        if geom != "1D-Z" and not horizontal and e.get("surface", "segment") == "segment":
            y0, y1 = sorted((ev.num(e.get("y0", "0")), ev.num(e.get("y1", "0"))))
            ymin, ymax = ev.num(dom["y_min"]), ev.num(dom["y_max"])
            bnd = m["boundaries"]
            if (y1 - y0) < 0.99 * (ymax - ymin):
                Dl = Ei / dV
                if geom == "2D-RZ":
                    Rr = max(y1, 1e-30) / d
                    fac = max(1 + 1 / (4 * Rr), 1.328 / math.sqrt(Rr) * (1 + 0.34 * Rr ** 1.5))
                else:
                    sym = abs(y0 - ymin) < 1e-12 and bnd["y_min"].get("field") == "neumann"
                    wr = max((2 if sym else 1) * (y1 - y0), 1e-30) / d
                    fac = max(1 + 1 / (math.pi * wr), 9 / 8 / math.sqrt(wr) * (1 + 0.413 * math.sqrt(wr)),
                              3.302 * max(Dl, 1e-12) ** 0.5274 / wr)
                Jlim *= max(fac, 1.0)
        J = abs(ev.value(e.get("current_density", "0"), t=0.0))
        if J > 2 * Jlim:
            out.append(("warning", f"Emitter '{e['name']}': beam J = {J:.3g} A/m² is {J / Jlim:.3g} x the "
                        f"space-charge limit of the gap (~{Jlim:.3g} A/m² for {dV:.4g} V over {d:.3g} m). "
                        "A virtual cathode will form at the emitter and almost all current will be "
                        "reflected. Use J below this limit, more energy, or the 'scl' model."))
    return out
