"""Matplotlib plotting shared by the GUI, the CLI and the benchmark scripts."""
from __future__ import annotations

import numpy as np

from .. import constants as C
from .reader import beam_moments, species_names

MAT_COLORS = {"conductor": "#9aa5b1", "dielectric": "#9ecae1", "absorber": "#fdae6b", "vacuum": "#ffffff",
              "lossy": "#c9b8f0"}
BND_STYLE = {"conductor": dict(color="k", lw=3, ls="-"),
             "neumann": dict(color="#2b8cbe", lw=2, ls=":"),
             "periodic": dict(color="#31a354", lw=2, ls="-."),
             "absorbing": dict(color="#e6550d", lw=3, ls="--"),
             "axis": dict(color="#756bb1", lw=1.5, ls=(0, (8, 3, 2, 3)))}
SPECIES_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def _lbl(geometry):
    return ("z (m)", "r (m)") if geometry == "2D-RZ" else ("z (m)", "y (m)")


def plot_geometry(ax, model, show_mesh=False, painted=False, selected=None):
    """Draw the model set-up: domain, boundaries, objects, emitters, loaders, sources, probes."""
    from matplotlib.patches import Polygon as MplPolygon, Rectangle
    from ..geometry.shapes import Shape
    from ..model.model import complete, evaluator
    m = complete(model)
    ev = evaluator(m)
    geom = m["simulation"]["geometry"]
    d = m["domain"]
    z0, z1 = ev.num(d["z_min"]), ev.num(d["z_max"])
    nz = int(ev.num(d["nz"]))
    ax.clear()
    if geom == "1D-Z":
        y0, y1 = 0.0, 1.0
    else:
        y0, y1 = ev.num(d["y_min"]), ev.num(d["y_max"])
    ny = int(ev.num(d["ny"])) if geom != "1D-Z" else 1
    if painted and geom != "1D-Z":
        try:
            from ..mesh.grid import Materials, grid_from_model
            g = grid_from_model(m, ev)
            mats = Materials(g, m["objects"], m["boundaries"], ev)
            cm = mats.cell_mat.T.astype(float)
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([MAT_COLORS["vacuum"], MAT_COLORS["conductor"], MAT_COLORS["dielectric"],
                                   MAT_COLORS["absorber"]])
            ax.pcolormesh(g.z, g.y, cm, cmap=cmap, vmin=-0.5, vmax=3.5, shading="flat")
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, f"cannot paint mesh: {exc}", transform=ax.transAxes, ha="center")
    else:
        import matplotlib as mpl
        vac = mpl.rcParams["axes.facecolor"]
        ax.add_patch(Rectangle((z0, y0), z1 - z0, y1 - y0, fc=vac, ec="none", zorder=0))
        for i, obj in enumerate(m["objects"]):
            try:
                shp = Shape(obj, ev)
            except Exception:  # noqa: BLE001
                continue
            mat = obj.get("material", "conductor")
            fc = vac if mat == "vacuum" else MAT_COLORS.get(mat, "#cccccc")
            ec = "#d62728" if selected == ("objects", i) else "#444444"
            lw = 2.5 if selected == ("objects", i) else 0.8
            if shp.kind == "expression":
                Z, Y = np.meshgrid(np.linspace(z0, z1, 300), np.linspace(y0, y1, 300))
                ins = shp.contains(Z, Y)
                ax.contourf(Z, Y, ins.astype(float), levels=[0.5, 1.5], colors=[fc], zorder=1)
                continue
            outl = shp.outline()
            if shp.kind == "annulus":
                th = np.linspace(0, 2 * np.pi, 200)
                zc, yc, ri, ro = shp.p
                pts = np.r_[np.c_[zc + ro * np.cos(th), yc + ro * np.sin(th)],
                            np.c_[zc + ri * np.cos(th[::-1]), yc + ri * np.sin(th[::-1])]]
                ax.add_patch(MplPolygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=1 + i * 0.01))
            else:
                for pl in outl:
                    ax.add_patch(MplPolygon(pl, closed=True, fc=fc, ec=ec, lw=lw, zorder=1 + i * 0.01,
                                            hatch="///" if mat == "lossy" else None,
                                            alpha=0.85 if mat == "lossy" else 1.0))
            if outl:
                cz, cy = np.mean(outl[0][:-1], axis=0)
                span = np.ptp(outl[0], axis=0)
                small = span[0] < 0.9 * (z1 - z0) or span[1] < 0.9 * (y1 - y0)
                if z0 <= cz <= z1 and y0 <= cy <= y1 and mat != "vacuum" and small:
                    ax.text(cz, cy, obj.get("name", ""), fontsize=7, ha="center", va="center", zorder=5,
                            color="#222222")
    if show_mesh:
        try:
            from matplotlib.collections import LineCollection
            from ..mesh.grid import grid_from_model
            g = grid_from_model(m, ev)
            import matplotlib as mpl
            dark = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(mpl.rcParams["axes.facecolor"]))[2] < 0.5
            col = "#8fa3bf" if dark else "#5b6b82"
            zs = g.z
            ys = g.y if g.dim == 2 else np.array([y0, y1])
            if hasattr(g, "local_mesh"):
                mesh = g.local_mesh
                segs = mesh.points[mesh.edge_nodes]
                ax.add_collection(LineCollection(segs, colors=col, linewidths=0.3, alpha=0.75, zorder=2.5))
                for region in mesh.regions:
                    ax.add_patch(MplPolygon(region.points, closed=True, fc="#2a78d6", ec="#2a78d6",
                                            alpha=0.12, lw=1, zorder=2))
                note = f"{len(mesh.points):,} nodes, {len(mesh.triangles):,} triangles (fixed local mesh)"
            else:
                # thin out only if there are far more lines than pixels (zoom in to see all of them)
                sz = max(1, len(zs) // 600)
                sy = max(1, len(ys) // 600)
                zl = np.r_[zs[::sz], zs[-1]]
                segs = [[(z, ys[0]), (z, ys[-1])] for z in zl]
                if g.dim == 2:
                    yl = np.r_[ys[::sy], ys[-1]]
                    segs += [[(zs[0], y), (zs[-1], y)] for y in yl]
                ax.add_collection(LineCollection(segs, colors=col, linewidths=0.35, alpha=0.75, zorder=2.5))
                # highlight refinement regions
                for r in m.get("mesh_refinements", []):
                    if not r.get("enabled", True):
                        continue
                    a, b = sorted((ev.num(r["start"]), ev.num(r["end"])))
                    if r.get("axis", "z") == "z":
                        ax.axvspan(a, b, color="#2a78d6", alpha=0.08, zorder=0.5, lw=0)
                    elif g.dim == 2:
                        ax.axhspan(a, b, color="#2a78d6", alpha=0.08, zorder=0.5, lw=0)
                note = f"{g.nz} x {g.ny} cells" if g.dim == 2 else f"{g.nz} cells"
                if sz > 1 or sy > 1:
                    note += "  (every %d-th line drawn; zoom in for all)" % max(sz, sy)
            ax.text(0.99, 0.01, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color=col, zorder=20)
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.05, f"cannot build mesh: {exc}", transform=ax.transAxes, ha="center", fontsize=8)
    # boundaries
    b = m["boundaries"]
    segs = {"z_min": ([z0, z0], [y0, y1]), "z_max": ([z1, z1], [y0, y1]),
            "y_min": ([z0, z1], [y0, y0]), "y_max": ([z0, z1], [y1, y1])}
    for side, (xs, ys) in segs.items():
        if geom == "1D-Z" and side.startswith("y"):
            continue
        f = b[side]["field"]
        if geom == "2D-RZ" and side == "y_min" and abs(y0) < 1e-15:
            f = "axis"
        st = BND_STYLE.get(f, {})
        ax.plot(xs, ys, zorder=6, clip_on=False, **st)
        if f == "conductor" and str(b[side].get("voltage", "0")).strip() not in ("0", "0.0", ""):
            ax.text(np.mean(xs), np.mean(ys), f" V={b[side]['voltage']} ", fontsize=7, color="k",
                    ha="center", va="center", zorder=7,
                    bbox=dict(fc="#ffffcc", ec="none", alpha=0.8, pad=1))
    # loaders
    for i, ld in enumerate(m["loaders"]):
        try:
            lz0, lz1 = ev.num(ld["z0"]), ev.num(ld["z1"])
            ly0, ly1 = (ev.num(ld["y0"]), ev.num(ld["y1"])) if geom != "1D-Z" else (y0, y1)
            ax.add_patch(Rectangle((lz0, ly0), lz1 - lz0, ly1 - ly0, fc="#2ca02c", alpha=0.12,
                                   ec="#2ca02c", ls="--", lw=1.2 if selected != ("loaders", i) else 2.5,
                                   zorder=3))
        except Exception:  # noqa: BLE001
            pass
    # emitters
    for i, em in enumerate(m["emitters"]):
        lw = 4 if selected == ("emitters", i) else 2.5
        try:
            if em.get("surface", "segment") == "object":
                for obj in m["objects"]:
                    if obj.get("name") == em.get("object"):
                        for pl in Shape(obj, ev).outline():
                            ax.plot(pl[:, 0], pl[:, 1], color="#d62728", lw=lw, zorder=8)
                continue
            ez0 = ev.num(em["z0"])
            if geom == "1D-Z":
                ax.plot([ez0, ez0], [y0, y1], color="#d62728", lw=lw, zorder=8)
            else:
                ez1, ey0, ey1 = ev.num(em["z1"]), ev.num(em["y0"]), ev.num(em["y1"])
                ax.plot([ez0, ez1], [ey0, ey1], color="#d62728", lw=lw + 1.5, zorder=8, solid_capstyle="butt",
                        clip_on=False)
                L = np.hypot(ez1 - ez0, ey1 - ey0)
                if L < 0.05 * max(z1 - z0, y1 - y0):      # small emitter: mark it so it is visible
                    ax.plot(0.5 * (ez0 + ez1), 0.5 * (ey0 + ey1), marker=">", color="#d62728", ms=9,
                            zorder=9, clip_on=False)
                    ax.annotate(f"emitter {em.get('name', '')} ({L * 1e3:.3g} mm)",
                                (0.5 * (ez0 + ez1), 0.5 * (ey0 + ey1)), xytext=(10, -12),
                                textcoords="offset points", fontsize=7, color="#a01d1d", zorder=11)
        except Exception:  # noqa: BLE001
            pass
    # EM sources
    for i, src in enumerate(m["sources"]):
        try:
            zs = ev.num(src["z_position"])
            ax.plot([zs, zs], [y0, y1], color="#9467bd", lw=2 if selected != ("sources", i) else 4,
                    ls=(0, (4, 2)), zorder=8)
            ax.annotate("", xy=(zs + 0.05 * (z1 - z0) * (1 if src.get("direction", "+z") == "+z" else -1),
                                0.5 * (y0 + y1)), xytext=(zs, 0.5 * (y0 + y1)),
                        arrowprops=dict(arrowstyle="->", color="#9467bd", lw=2), zorder=9)
        except Exception:  # noqa: BLE001
            pass
    for pr in m["probes"]:
        try:
            pzz, pyy = ev.num(pr["z"]), (ev.num(pr["y"]) if geom != "1D-Z" else 0.5)
            ax.plot(pzz, pyy, "x", color="#ff7f0e", ms=8, mew=2, zorder=10)
            ax.annotate(f"probe {pr.get('name', '')} ({pr.get('quantity', '')})", (pzz, pyy),
                        xytext=(6, 6), textcoords="offset points", fontsize=7, color="#b35900", zorder=11)
        except Exception:  # noqa: BLE001
            pass
    for pr in m.get("line_probes", []):
        try:
            a_, b_ = ev.num(pr.get("start", "0")), ev.num(pr.get("end", "0"))
            if pr.get("along", "z") == "y" and geom != "1D-Z":
                zc = ev.num(pr.get("z", "0"))
                pz0, py0, pz1, py1 = zc, a_, zc, b_
            else:
                yc = ev.num(pr.get("y", "0")) if geom != "1D-Z" else 0.5
                pz0, py0, pz1, py1 = a_, yc, b_, yc
            ax.plot([pz0, pz1], [py0, py1], color="#ff7f0e", lw=2, ls=(0, (3, 2)), zorder=10)
            ax.plot([pz0, pz1], [py0, py1], "|" if pr.get("along", "z") == "z" else "_", color="#ff7f0e",
                    ms=8, mew=2, zorder=10)
            ax.annotate(f"line probe {pr.get('name', '')}: {pr.get('quantity', '')} {pr.get('extremum', 'min')}",
                        (0.5 * (pz0 + pz1), 0.5 * (py0 + py1)), xytext=(0, 7), textcoords="offset points",
                        fontsize=7, color="#b35900", ha="center", zorder=11)
        except Exception:  # noqa: BLE001
            pass
    for pl in m["power_lines"]:
        try:
            pos, a, bb = ev.num(pl["position"]), ev.num(pl["start"]), ev.num(pl["end"])
            if pl["orientation"] == "z" or geom == "1D-Z":
                ax.plot([pos, pos], [a, bb] if geom != "1D-Z" else [y0, y1], color="#17becf", lw=3, zorder=9)
            else:
                ax.plot([a, bb], [pos, pos], color="#17becf", lw=3, zorder=9)
        except Exception:  # noqa: BLE001
            pass
    # legend of the symbols in use
    from matplotlib.lines import Line2D
    handles = []
    used = set()
    for side in ("z_min", "z_max", "y_min", "y_max"):
        if geom == "1D-Z" and side.startswith("y"):
            continue
        f = b[side]["field"]
        if geom == "2D-RZ" and side == "y_min" and abs(y0) < 1e-15:
            f = "axis"
        if f not in used:
            used.add(f)
            st = dict(BND_STYLE.get(f, {}))
            handles.append(Line2D([], [], label=f"{f} boundary", **st))
    if m["emitters"]:
        handles.append(Line2D([], [], color="#d62728", lw=2.5, label="emitter"))
    if m["probes"]:
        handles.append(Line2D([], [], ls="none", marker="x", color="#ff7f0e", mew=2, label="point probe"))
    if m.get("line_probes"):
        handles.append(Line2D([], [], color="#ff7f0e", lw=2, ls=(0, (3, 2)), label="line probe (min/max)"))
    if m["power_lines"]:
        handles.append(Line2D([], [], color="#17becf", lw=3, label="power monitor"))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                  framealpha=0.9)
    xl, yl = _lbl(geom)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl if geom != "1D-Z" else "")
    if geom == "1D-Z":
        ax.set_yticks([])
    ax.set_xlim(z0 - 0.02 * (z1 - z0), z1 + 0.02 * (z1 - z0))
    ax.set_ylim(y0 - 0.02 * (y1 - y0), y1 + 0.02 * (y1 - y0))
    if geom != "1D-Z":
        ratio = (z1 - z0) / max(y1 - y0, 1e-300)
        ax.set_aspect("equal" if 0.25 < ratio < 4 else "auto", adjustable="box")
    ax.grid(False)                         # geometry / mesh view: no axis grid (the mesh is drawn itself)
    ax.set_title(f"{m.get('name', '')}  [{geom}, {m['simulation']['solver']}]", fontsize=9)


def field_names(fdata):
    skip = {"t", "step", "z", "y", "geometry", "cell_mat"}
    return [k for k in fdata if k not in skip and not k.startswith(("history", "line_")) and "__" not in k
            and k != "progress" and np.ndim(fdata[k]) >= 1 and np.size(fdata[k]) > 1]


def masked_field(fdata, qty):
    """Field values with the inside of conductors set to NaN (a PEC carries no field: nothing is
    shown there).  Conductor surfaces keep their physical value."""
    A = np.array(fdata[qty], dtype=float)
    cm = fdata.get("cell_mat")
    if cm is not None and np.ndim(cm) == A.ndim:
        from ..mesh.grid import conductor_interior_nodes
        inner = conductor_interior_nodes(cm)
        if inner.shape == A.shape:
            A[inner] = np.nan
    return A


def _conductor_cover(ax, fdata, z, y, mirror):
    """Paint the conductor cells with the axes background colour so that the field map shows
    nothing inside the metal (sharp at the conductor surface)."""
    cm = fdata.get("cell_mat")
    if cm is None or np.ndim(cm) != 2:
        return
    from matplotlib.colors import ListedColormap
    c = (np.asarray(cm) == 1)
    if mirror:
        c = np.concatenate([c[:, ::-1], c], axis=1)
    if not c.any() or c.shape != (len(z) - 1, len(y) - 1):
        return
    M = np.ma.masked_where(~c.T, np.ones(c.T.shape))
    ax.pcolormesh(z, y, M, cmap=ListedColormap([ax.get_facecolor()]), vmin=0, vmax=2, shading="flat",
                  zorder=2, rasterized=True)


def plot_field(ax, fdata, qty, cmap=None, mirror=False, colorbar_owner=None, vlim=None, aspect=None):
    if "local_points" in fdata and "local_" + qty in fdata:
        return _plot_local_field(ax, fdata, qty, cmap, mirror, vlim, aspect)
    geom = str(fdata.get("geometry", "2D-ZY"))
    A = masked_field(fdata, qty)              # NaN inside conductors
    z, y = fdata["z"], fdata["y"]
    ax.clear()
    t = float(fdata.get("t", 0.0))
    if A.ndim == 1:
        ax.plot(z, A, color="#1f77b4", lw=1.5)
        ax.set_xlabel("z (m)")
        ax.set_ylabel(qty)
        ax.grid(alpha=0.3)
        ax.set_title(f"{qty}   t = {t:.4g} s", fontsize=9)
        return None
    mirror = bool(mirror and geom == "2D-RZ")
    if mirror:
        sgn = -1.0 if qty in ("Er", "Etheta", "Br", "Btheta") else 1.0
        A = np.concatenate([sgn * A[:, :0:-1], A], axis=1)
        y = np.concatenate([-y[:0:-1], y])
    sym = qty.startswith(("E", "B", "J")) or (qty.startswith("rho") and np.nanmin(A) < 0 < np.nanmax(A))
    if vlim is None:
        vmax = np.nanmax(np.abs(A)) if sym else np.nanmax(A)
        vmin = -vmax if sym else np.nanmin(A)
    else:
        vmin, vmax = vlim
    if vmax == vmin:
        vmax = vmin + 1e-30
    ratio = (z[-1] - z[0]) / max(y[-1] - y[0], 1e-300)
    asp = aspect or ("equal" if 1 / 3 < ratio < 3 else "auto")
    cm_ = cmap or ("RdBu_r" if sym else "viridis")
    uni = (np.allclose(np.diff(z), (z[-1] - z[0]) / (len(z) - 1), rtol=1e-6)
           and np.allclose(np.diff(y), (y[-1] - y[0]) / max(len(y) - 1, 1), rtol=1e-6))
    if uni:
        im = ax.imshow(np.ma.masked_invalid(A.T), origin="lower", extent=(z[0], z[-1], y[0], y[-1]), aspect=asp,
                       cmap=cm_, vmin=vmin, vmax=vmax, interpolation="bilinear")
    else:                                   # refined (non-uniform) mesh: nodal values, true positions
        im = ax.pcolormesh(z, y, np.ma.masked_invalid(A.T), shading="gouraud", cmap=cm_, vmin=vmin, vmax=vmax)
        ax.set_aspect(asp)
    _conductor_cover(ax, fdata, z, y, mirror)
    ax.set_xlim(z[0], z[-1])
    ax.set_ylim(y[0], y[-1])
    xl, yl = _lbl(geom)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(False)                         # 2D field (contour) map: no axis grid
    ax.set_title(f"{qty}   t = {t:.4g} s", fontsize=9)
    return im


def plot_linecut(ax, fdata, qty, along="z", position=None):
    A = masked_field(fdata, qty)              # gaps inside conductors
    z, y = fdata["z"], fdata["y"]
    ax.clear()
    if A.ndim == 1:
        ax.plot(z, A)
        ax.set_xlabel("z (m)")
    elif along == "z":
        pos = y[len(y) // 2] if position is None else position
        j = int(np.argmin(np.abs(y - pos)))
        ax.plot(z, A[:, j])
        ax.set_xlabel(f"z (m)   at y/r = {y[j]:.4g}")
    else:
        pos = z[len(z) // 2] if position is None else position
        i = int(np.argmin(np.abs(z - pos)))
        ax.plot(y, A[i, :])
        ax.set_xlabel(f"y/r (m)   at z = {z[i]:.4g}")
    ax.set_ylabel(qty)
    ax.grid(alpha=0.3)


PHASE_VARS = ("z", "y", "vz", "vy", "vx", "uz", "uy", "ux", "gamma", "E_kin(eV)")


def phase_var(pdata, sp, name, mass=C.me):
    u0, u1, u2 = pdata[f"{sp}__u0"], pdata[f"{sp}__u1"], pdata[f"{sp}__u2"]
    g = np.sqrt(1 + (u0 ** 2 + u1 ** 2 + u2 ** 2) / C.c ** 2)
    if name == "z":
        return pdata[f"{sp}__z"]
    if name == "y":
        return pdata[f"{sp}__y"]
    if name in ("uz", "uy", "ux"):
        return {"uz": u0, "uy": u1, "ux": u2}[name]
    if name in ("vz", "vy", "vx"):
        return {"vz": u0, "vy": u1, "vx": u2}[name] / g
    if name == "gamma":
        return g
    return (u0 ** 2 + u1 ** 2 + u2 ** 2) / (g + 1) * mass / C.e


def plot_phase(ax, pdata, species=None, xv="z", yv="vz", masses=None, s=1.0):
    ax.clear()
    names = species_names(pdata) if species in (None, "all") else [species]
    for k, sp in enumerate(names):
        if f"{sp}__z" not in pdata or len(pdata[f"{sp}__z"]) == 0:
            continue
        mass = (masses or {}).get(sp, C.me)
        ax.scatter(phase_var(pdata, sp, xv, mass), phase_var(pdata, sp, yv, mass), s=s,
                   color=SPECIES_COLORS[k % len(SPECIES_COLORS)], label=sp, rasterized=True, lw=0)
    ax.set_xlabel(xv)
    ax.set_ylabel(yv)
    ax.grid(alpha=0.3)
    if len(names) > 1:
        ax.legend(markerscale=6, fontsize=8)
    ax.set_title(f"Phase space   t = {float(pdata.get('t', 0)):.4g} s", fontsize=9)


def plot_history(ax, H, cols, keys, logy=False):
    ax.clear()
    if H is None or len(H) == 0:
        return
    t = H[:, cols.index("t")]
    for k in keys:
        if k in cols:
            ax.plot(t, H[:, cols.index(k)], label=k, lw=1.2)
    ax.set_xlabel("t (s)")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    if keys:
        ax.legend(fontsize=8)


def plot_spectrum(ax, H, cols, key):
    ax.clear()
    if H is None or len(H) < 8 or key not in cols:
        return
    t = H[:, cols.index("t")]
    s = H[:, cols.index(key)]
    s = s - s.mean()
    n = len(s) * 8
    F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n))
    f = np.fft.rfftfreq(n, t[1] - t[0])
    ax.semilogy(f / 1e9, F + 1e-300)
    ax.set_xlabel("f (GHz)")
    ax.set_ylabel(f"|FFT {key}|")
    ax.grid(alpha=0.3)
    fpk = f[np.argmax(F[1:]) + 1]
    ax.set_title(f"peak at {fpk / 1e9:.5g} GHz", fontsize=9)
    return fpk


def plot_beam(axs, pdata, species, geometry, nbins=40):
    bm = beam_moments(pdata, species, geometry, nbins)
    for a in axs:
        a.clear()
    if bm is None:
        return
    ok = bm["count"] > 0
    axs[0].plot(bm["z"][ok], bm["rms"][ok] * 1e3, "o-", ms=3)
    axs[0].set_ylabel("rms size (mm)")
    axs[1].plot(bm["z"][ok], bm["emit_n"][ok] * 1e6, "o-", ms=3, color="#d62728")
    axs[1].set_ylabel("norm. rms emittance (mm mrad)")
    axs[1].set_xlabel("z (m)")
    for a in axs:
        a.grid(alpha=0.3)


def _plot_local_field(ax, data, qty, cmap=None, mirror=False, vlim=None, aspect=None):
    """Render the actual P1 field, not a resampled background rectangle."""
    from matplotlib.tri import Triangulation
    p = np.asarray(data["local_points"])
    cells = np.asarray(data["local_triangles"])
    values = np.asarray(data["local_" + qty])
    mask = np.asarray(data.get("local_cell_mat", np.zeros(len(cells)))) == 1
    tri = Triangulation(*p.T, triangles=cells, mask=mask)
    valid = np.unique(cells[~mask])
    used = values[valid]
    sym = qty.startswith(("E", "B", "J")) or (qty.startswith("rho") and used.min() < 0 < used.max())
    if vlim is None:
        hi = float(np.max(abs(used))) if sym else float(used.max())
        lo = -hi if sym else float(used.min())
    else:
        lo, hi = vlim
    if lo == hi: hi = lo + 1e-30
    cm = cmap or ("RdBu_r" if sym else "viridis")
    ax.clear()
    im = ax.tripcolor(tri, values, shading="gouraud", cmap=cm, vmin=lo, vmax=hi, rasterized=True)
    rz = str(data.get("geometry")) == "2D-RZ"
    if mirror and rz:
        other = Triangulation(p[:,0], -p[:,1], cells[:,[0,2,1]], mask=mask)
        sign = -1 if qty in ("Er", "Etheta", "Br", "Btheta") else 1
        ax.tripcolor(other, sign*values, shading="gouraud", cmap=cm, vmin=lo, vmax=hi, rasterized=True)
    ax.set_xlim(p[:,0].min(), p[:,0].max())
    ax.set_ylim(-p[:,1].max() if mirror and rz else p[:,1].min(), p[:,1].max())
    ax.set_aspect(aspect or "auto")
    ax.set_xlabel("z (m)");ax.set_ylabel("r (m)" if rz else "y (m)")
    ax.set_title(f"{qty}   t = {float(data.get('t',0)):.4g} s", fontsize=9)
    ax.grid(False)
    return im
