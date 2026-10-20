"""Live monitor and post-processing panels."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget)

from ..post import plotting as P
from ..post.reader import Results, species_names
from .canvas import PlotWidget


def _eng(ax, unit_x="", unit_y="", places=1):
    """Tick labels with one SI prefix per axis; the unit goes into the axis label ('z (mm)') and all
    ticks of an axis have the same number of decimals."""
    from ..post.ticks import unit_axes
    unit_axes(ax, unit_x, unit_y)


def _freq_unit(f):
    """Scale and SI unit so that the frequency f [Hz] reads between 1 and 1000 (Hz ... THz)."""
    for s, u in ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(f) >= s:
            return s, u
    return 1.0, "Hz"


def _rzname(v, rz):
    """In RZ the second coordinate is r: y -> r, vy -> vr, uy -> ur, vx/ux -> v_theta/u_theta."""
    if not rz:
        return v
    return {"y": "r", "vy": "vr", "uy": "ur", "vx": "vθ", "ux": "uθ"}.get(v, v)


def field_unit(q):
    q = str(q)
    if q == "phi":
        return "V"
    if q.startswith("rho"):
        return "C/m³"
    if q.startswith("J"):
        return "A/m²"
    if q.startswith("E"):
        return "V/m"
    if q.startswith("B"):
        return "T"
    return ""


PHASE_UNITS = {"z": "m", "y": "m", "vz": "m/s", "vy": "m/s", "vx": "m/s", "uz": "m/s", "uy": "m/s",
               "ux": "m/s", "gamma": "", "E_kin(eV)": "eV"}


class _Envelope:
    """Expand-only axis range built from the min/max of all data seen in this run."""

    def __init__(self):
        self.lo = None
        self.hi = None

    def update(self, a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return
        lo, hi = float(a.min()), float(a.max())
        self.lo = lo if self.lo is None else min(self.lo, lo)
        self.hi = hi if self.hi is None else max(self.hi, hi)

    def limits(self, margin=0.05, symmetric=False):
        if self.lo is None:
            return None
        lo, hi = self.lo, self.hi
        if symmetric:
            m = max(abs(lo), abs(hi)) or 1.0
            return -m * (1 + margin), m * (1 + margin)
        if hi == lo:
            d = abs(hi) * 0.1 or 1.0
            return lo - d, hi + d
        d = (hi - lo) * margin
        return lo - d, hi + d


class LivePanel(QWidget):
    """Live plots refreshed from <output>/live.npz while the solver runs.

    Axis ranges are fixed per run: spatial axes use the domain, other axes and colour
    scales grow from the min/max of the data seen so far (they never jump back), the
    time axis spans the whole run starting at t = 0, and tick labels use engineering
    notation with a constant number of decimals.  The history panel shows any number of
    signals; an inset gives the FFT spectrum of the first selected probe (or signal)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QMenu, QToolButton
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.qty = QComboBox()
        self.phx = QComboBox()
        self.phy = QComboBox()
        self.phx.addItems(P.PHASE_VARS)
        self.phy.addItems(P.PHASE_VARS)
        self.phx.setCurrentText("z")
        self.phy.setCurrentText("vz")
        self.sig_btn = QToolButton()
        self.sig_btn.setText("Signals ▾")
        self.sig_btn.setPopupMode(QToolButton.InstantPopup)
        self.sig_menu = QMenu(self)
        self.sig_btn.setMenu(self.sig_menu)
        self.fft_chk = QCheckBox("FFT inset")
        self.fft_chk.setChecked(True)
        for lab, w in (("Field:", self.qty), ("Phase space:", self.phx), (None, self.phy),
                       ("History:", self.sig_btn), (None, self.fft_chk)):
            if lab:
                bar.addWidget(QLabel(lab))
            bar.addWidget(w)
        bar.addStretch()
        lay.addLayout(bar, 0)
        self.plot = PlotWidget(nrows=2, ncols=2)
        lay.addWidget(self.plot, 1)
        self.data = None
        self.checked = set()
        self.sig_names = []
        self.reset()
        for w in (self.qty, self.phx, self.phy):
            w.currentTextChanged.connect(lambda *_: self.redraw())
        self.fft_chk.toggled.connect(lambda *_: self.redraw())

    def reset(self):
        self.env = {}
        self.data = None
        self.checked = set()
        self.sig_names = []
        self._user_sig = False

    def clear(self, message="Press Run (F5) to start - live plots appear here"):
        """Forget the previous run completely and blank the four plots."""
        self.reset()
        try:
            self.sig_menu.clear()
        except Exception:  # noqa: BLE001
            pass
        self._insets = []
        axs = self.plot.set_grid(2, 2)
        for a, title in zip(axs, ("Field", "Phase space", "Particles", "History")):
            a.set_title(title, fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        axs[0].text(0.5, 0.5, message, transform=axs[0].transAxes, ha="center", va="center", fontsize=8,
                    alpha=0.7, wrap=True)
        self.plot.draw()

    def _e(self, key):
        if key not in self.env:
            self.env[key] = _Envelope()
        return self.env[key]

    # ------------------------------------------------------------------
    def _rebuild_signal_menu(self, cols):
        names = [c for c in cols if c not in ("step", "t")]
        if names == self.sig_names:
            return
        self.sig_names = names
        if not self._user_sig:
            pref = [c for c in names if c.startswith(("I_", "Iemit_"))]
            probes = [c for c in names if c.startswith("probe_")]
            self.checked = set(probes[:1] if probes else pref) or set(names[:1])
        self.sig_menu.clear()
        for c in names:
            act = self.sig_menu.addAction(c)
            act.setCheckable(True)
            act.setChecked(c in self.checked)
            act.toggled.connect(lambda on, c=c: self._toggle_sig(c, on))

    def _toggle_sig(self, c, on):
        self._user_sig = True
        (self.checked.add if on else self.checked.discard)(c)
        self.redraw()

    def update_from(self, d):
        if d is None:
            return
        prog = d.get("progress")
        if self.data is not None and prog is not None and self.data.get("progress") is not None:
            if int(prog[0]) < int(self.data["progress"][0]):      # a new run started
                self.reset()
        self.data = d
        names = P.field_names(d) + [f"line: {str(x).split('|')[0]}" for x in d.get("line_probes", [])]
        cur = self.qty.currentText()
        if [self.qty.itemText(i) for i in range(self.qty.count())] != names:
            self.qty.blockSignals(True)
            self.qty.clear()
            self.qty.addItems(names)
            self.qty.setCurrentText(cur if cur in names else ("phi" if "phi" in names else names[0]))
            self.qty.blockSignals(False)
        self._rebuild_signal_menu([str(c) for c in d.get("history_columns", [])])
        self.redraw()

    def redraw(self):
        d = self.data
        if d is None:
            return
        ax = self.plot.axes
        geom = str(d.get("geometry", ""))
        z, y = np.asarray(d["z"]), np.asarray(d["y"])
        try:
            from ..post.reader import species_names
            field_only = not species_names(d)          # no particles: a pure field solution
            # 1. field map with a fixed (growing-only) colour scale
            q = self.qty.currentText()
            if q.startswith("line: "):
                self._line_plot(ax[0], d, q[6:])
            elif q in d:
                self._field_map(ax[0], d, q, z)
            if field_only:
                # panels 2-4 show the line probes and a second field component instead of particles
                lines = list(self._line_meta(d))
                if lines:
                    self._line_plot(ax[1], d, lines[0])
                else:
                    ax[1].clear()
                    ax[1].set_title("add a line probe to plot a profile here", fontsize=9)
                q2 = next((k for k in ("Ez", "Er", "Ey", "phi") if k in d and k != q), None)
                if q2 is not None:
                    self._field_map(ax[2], d, q2, z, robust=True)
                if len(lines) > 1:
                    self._line_plot(ax[3], d, lines[1])
                else:
                    self._history(ax[3], d)
                self.plot.draw()
                return
            # 2. phase space with stable limits
            xv, yv = self.phx.currentText(), self.phy.currentText()
            P.plot_phase(ax[1], d, "all", xv, yv)
            self._phase_limits(ax[1], d, xv, yv, z, y)
            # 3. particles (2D) or charge density (1D)
            if geom != "1D-Z":
                P.plot_phase(ax[2], d, "all", "z", "y")
                ax[2].set_title("Particles", fontsize=9)
                ax[2].set_xlim(z[0], z[-1])
                ax[2].set_ylim(y[0], y[-1])
                ax[2].set_xlabel("z")
                ax[2].set_ylabel("r" if geom == "2D-RZ" else "y")
                _eng(ax[2], "m", "m")
            elif "rho" in d:
                e = self._e("rho1d")
                e.update(d["rho"])
                P.plot_field(ax[2], d, "rho")
                ax[2].set_xlim(z[0], z[-1])
                ax[2].set_ylim(*e.limits(0.05))
                _eng(ax[2], "m", "C/m³")
            # 4. history with fixed time axis [0, t_end] and optional FFT inset
            self._history(ax[3], d)
        except Exception as exc:  # noqa: BLE001
            ax[3].set_title(f"plot error: {exc}", fontsize=8)
        self.plot.draw()

    def _field_map(self, a, d, q, z, robust=False):
        """2D field map (or 1D profile) of quantity q with a growing-only colour scale.

        robust: colour scale from the 99.5th percentile of |q| (field-only runs, where the singular
        field at conductor corners would otherwise wash out the map)."""
        A = P.masked_field(d, q)          # conductor interiors excluded from the colour scale
        e = self._e("field:" + q)
        e.update(A)
        sym = q.startswith(("E", "B", "J")) or (q.startswith("rho") and e.lo < 0 < e.hi)
        vl = e.limits(0.0, symmetric=sym)
        if robust and A.ndim == 2 and np.isfinite(A).any():
            # area-weighted percentile, so that the many small cells of a refined mesh do not dominate
            zz, yy = np.asarray(d["z"], float), np.asarray(d["y"], float)
            wz = np.gradient(zz) if len(zz) > 1 else np.ones_like(zz)
            wy = np.gradient(yy) if len(yy) > 1 else np.ones_like(yy)
            W = np.outer(wz, wy)
            ok = np.isfinite(A)
            v, w = np.abs(A[ok]), W[ok]
            k = np.argsort(v)
            cw = np.cumsum(w[k])
            m = float(v[k][min(np.searchsorted(cw, 0.99 * cw[-1]), len(v) - 1)]) or 1.0
            vl = (-m, m) if sym else (float(np.nanmin(A)), float(np.nanmax(A)))
        im = P.plot_field(a, d, q, vlim=vl, aspect="auto")
        if A.ndim == 1:
            a.set_xlim(z[0], z[-1])
            a.set_ylim(*e.limits(0.05))
            _eng(a, "m", "")
        else:
            _eng(a, "m", "m")
            self._overlay_lines(a, d)
        self.plot.colorbar(im, a)
        cb = self.plot._cbars.get(id(a))
        if cb is not None:
            from ..post.ticks import unit_colorbar
            unit_colorbar(cb, field_unit(q), q)

    @staticmethod
    def _line_meta(d):
        out = {}
        for x in d.get("line_probes", []):
            f = str(x).split("|")
            if len(f) >= 7:
                out[f[0]] = f
        return out

    def _overlay_lines(self, a, d):
        """Dashed line + name of every line probe on the 2D field map."""
        for name, f in self._line_meta(d).items():
            try:
                z0, y0, z1, y1 = (float(v) for v in f[3:7])
            except ValueError:
                continue
            a.plot([z0, z1], [y0, y1], color="white", lw=1.4, ls=(0, (4, 2)), alpha=0.9)
            a.annotate(name, (z1, y1), xytext=(3, 3), textcoords="offset points", fontsize=7, color="white")

    def _line_plot(self, a, d, name):
        """Profile of a line probe with its minimum / maximum marked and labelled."""
        from matplotlib.ticker import EngFormatter
        self.plot.remove_colorbar(a)
        a.clear()
        f = self._line_meta(d).get(name)
        key = f"line_{name}"
        if f is None or key not in d:
            a.set_title(f"line probe {name}: no data yet", fontsize=9)
            return
        qty, along = f[1], f[2]
        ext = f[7] if len(f) > 7 else "min"
        s, v = np.asarray(d[key + "_s"], float), np.asarray(d[key], float)
        e = self._e("line:" + name)
        e.update(v)
        geom = str(d.get("geometry", ""))
        alab = along if along == "z" else ("r" if geom == "2D-RZ" else "y")
        a.plot(s, v, lw=1.5, color="#1f77b4", label=f"{qty}({alab})")
        fv, fs = EngFormatter(unit=field_unit(qty), places=3), EngFormatter(unit="m", places=3)
        ok = np.isfinite(v)                               # NaN = inside a conductor (no field)
        if not ok.any():
            a.set_title(f"line probe {name}: the line lies inside a conductor", fontsize=9)
            return
        w_ = np.trapezoid(ok.astype(float), s) if len(s) > 1 else 0.0
        avg = float(np.trapezoid(np.where(ok, v, 0.0), s) / w_) if w_ != 0 else float(np.nanmean(v))
        a.axhline(avg, color="#7f7f7f", ls="--", lw=1, label=f"average {fv(avg)}")
        if "min" in ext:
            k = int(np.nanargmin(v))
            a.plot(s[k], v[k], "v", color="#d62728", ms=8, label=f"min {fv(v[k])} at {alab} = {fs(s[k])}")
        if "max" in ext:
            k = int(np.nanargmax(v))
            a.plot(s[k], v[k], "^", color="#2ca02c", ms=8, label=f"max {fv(v[k])} at {alab} = {fs(s[k])}")
        a.set_xlim(s[0], s[-1])
        lim = e.limits(0.08)
        if lim:
            a.set_ylim(*lim)
        a.set_xlabel(alab)
        a.set_ylabel(qty)
        a.set_title(f"{qty} along line '{name}'   t = {float(d.get('t', 0.0)):.4g} s", fontsize=9)
        a.legend(fontsize=7, loc="best")
        _eng(a, "m", field_unit(qty))

    def _phase_limits(self, a, d, xv, yv, z, y):
        from ..post.reader import species_names
        for axis, var in (("x", xv), ("y", yv)):
            if var == "z":
                lim = (z[0], z[-1])
            elif var == "y" and len(y) > 1:
                lim = (y[0], y[-1])
            else:
                e = self._e("phase:" + var)
                for sp in species_names(d):
                    if f"{sp}__z" in d and len(d[f"{sp}__z"]):
                        e.update(P.phase_var(d, sp, var))
                lim = e.limits(0.05)
            if lim is not None:
                (a.set_xlim if axis == "x" else a.set_ylim)(*lim)
        from ..post.ticks import unit_axes
        rz = str(self.data.get("geometry", "")) == "2D-RZ" if self.data is not None else False
        unit_axes(a, PHASE_UNITS.get(xv, ""), PHASE_UNITS.get(yv, ""), xlabel=_rzname(xv, rz),
                  ylabel=_rzname(yv, rz))

    def _history(self, a, d):
        a.clear()
        for extra in list(getattr(self, "_insets", [])):
            try:
                extra.remove()
            except Exception:  # noqa: BLE001
                pass
        self._insets = []
        H = d.get("history")
        cols = [str(c) for c in d.get("history_columns", [])]
        if H is None or not np.size(H) or not self.checked:
            return
        H = np.asarray(H)
        t = H[:, cols.index("t")]
        keys = [c for c in self.sig_names if c in self.checked and c in cols]
        e = self._e("hist:" + ",".join(sorted(keys)))
        t_end = float(d.get("t_end", t[-1] if len(t) else 1.0))
        late = t > 0.02 * t_end           # ignore the start-up transient for the axis range
        for k in keys:
            v = H[:, cols.index(k)]
            e.update(v[late] if late.any() else v)
            a.plot(t, v, lw=1.2, label=k)
        a.set_xlim(0.0, t_end if t_end > 0 else (t[-1] or 1.0))
        lim = e.limits(0.08)
        if lim:
            a.set_ylim(*lim)
        a.set_xlabel("t")
        units = [str(u) for u in d.get("history_units", [])]
        ku = {units[cols.index(k)] for k in keys} if len(units) == len(cols) else set()
        unit = ku.pop() if len(ku) == 1 else ""
        if keys and all(k.startswith(("I_", "Iemit_")) for k in keys):
            a.set_ylabel("current", fontsize=8)
        elif keys and all(k.startswith("probe_") for k in keys):
            a.set_ylabel("probe", fontsize=8)
        _eng(a, "s", unit)
        if keys:      # legend above the plot (in place of the title) so it never hides data or the inset
            a.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(keys), 3),
                     frameon=False, handlelength=1.5, columnspacing=1.0)
        else:
            a.set_title("History", fontsize=9)
        if self.fft_chk.isChecked() and len(t) > 16:
            probes = [k for k in keys if k.startswith("probe_")]
            cand = [k for k in (probes or keys) if np.ptp(H[len(t) // 2:, cols.index(k)]) > 0]
            if not cand:
                return
            k = cand[0]
            # spectrum from the last raw (undecimated) samples, after the start-up
            T = np.asarray(d.get("history_tail", H))
            if T.ndim != 2 or len(T) < 16:
                T = H
            tt = T[:, cols.index("t")]
            sT = T[:, cols.index(k)]
            half = len(sT) // 4 if len(sT) >= 4096 else len(sT) // 2
            ss = sT[half:] - sT[half:].mean()
            dts = np.diff(tt[half:])
            if len(ss) > 8 and np.any(ss) and len(dts) and dts.min() > 0:
                n = len(ss) * 8
                F = np.abs(np.fft.rfft(ss * np.hanning(len(ss)), n))
                f = np.fft.rfftfreq(n, float(np.median(dts)))
                # put the inset in the half of the plot where the late-time data are not
                ylo, yhi = a.get_ylim()
                lv = np.concatenate([H[len(H) // 2:, cols.index(kk)] for kk in keys])
                upper = np.mean(lv > 0.5 * (ylo + yhi)) > 0.5
                ins = a.inset_axes([0.60, 0.08 if upper else 0.55, 0.37, 0.36])
                self._insets.append(ins)
                fpk = f[1:][np.argmax(F[1:])]
                fs, fu = _freq_unit(fpk)
                ins.semilogy(f[1:] / fs, F[1:] + 1e-300, lw=0.9, color="#6250d6")
                ins.axvline(fpk / fs, color="#e34948", lw=0.8, ls="--")
                ins.set_title(f"Peak: {fpk / fs:.2f} {fu}", fontsize=7)
                ins.tick_params(labelsize=6)
                from matplotlib.ticker import MaxNLocator
                ins.xaxis.set_major_locator(MaxNLocator(4))
                ins.set_xlim(0, min(f[-1], 6 * fpk if fpk > 0 else f[-1]) / fs)
                ins.set_yticklabels([])


class ResultsPanel(QWidget):
    PLOTS = ("Field map", "Line cut z", "Line cut y/r", "Phase space", "Particles (z-y)", "History",
             "Spectrum (FFT)", "Beam envelope")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        lay = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split)
        ctl = QWidget()
        v = QVBoxLayout(ctl)
        self.dir_label = QLabel("No results loaded")
        self.dir_label.setWordWrap(True)
        b = QPushButton("Open results folder...")
        b.clicked.connect(self.choose_dir)
        br = QPushButton("Refresh")
        br.clicked.connect(self.refresh)
        v.addWidget(self.dir_label)
        v.addWidget(b)
        v.addWidget(br)
        self.kind = QComboBox()
        self.kind.addItems(self.PLOTS)
        v.addWidget(QLabel("Plot type"))
        v.addWidget(self.kind)
        self.qty = QComboBox()
        v.addWidget(QLabel("Quantity / species"))
        v.addWidget(self.qty)
        self.sp = QComboBox()
        v.addWidget(self.sp)
        self.phx = QComboBox()
        self.phy = QComboBox()
        self.phx.addItems(P.PHASE_VARS)
        self.phy.addItems(P.PHASE_VARS)
        self.phy.setCurrentText("vz")
        v.addWidget(QLabel("Phase-space axes (x, y)"))
        v.addWidget(self.phx)
        v.addWidget(self.phy)
        self.mirror = QCheckBox("Mirror RZ about the axis")
        v.addWidget(self.mirror)
        self.logy = QCheckBox("Log scale (history)")
        v.addWidget(self.logy)
        v.addWidget(QLabel("History signals"))
        self.hlist = QListWidget()
        self.hlist.setMaximumHeight(180)
        v.addWidget(self.hlist)
        v.addWidget(QLabel("Snapshot"))
        self.slider = QSlider(Qt.Horizontal)
        v.addWidget(self.slider)
        self.snap_label = QLabel("")
        v.addWidget(self.snap_label)
        be = QPushButton("Export figure (PNG/PDF)...")
        be.clicked.connect(self.export_fig)
        ba = QPushButton("Export animation (GIF/MP4)...")
        ba.clicked.connect(self.export_anim)
        bc = QPushButton("Export data (CSV)...")
        bc.clicked.connect(self.export_csv)
        v.addWidget(be)
        v.addWidget(ba)
        v.addWidget(bc)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        v.addStretch()
        from PySide6.QtWidgets import QScrollArea
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(ctl)
        sc.setMinimumWidth(200)
        split.addWidget(sc)
        self.plot = PlotWidget()
        split.addWidget(self.plot)
        split.setSizes([260, 900])
        for w in (self.kind, self.qty, self.sp, self.phx, self.phy):
            w.currentTextChanged.connect(lambda *_: self.redraw())
        self.mirror.toggled.connect(lambda *_: self.redraw())
        self.logy.toggled.connect(lambda *_: self.redraw())
        self.slider.valueChanged.connect(lambda *_: self.redraw())
        self.hlist.itemChanged.connect(lambda *_: self.redraw())

    # ------------------------------------------------------------------
    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder")
        if d:
            self.load(d)

    def clear(self):
        self.res = None
        self.dir_label.setText("(no results loaded)")
        self.plot.set_grid(1, 1)
        self.plot.draw()

    def load(self, d):
        if not os.path.isdir(d):
            return
        self.res = Results(d)
        self.dir_label.setText(d)
        self.refresh()

    def refresh(self):
        if self.res is None:
            return
        self.res.refresh()
        f = self.res.field(-1)
        if f is not None:
            names = P.field_names(f)
            cur = self.qty.currentText()
            self.qty.blockSignals(True)
            self.qty.clear()
            self.qty.addItems(names)
            if cur in names:
                self.qty.setCurrentText(cur)
            self.qty.blockSignals(False)
        p = self.res.particles(-1)
        if p is not None:
            cur = self.sp.currentText()
            self.sp.blockSignals(True)
            self.sp.clear()
            self.sp.addItems(["all"] + species_names(p))
            if cur:
                self.sp.setCurrentText(cur)
            self.sp.blockSignals(False)
        H, cols = self.res.history()
        checked = {self.hlist.item(i).text() for i in range(self.hlist.count())
                   if self.hlist.item(i).checkState() == Qt.Checked}
        self.hlist.blockSignals(True)
        self.hlist.clear()
        for c in cols:
            if c in ("step", "t"):
                continue
            it = QListWidgetItem(c)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            on = c in checked or (not checked and c.startswith("I_"))
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.hlist.addItem(it)
        self.hlist.blockSignals(False)
        n = max(len(self.res.field_files), len(self.res.particle_files))
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setValue(max(0, n - 1))
        info = self.res.info
        self.info.setText(f"{info.get('name', '')}<br>{info.get('geometry', '')} / {info.get('solver', '')}<br>"
                          f"dt = {info.get('dt', 0):.3g} s, steps = {info.get('n_steps', '')}<br>"
                          f"{len(self.res.field_files)} field / {len(self.res.particle_files)} particle snapshots")
        self.redraw()

    def _checked(self):
        return [self.hlist.item(i).text() for i in range(self.hlist.count())
                if self.hlist.item(i).checkState() == Qt.Checked]

    def redraw(self):
        if self.res is None:
            return
        kind = self.kind.currentText()
        if kind == "Beam envelope":
            axs = self.plot.set_grid(2, 1)
        else:
            axs = self.plot.set_grid(1, 1)
        ax = axs[0]
        i = self.slider.value()
        try:
            if kind in ("Field map", "Line cut z", "Line cut y/r"):
                files = self.res.field_files
                if not files:
                    return
                f = self.res.field(min(i, len(files) - 1))
                q = self.qty.currentText() or "phi"
                if q not in f:
                    return
                self.snap_label.setText(f"t = {float(f['t']):.4g} s  (step {int(f['step'])})")
                if kind == "Field map":
                    im = P.plot_field(ax, f, q, mirror=self.mirror.isChecked())
                    self.plot.colorbar(im, ax)
                else:
                    P.plot_linecut(ax, f, q, "z" if kind == "Line cut z" else "y")
            elif kind in ("Phase space", "Particles (z-y)", "Beam envelope"):
                files = self.res.particle_files
                if not files:
                    return
                p = self.res.particles(min(i, len(files) - 1))
                self.snap_label.setText(f"t = {float(p['t']):.4g} s")
                masses = {s["name"]: s["m"] for s in self.res.info.get("species", [])}
                sp = self.sp.currentText() or "all"
                if kind == "Phase space":
                    P.plot_phase(ax, p, sp, self.phx.currentText(), self.phy.currentText(), masses)
                elif kind == "Particles (z-y)":
                    P.plot_phase(ax, p, sp, "z", "y", masses)
                else:
                    sp = sp if sp != "all" else (species_names(p) or [""])[0]
                    P.plot_beam(axs, p, sp, self.res.geometry)
            elif kind == "History":
                H, cols = self.res.history()
                P.plot_history(ax, H, cols, self._checked(), self.logy.isChecked())
            elif kind == "Spectrum (FFT)":
                H, cols = self.res.history()
                ch = self._checked()
                if ch:
                    P.plot_spectrum(ax, H, cols, ch[0])
            # consistent engineering tick labels
            if kind == "Field map":
                _eng(ax, "m", "m" if np.ndim(f[q]) == 2 else "", 1)
            elif kind in ("Line cut z", "Line cut y/r"):
                _eng(ax, "m", "")
            elif kind in ("Phase space", "Particles (z-y)"):
                xv, yv = (self.phx.currentText(), self.phy.currentText()) if kind == "Phase space" else ("z", "y")
                from ..post.ticks import unit_axes
                rz = str(self.res.info.get("geometry", "")) == "2D-RZ"
                unit_axes(ax, PHASE_UNITS.get(xv, ""), PHASE_UNITS.get(yv, ""), xlabel=_rzname(xv, rz),
                          ylabel=_rzname(yv, rz))
            elif kind == "History":
                _eng(ax, "s", "")
        except Exception as exc:  # noqa: BLE001
            ax.set_title(f"plot error: {exc}", fontsize=8)
        self.plot.draw()

    def export_fig(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Export figure", "figure.png", "Images (*.png *.pdf *.svg)")
        if fn:
            self.plot.fig.savefig(fn, dpi=200)

    def export_anim(self):
        if self.res is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export animation", "movie.gif", "Animation (*.gif *.mp4)")
        if not fn:
            return
        from matplotlib.animation import FuncAnimation, PillowWriter
        n = self.slider.maximum() + 1

        def frame(k):
            self.slider.setValue(k)
            return []
        anim = FuncAnimation(self.plot.fig, frame, frames=n, blit=False)
        if fn.endswith(".gif"):
            anim.save(fn, writer=PillowWriter(fps=8))
        else:
            anim.save(fn, fps=8)

    def export_csv(self):
        if self.res is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export data", "data.csv", "CSV (*.csv)")
        if not fn:
            return
        kind = self.kind.currentText()
        if kind == "History":
            H, cols = self.res.history()
            np.savetxt(fn, H, delimiter=",", header=",".join(cols), comments="")
        else:
            f = self.res.field(self.slider.value())
            q = self.qty.currentText()
            np.savetxt(fn, np.atleast_2d(f[q]), delimiter=",")
