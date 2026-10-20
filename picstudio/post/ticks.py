"""Axis tick formatting with ONE SI prefix per axis (e.g. 0.0 0.2 ... 1.0 with label 'z (mm)').

Matplotlib's EngFormatter picks a prefix per tick ("0.0 m, 200.0 µm, 1.0 mm"), which gives long,
overlapping labels.  Here the prefix is chosen from the largest |value| in view, the unit goes into
the axis label, and the number of decimals is set from the tick spacing, so all ticks of an axis
have the same number of decimals.
"""
from __future__ import annotations

import math
import re

from matplotlib.ticker import Formatter, MaxNLocator

PREFIX = {-18: "a", -15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


def _strip_unit(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", label or "").strip()


class UnitFormatter(Formatter):
    def __init__(self, unit="", base=None, label=True):
        self.unit = unit
        self.base = base
        self.label = label
        self.exp = 0
        self.places = 1

    def _update(self):
        try:
            lo, hi = self.axis.get_view_interval()
        except Exception:  # noqa: BLE001
            return
        big = max(abs(lo), abs(hi))
        e = 0
        if big > 0 and math.isfinite(big):
            e = int(math.floor(math.log10(big) / 3.0) * 3)
            # keep values like 0.5 mm as "500 µm"? no: prefer 0.1 <= scaled max < 1000
            e = max(min(e, 12), -18)
        self.exp = e
        locs = [x for x in self.locs if lo - 1e-12 * abs(hi - lo) <= x <= hi + 1e-12 * abs(hi - lo)] or self.locs
        if len(locs) > 1:
            step = abs(locs[1] - locs[0]) / 10.0 ** e
            p = 0          # fewest decimals that represent the tick step exactly (0.25 -> 2, 0.2 -> 1)
            while p < 4 and abs(step * 10 ** p - round(step * 10 ** p)) > 1e-6 * max(1.0, step * 10 ** p):
                p += 1
            self.places = p
        else:
            self.places = 1
        if not self.label:
            return
        base = self.base
        if base is None:
            base = _strip_unit(self.axis.get_label_text())
            self.base = base
        pre = PREFIX.get(e, f"1e{e} ")
        if self.unit:
            u = pre + self.unit
        elif e:
            u = f"×10^{e}" if e not in (3, 6, 9) else {3: "×10³", 6: "×10⁶", 9: "×10⁹"}[e]
        else:
            u = ""
        txt = f"{base} ({u})" if (base and u) else (base or u)
        if self.axis.get_label_text() != txt:
            self.axis.set_label_text(txt)

    def set_locs(self, locs):
        self.locs = list(locs)
        self._update()

    def __call__(self, x, pos=None):
        v = x / 10.0 ** self.exp
        if abs(v) < 0.5 * 10 ** (-self.places):
            v = 0.0
        return f"{v:.{self.places}f}"


def unit_axes(ax, unit_x="", unit_y="", xlabel=None, ylabel=None, nx=5, ny=6):
    """Apply one-prefix-per-axis formatting to both axes of ``ax``."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(UnitFormatter(unit_x))
    ax.yaxis.set_major_formatter(UnitFormatter(unit_y))
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))


def unit_colorbar(cb, unit, label=""):
    cb.ax.yaxis.set_major_formatter(UnitFormatter(unit, base=label))
