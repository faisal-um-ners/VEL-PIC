"""Matplotlib canvas embedded in Qt."""
from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget  # noqa: E402


class PlotWidget(QWidget):
    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(6, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumSize(120, 80)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setObjectName("plotToolbar")
        from PySide6.QtCore import QSize
        self.toolbar.setIconSize(QSize(16, 16))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(150, 120)
        self.set_grid(nrows, ncols)

    def set_grid(self, nrows, ncols):
        self.fig.clear()
        self.axes = [self.fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
        self._cbars = {}
        return self.axes

    def colorbar(self, im, ax):
        if im is None:
            return
        cb = self._cbars.get(id(ax))
        if cb is not None:
            try:
                cb.remove()
            except Exception:  # noqa: BLE001
                pass
        self._cbars[id(ax)] = self.fig.colorbar(im, ax=ax, shrink=0.85)

    def remove_colorbar(self, ax):
        cb = self._cbars.pop(id(ax), None)
        if cb is not None:
            try:
                cb.remove()
            except Exception:  # noqa: BLE001
                pass

    def draw(self):
        self.canvas.draw_idle()
