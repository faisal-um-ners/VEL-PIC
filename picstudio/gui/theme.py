"""Visual theme of the VEL-PIC GUI: Qt style sheets (light / dark), icons and the matching
matplotlib style.  Categorical colours follow a validated colour-blind-safe palette."""
from __future__ import annotations

import matplotlib as mpl
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

THEMES = {
    "light": dict(
        window="#eef1f5", panel="#ffffff", panel2="#f6f8fb", border="#d5dbe3", text="#1d2733",
        text2="#5a6676", accent="#2a6fd6", accent2="#1f5bb5", accent_soft="#dbe8fb",
        header1="#1e2b3c", header2="#2c3e55", header_text="#ffffff", select="#2a6fd6",
        select_text="#ffffff", input="#ffffff", handle="#c9d2dd", plot_bg="#fcfcfb",
        plot_text="#0b0b0b", plot_text2="#52514e", grid="#e3e6ea", series=SERIES_LIGHT),
    "dark": dict(
        window="#15191f", panel="#1d232b", panel2="#222a33", border="#323c48", text="#e6eaf0",
        text2="#9aa6b5", accent="#4a8ff0", accent2="#3a7bd8", accent_soft="#24344a",
        header1="#0d1117", header2="#18202b", header_text="#e6eaf0", select="#3a7bd8",
        select_text="#ffffff", input="#161b22", handle="#3a4552", plot_bg="#1a1a19",
        plot_text="#f0efec", plot_text2="#c3c2b7", grid="#34363a", series=SERIES_DARK),
}


def _icon_path(fname):
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", fname).replace("\\", "/")


def stylesheet(name="light"):
    c = THEMES[name]
    chk = _icon_path("check.svg")
    return f"""
QWidget {{ background: {c['window']}; color: {c['text']}; font-size: 10pt; }}
QMainWindow::separator {{ background: {c['handle']}; width: 6px; height: 6px; }}
QMenuBar {{ background: {c['header1']}; color: {c['header_text']}; padding: 2px; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {c['header2']}; }}
QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['accent']}; color: {c['select_text']}; }}
QToolBar {{ background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {c['header2']}, stop:1 {c['header1']});
           border: none; padding: 4px; spacing: 4px; }}
QToolBar QToolButton {{ background: transparent; color: {c['header_text']}; border: 1px solid transparent;
           border-radius: 5px; padding: 4px 9px; font-weight: 600; }}
QToolBar QToolButton:hover {{ background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); }}
QToolBar QToolButton:pressed {{ background: rgba(255,255,255,0.22); }}
QToolBar QToolButton#runButton {{ background: #1f9d55; border: 1px solid #1a8a4a; }}
QToolBar QToolButton#runButton:hover {{ background: #22b05f; }}
QToolBar QToolButton#stopButton {{ background: #c94141; border: 1px solid #b33838; }}
QToolBar QToolButton#stopButton:hover {{ background: #de4c4c; }}
QToolBar QToolButton#pauseButton {{ background: #d98a00; border: 1px solid #c07a00; }}
QToolBar QToolButton#pauseButton:hover {{ background: #eb9a10; }}
QToolBar::separator {{ background: rgba(255,255,255,0.18); width: 1px; margin: 4px 6px; }}
QToolBar#plotToolbar {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 5px;
           padding: 1px; spacing: 1px; }}
QToolBar#plotToolbar QToolButton {{ color: {c['text']}; font-weight: normal; padding: 2px; border-radius: 4px; }}
QToolBar#plotToolbar QToolButton:hover {{ background: {c['accent_soft']}; border: 1px solid {c['border']}; }}
QToolBar#plotToolbar QLabel {{ background: transparent; color: {c['text2']}; font-size: 8.5pt; }}
QFrame#panel {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
QLabel#panelTitle {{ background: transparent; color: {c['text']}; font-weight: 700; font-size: 10.5pt;
           padding: 6px 4px 2px 6px; }}
QLabel#hint {{ background: transparent; color: {c['text2']}; }}
QTreeWidget, QListWidget, QTableWidget, QPlainTextEdit, QTextBrowser {{ background: {c['panel']};
           border: 1px solid {c['border']}; border-radius: 6px; selection-background-color: {c['select']};
           selection-color: {c['select_text']}; }}
QTreeWidget::item {{ padding: 3px 2px; border-radius: 4px; }}
QTreeWidget::item:hover {{ background: {c['accent_soft']}; }}
QTreeWidget::item:selected {{ background: {c['select']}; color: {c['select_text']}; }}
QHeaderView::section {{ background: {c['panel2']}; border: none; border-bottom: 1px solid {c['border']};
           padding: 4px; font-weight: 600; }}
QScrollArea {{ border: none; background: {c['panel']}; }}
QScrollArea > QWidget > QWidget {{ background: {c['panel']}; }}
QGroupBox {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-radius: 6px; margin-top: 14px;
           padding: 8px 6px 6px 6px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['accent']}; }}
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit#small {{ background: {c['input']}; border: 1px solid {c['border']};
           border-radius: 5px; padding: 3px 6px; min-height: 20px; }}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {c['panel']}; selection-background-color: {c['accent']}; }}
QPushButton {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 5px;
           padding: 5px 12px; }}
QPushButton:hover {{ border: 1px solid {c['accent']}; background: {c['accent_soft']}; }}
QPushButton:pressed {{ background: {c['accent']}; color: {c['select_text']}; }}
QPushButton:disabled {{ color: {c['text2']}; background: {c['panel2']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['select_text']}; border: 1px solid {c['accent2']};
           font-weight: 600; }}
QPushButton#primary:hover {{ background: {c['accent2']}; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QMenu::indicator {{ width: 14px; height: 14px; border: 1px solid {c['handle']};
           border-radius: 3px; background: {c['input']}; }}
QMenu::indicator {{ margin-left: 4px; }}
QCheckBox::indicator:hover {{ border: 1px solid {c['accent']}; }}
QCheckBox::indicator:checked, QMenu::indicator:checked {{ background: {c['accent']}; border: 1px solid {c['accent2']};
           image: url("{chk}"); }}
QCheckBox::indicator:disabled {{ background: {c['panel2']}; }}
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 6px; background: {c['panel']}; top: -1px; }}
QTabBar::tab {{ background: {c['panel2']}; border: 1px solid {c['border']}; border-bottom: none;
           padding: 6px 16px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px;
           color: {c['text2']}; }}
QTabBar::tab:selected {{ background: {c['panel']}; color: {c['accent']}; font-weight: 700;
           border-top: 3px solid {c['accent']}; }}
QSplitter::handle {{ background: {c['window']}; }}
QSplitter::handle:hover {{ background: {c['accent_soft']}; }}
QSplitter::handle:horizontal {{ width: 7px; image: none; }}
QSplitter::handle:vertical {{ height: 7px; }}
QStatusBar {{ background: {c['header1']}; color: {c['header_text']}; }}
QStatusBar QLabel {{ background: transparent; color: {c['header_text']}; }}
QProgressBar {{ background: {c['header2']}; border: 1px solid {c['header2']}; border-radius: 6px;
           text-align: center; color: {c['header_text']}; min-height: 14px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f9d55, stop:1 #3ccf7a);
           border-radius: 5px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['handle']}; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['handle']}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QToolTip {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}
"""


def apply_matplotlib(name="light"):
    c = THEMES[name]
    mpl.rcParams.update({
        "figure.facecolor": c["panel"], "axes.facecolor": c["plot_bg"], "savefig.facecolor": c["plot_bg"],
        "axes.edgecolor": c["grid"], "axes.labelcolor": c["plot_text2"], "axes.titlecolor": c["plot_text"],
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 9, "axes.grid": True,
        "grid.color": c["grid"], "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "xtick.color": c["plot_text2"], "ytick.color": c["plot_text2"], "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": True,
        "legend.facecolor": c["panel"], "legend.edgecolor": c["grid"], "legend.labelcolor": c["plot_text"],
        "lines.linewidth": 1.6, "axes.prop_cycle": mpl.cycler(color=c["series"]),
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
        "image.cmap": "viridis", "text.color": c["plot_text"],
    })


# ---------------------------------------------------------------------------
_ICON_SPECS = {
    # section -> (colour, glyph)
    "root": ("#2a6fd6", "M"), "parameters": ("#6250d6", "P"), "simulation": ("#1f9d55", "S"),
    "domain": ("#0e8a9a", "#"), "mesh_refinements": ("#0b6f7c", "▦"), "boundaries": ("#52617a", "B"), "objects": ("#7b8794", "G"),
    "species": ("#e34948", "e"), "loaders": ("#1baf7a", "L"), "emitters": ("#eb6834", "E"),
    "external_fields": ("#2a78d6", "F"), "sources": ("#9b59b6", "~"), "probes": ("#eda100", "x"), "line_probes": ("#d98200", "↔"),
    "power_lines": ("#17a2b8", "W"), "diagnostics": ("#5a6676", "O"), "parallel": ("#008300", "||"),
    "sweep": ("#d55181", "∑"), "results": ("#2a6fd6", "R"),
}


def section_icon(sec, size=18):
    col, glyph = _ICON_SPECS.get(sec, ("#7b8794", "•"))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(1, 1, size - 2, size - 2), 4, 4)
    p.fillPath(path, QColor(col))
    p.setPen(QPen(QColor("#ffffff")))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(int(size * 0.62))
    p.setFont(f)
    p.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, glyph)
    p.end()
    return QIcon(pm)


def action_icon(kind, size=20):
    """Simple vector icons for the toolbar (drawn, so no image files are needed)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    w = QColor("#ffffff")
    p.setPen(QPen(w, 1.8))
    s = size
    if kind == "run":
        path = QPainterPath()
        path.moveTo(s * 0.3, s * 0.2)
        path.lineTo(s * 0.8, s * 0.5)
        path.lineTo(s * 0.3, s * 0.8)
        path.closeSubpath()
        p.fillPath(path, w)
    elif kind == "stop":
        p.fillRect(QRectF(s * 0.27, s * 0.27, s * 0.46, s * 0.46), w)
    elif kind == "pause":
        p.fillRect(QRectF(s * 0.28, s * 0.24, s * 0.15, s * 0.52), w)
        p.fillRect(QRectF(s * 0.57, s * 0.24, s * 0.15, s * 0.52), w)
    elif kind == "open":
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.3, s * 0.76, s * 0.5), 2, 2)
        p.drawLine(int(s * 0.12), int(s * 0.3), int(s * 0.35), int(s * 0.18))
        p.drawLine(int(s * 0.35), int(s * 0.18), int(s * 0.5), int(s * 0.3))
    elif kind == "save":
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.15, s * 0.64, s * 0.7), 2, 2)
        p.fillRect(QRectF(s * 0.32, s * 0.15, s * 0.36, s * 0.22), w)
        p.drawRect(QRectF(s * 0.3, s * 0.55, s * 0.4, s * 0.3))
    elif kind == "check":
        p.setPen(QPen(w, 2.4))
        p.drawLine(int(s * 0.2), int(s * 0.52), int(s * 0.42), int(s * 0.74))
        p.drawLine(int(s * 0.42), int(s * 0.74), int(s * 0.82), int(s * 0.28))
    elif kind == "new":
        p.drawRoundedRect(QRectF(s * 0.25, s * 0.12, s * 0.5, s * 0.76), 2, 2)
        p.drawLine(int(s * 0.5), int(s * 0.35), int(s * 0.5), int(s * 0.65))
        p.drawLine(int(s * 0.35), int(s * 0.5), int(s * 0.65), int(s * 0.5))
    elif kind == "sweep":
        for k, h in enumerate((0.35, 0.6, 0.45, 0.75)):
            x = s * (0.18 + 0.18 * k)
            p.fillRect(QRectF(x, s * (0.85 - h), s * 0.12, s * h), w)
    elif kind == "results":
        p.drawLine(int(s * 0.15), int(s * 0.85), int(s * 0.85), int(s * 0.85))
        p.drawLine(int(s * 0.15), int(s * 0.85), int(s * 0.15), int(s * 0.15))
        path = QPainterPath()
        path.moveTo(s * 0.18, s * 0.7)
        path.cubicTo(s * 0.4, s * 0.1, s * 0.55, s * 0.9, s * 0.85, s * 0.3)
        p.drawPath(path)
    elif kind == "theme":
        p.drawEllipse(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6))
        path = QPainterPath()
        path.moveTo(s * 0.5, s * 0.2)
        path.arcTo(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 90, 180)
        path.closeSubpath()
        p.fillPath(path, w)
    p.end()
    return QIcon(pm)
