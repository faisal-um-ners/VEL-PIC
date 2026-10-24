"""Main window: model builder tree, settings, graphics, run control, results."""
from __future__ import annotations

import copy
import os
import shutil
import sys

import numpy as np

from PySide6.QtCore import QProcess, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel,
                               QSizePolicy,
                               QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .. import AFFILIATION, APP_NAME, APP_SUBTITLE, APP_TAGLINE, AUTHOR, CITATION, COPYRIGHT, DOI, LICENSE, URL, __version__
from ..model import model as M
from ..model import schema as S
from ..post import plotting as P
from .canvas import PlotWidget
from .forms import BoundaryPanel, FieldForm, ParameterTable
from .results import LivePanel, ResultsPanel

TREE_ORDER = ["parameters", "simulation", "domain", "mesh_refinements", "boundaries", "objects", "species", "loaders",
              "emitters", "external_fields", "sources", "probes", "line_probes", "power_lines", "diagnostics",
              "parallel", "sweep", "results"]
ADD_LABELS = {"mesh_refinements": "refinement region", "objects": "object", "species": "species", "loaders": "loader", "emitters": "emitter",
              "external_fields": "applied field", "sources": "source", "probes": "point probe", "line_probes": "line probe",
              "power_lines": "power monitor"}


def examples_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "..", "examples"), os.path.join(here, "..", "examples")):
        if os.path.isdir(c):
            return os.path.abspath(c)
    return None


class MainWindow(QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1500, 920)
        self.settings = QSettings("PICStudio", "PICStudio")
        self.model = M.new_model("2D-ZY")
        self.path = None
        self.dirty = False
        self.proc = None
        self.run_out = None
        self.selected = None
        self.theme = self.settings.value("theme", "light") or "light"
        from .theme import apply_matplotlib, stylesheet
        QApplication.instance().setStyleSheet(stylesheet(self.theme))
        apply_matplotlib(self.theme)
        self._build_ui()
        self._build_menus()
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._poll_live)
        self.geo_timer = QTimer(self)
        self.geo_timer.setSingleShot(True)
        self.geo_timer.timeout.connect(self.draw_geometry)
        if path:
            self.open_file(path)
        else:
            self.refresh_all()

    # ======================================================================
    # UI construction
    # ======================================================================
    def _panel(self, title):
        """Rounded card with a title label; returns (frame, layout)."""
        fr = QFrame()
        fr.setObjectName("panel")
        lay = QVBoxLayout(fr)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(4)
        lab = QLabel(title)
        lab.setObjectName("panelTitle")
        lay.addWidget(lab)
        return fr, lay, lab

    def _build_ui(self):
        # vertical splitter: [work area] / [messages] - both directions are draggable
        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.setHandleWidth(7)
        self.setCentralWidget(self.vsplit)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(7)
        self.hsplit = split

        left, lv, _ = self._panel("Model Builder")
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setIndentation(16)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        lv.addWidget(self.tree)
        hb = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_add.setObjectName("primary")
        self.btn_del = QPushButton("Delete")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_add.clicked.connect(self.add_item)
        self.btn_del.clicked.connect(self.delete_item)
        self.btn_dup.clicked.connect(self.duplicate_item)
        for bt in (self.btn_add, self.btn_del, self.btn_dup):
            hb.addWidget(bt)
        lv.addLayout(hb)
        left.setMinimumWidth(180)
        split.addWidget(left)

        mid, mv, self.settings_title = self._panel("Settings")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        mv.addWidget(self.scroll)
        mid.setMinimumWidth(220)
        split.addWidget(mid)

        right, rv, _ = self._panel("Graphics")
        self.tabs = QTabWidget()
        geo = QWidget()
        gv = QVBoxLayout(geo)
        gh = QHBoxLayout()
        self.chk_mesh = QCheckBox("Show mesh")
        self.chk_paint = QCheckBox("Show painted cells (staircase)")
        self.chk_mesh.toggled.connect(self.draw_geometry)
        self.chk_paint.toggled.connect(self.draw_geometry)
        gh.addWidget(self.chk_mesh)
        gh.addWidget(self.chk_paint)
        gh.addStretch()
        self.geo_info = QLabel("")
        self.geo_info.setObjectName("hint")
        gh.addWidget(self.geo_info)
        gv.addLayout(gh, 0)
        self.geo_plot = PlotWidget()
        gv.addWidget(self.geo_plot, 1)
        self.tabs.addTab(geo, "Geometry")
        self.live = LivePanel()
        self.tabs.addTab(self.live, "Live monitor")
        self.results = ResultsPanel()
        self.results.can_open = self._can_open_files
        self.tabs.addTab(self.results, "Results")
        rv.addWidget(self.tabs)
        right.setMinimumWidth(320)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, 1)
        split.setSizes([270, 400, 830])
        self.vsplit.addWidget(split)

        bottom, bv, _ = self._panel("Messages")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Menlo" if sys.platform == "darwin" else "Monospace", 9))
        bv.addWidget(self.log)
        bottom.setMinimumHeight(70)
        self.vsplit.addWidget(bottom)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 0)
        self.vsplit.setSizes([760, 150])
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.status = QLabel("Ready")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def set_theme(self, name):
        from .theme import apply_matplotlib, stylesheet
        self.theme = name
        self.settings.setValue("theme", name)
        QApplication.instance().setStyleSheet(stylesheet(name))
        apply_matplotlib(name)
        import matplotlib as mpl
        for pw, grid in ((self.geo_plot, (1, 1)), (self.live.plot, (2, 2)), (self.results.plot, (1, 1))):
            pw.fig.set_facecolor(mpl.rcParams["figure.facecolor"])
            pw.set_grid(*grid)
        self.draw_geometry()
        self.live.redraw()
        self.results.redraw()

    def _act(self, menu, text, slot, shortcut=None, tb=None, icon=None, name=None):
        a = QAction(text, self)
        if text == "About":
            # Qt/macOS otherwise relocates this item to the application menu.
            # Keep Help -> About visible on all supported platforms.
            a.setMenuRole(QAction.MenuRole.NoRole)
        if icon is not None:
            from .theme import action_icon
            a.setIcon(action_icon(icon))
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        if tb is not None:
            tb.addAction(a)
            if name:
                tb.widgetForAction(a).setObjectName(name)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        from PySide6.QtGui import QIcon
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "app_logo.svg")
        self.setWindowIcon(QIcon(logo_path))
        brand = QLabel()
        brand.setPixmap(QIcon(logo_path).pixmap(26, 26))
        brand.setStyleSheet("background: transparent; padding: 0 4px 0 6px;")
        tb.addWidget(brand)
        bname = QLabel(APP_NAME)
        bname.setStyleSheet("background: transparent; color: white; font-weight: 700; font-size: 12pt; "
                            "padding-right: 12px;")
        tb.addWidget(bname)
        tb.addSeparator()
        self._file_open_actions = []
        f = mb.addMenu("&File")
        newm = f.addMenu("New model")
        self._file_open_actions.append(newm.menuAction())
        for g in S.GEOMETRIES:
            for s in S.SOLVERS:
                self._act(newm, f"{g}  {s}", lambda _=False, g=g, s=s: self.new_model(g, s))
        self._file_open_actions.append(
            self._act(f, "Open...", self.open_dialog, "Ctrl+O", tb, icon="open"))
        self._act(f, "Save", self.save, "Ctrl+S", tb, icon="save")
        self._act(f, "Save as...", self.save_as, "Ctrl+Shift+S")
        exm = f.addMenu("Examples")
        self._file_open_actions.append(exm.menuAction())
        ed = examples_dir()
        if ed:
            for fn in sorted(os.listdir(ed)):
                if fn.endswith(".pic.json"):
                    self._act(exm, fn.replace(".pic.json", "").replace("_", " "),
                              lambda _=False, p=os.path.join(ed, fn): self.open_example(p))
        f.addSeparator()
        self._act(f, "Quit", self.close, "Ctrl+Q")
        mm = mb.addMenu("&Model")
        tb.addSeparator()
        self._act(mm, "Check model", self.check_model, "F7", tb, icon="check")
        self._act(mm, "Show mesh", self.show_mesh, "F6")
        self._act(mm, "Mesh / time-step info", self.mesh_info)
        self._act(mm, "Calibrate loss Q (cold test)...", self.calibrate_q)
        rm = mb.addMenu("&Run")
        tb.addSeparator()
        self.act_run = self._act(rm, "Run", self.run, "F5", tb, icon="run", name="runButton")
        self.act_pause = self._act(rm, "Pause", self.toggle_pause, "F8", tb, icon="pause", name="pauseButton")
        self.act_stop = self._act(rm, "Stop", self.stop, "Shift+F5", tb, icon="stop", name="stopButton")
        self._act(rm, "Run with MPI...", self.run_mpi)
        self._act(rm, "Run parametric sweep", self.run_sweep, tb=tb, icon="sweep")
        res = mb.addMenu("Re&sults")
        self._file_open_actions.append(
            self._act(res, "Open results folder...", self.results.choose_dir))
        self._file_open_actions.append(
            self._act(res, "Load results of this model", self.load_results, tb=tb, icon="results"))
        vm = mb.addMenu("&View")
        self._act(vm, "Light theme", lambda: self.set_theme("light"))
        self._act(vm, "Dark theme", lambda: self.set_theme("dark"))
        sp = QWidget()
        sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setStyleSheet("background: transparent;")
        tb.addWidget(sp)
        self._act(vm, "Toggle theme", lambda: self.set_theme("dark" if self.theme == "light" else "light"),
                  tb=tb, icon="theme")
        h = mb.addMenu("&Help")
        self._act(h, "User guide", lambda: self.show_guide("USER_GUIDE.md"))
        self._act(h, "Numerical methods", lambda: self.show_guide("THEORY.md"))
        self._act(h, "Validation report", self.show_validation)
        self._act(h, "How to cite VEXPACE", self.cite)
        self._act(h, "About", self.about)

    # ======================================================================
    # model tree
    # ======================================================================
    def refresh_all(self):
        self.model = M.complete(self.model)
        self.populate_tree()
        self.update_title()
        self.draw_geometry()

    def populate_tree(self, select=None):
        self.tree.blockSignals(True)
        self.tree.clear()
        from .theme import section_icon
        root = QTreeWidgetItem([f"{self.model.get('name', 'Untitled')}"])
        root.setIcon(0, section_icon("root"))
        root.setData(0, Qt.UserRole, ("root", None))
        f = root.font(0)
        f.setBold(True)
        root.setFont(0, f)
        self.tree.addTopLevelItem(root)
        target = None
        for sec in TREE_ORDER:
            title = S.SECTION_TITLES.get(sec, "Results") if sec != "results" else "Results"
            it = QTreeWidgetItem([title])
            it.setIcon(0, section_icon(sec))
            it.setData(0, Qt.UserRole, (sec, None))
            root.addChild(it)
            if select == (sec, None):
                target = it
            if sec in S.LIST_SECTIONS:
                for i, obj in enumerate(self.model[sec]):
                    label = obj.get("name", f"{sec}{i}")
                    extra = obj.get("shape") or obj.get("model") or obj.get("type") or obj.get("preset") or ""
                    if sec == "objects":
                        extra = f"{obj.get('material')} {obj.get('shape')}"
                    ch = QTreeWidgetItem([f"{label}  ({extra})" if extra else label])
                    ch.setData(0, Qt.UserRole, (sec, i))
                    it.addChild(ch)
                    if select == (sec, i):
                        target = ch
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self.tree.setCurrentItem(target or root)
        self._on_select()

    def _current(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it else ("root", None)

    def _on_select(self):
        sec, idx = self._current()
        self.selected = (sec, idx)
        self.show_settings(sec, idx)
        self.btn_add.setEnabled(sec in S.LIST_SECTIONS)
        self.btn_del.setEnabled(sec in S.LIST_SECTIONS and idx is not None)
        self.btn_dup.setEnabled(sec in S.LIST_SECTIONS and idx is not None)
        self.geo_timer.start(150)

    def _tree_menu(self, pos):
        sec, idx = self._current()
        if sec not in S.LIST_SECTIONS:
            return
        menu = QMenu(self)
        menu.addAction(f"Add {ADD_LABELS[sec]}", self.add_item)
        if idx is not None:
            menu.addAction("Duplicate", self.duplicate_item)
            menu.addAction("Move up", lambda: self.move_item(-1))
            menu.addAction("Move down", lambda: self.move_item(1))
            menu.addAction("Delete", self.delete_item)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def add_item(self):
        sec, _ = self._current()
        if sec not in S.LIST_SECTIONS:
            return
        d = S.defaults(S.LIST_SECTIONS[sec])
        names = {o.get("name") for o in self.model[sec]}
        base = d.get("name", sec)
        k = 1
        while f"{base}{k}" in names:
            k += 1
        d["name"] = f"{base}{k}"
        if sec in ("loaders", "emitters") and self.model["species"]:
            d["species"] = self.model["species"][0]["name"]
        self.model[sec].append(d)
        self.mark_dirty()
        self.populate_tree(select=(sec, len(self.model[sec]) - 1))

    def delete_item(self):
        sec, idx = self._current()
        if sec in S.LIST_SECTIONS and idx is not None:
            del self.model[sec][idx]
            self.mark_dirty()
            self.populate_tree(select=(sec, None))

    def duplicate_item(self):
        sec, idx = self._current()
        if sec in S.LIST_SECTIONS and idx is not None:
            d = copy.deepcopy(self.model[sec][idx])
            d["name"] = d.get("name", "") + "_copy"
            self.model[sec].insert(idx + 1, d)
            self.mark_dirty()
            self.populate_tree(select=(sec, idx + 1))

    def move_item(self, delta):
        sec, idx = self._current()
        L = self.model[sec]
        j = idx + delta
        if 0 <= j < len(L):
            L[idx], L[j] = L[j], L[idx]
            self.mark_dirty()
            self.populate_tree(select=(sec, j))

    # ======================================================================
    # settings panel
    # ======================================================================
    def show_settings(self, sec, idx):
        w = QWidget()
        v = QVBoxLayout(w)
        title = S.SECTION_TITLES.get(sec, sec)
        comps = {"1D-Z": ("z", "y", "x"), "2D-ZY": ("z", "y", "x"),
                 "2D-RZ": ("z", "r", "theta")}[self.model["simulation"]["geometry"]]
        ctx = {"species": [s["name"] for s in self.model["species"]],
               "objects": [o["name"] for o in self.model["objects"]], "components": comps}
        if sec == "root":
            title = "Model"
            from PySide6.QtWidgets import QFormLayout, QLineEdit
            fl = QFormLayout()
            name = QLineEdit(self.model.get("name", ""))
            desc = QPlainTextEdit(self.model.get("description", ""))

            def setname():
                self.model["name"] = name.text()
                self.mark_dirty()
                self.tree.topLevelItem(0).setText(0, name.text())

            def setdesc():
                self.model["description"] = desc.toPlainText()
                self.dirty = True
            name.editingFinished.connect(setname)
            desc.textChanged.connect(setdesc)
            fl.addRow("Name", name)
            fl.addRow("Description", desc)
            v.addLayout(fl)
            v.addWidget(QLabel(
                "<p>Build the model from top to bottom in the tree:<br>"
                "1. <b>Parameters</b> &nbsp;2. <b>Study/Solver</b> (geometry, ES/EM, time) &nbsp;"
                "3. <b>Domain & Mesh</b> &nbsp;4. <b>Boundaries</b><br>5. <b>Geometry objects</b> "
                "(conductors, dielectrics) &nbsp;6. <b>Species</b>, <b>loading</b>, <b>emission</b><br>"
                "7. <b>Applied fields</b>, <b>EM sources</b> &nbsp;8. <b>Probes / Output</b> &nbsp;"
                "9. <b>Run</b> (F5).</p>"
                "<p>Every numeric field accepts expressions of parameters and constants; voltages and "
                "currents may depend on time <i>t</i>, densities and profiles on <i>z, y (r)</i>.</p>"))
        elif sec == "parameters":
            t = ParameterTable(self.model["parameters"])
            t.changed.connect(self.mark_dirty)
            v.addWidget(t)
        elif sec == "boundaries":
            b = BoundaryPanel(self.model["boundaries"], self.model["simulation"]["geometry"])
            b.changed.connect(self.mark_dirty)
            v.addWidget(b)
        elif sec in S.SINGLE_SECTIONS:
            f = FieldForm(S.SINGLE_SECTIONS[sec], self.model[sec], ctx)
            f.changed.connect(lambda n, s=sec: self._single_changed(s, n))
            v.addWidget(f)
            if sec == "sweep":
                bt = QPushButton("Run sweep now")
                bt.clicked.connect(self.run_sweep)
                v.addWidget(bt)
            if sec == "domain":
                v.addWidget(self._mesh_buttons())
        elif sec in S.LIST_SECTIONS:
            if idx is None:
                v.addWidget(QLabel(f"{len(self.model[sec])} item(s).  Use <b>Add</b> (or right-click) to "
                                   f"create a new {ADD_LABELS[sec]}."))
                bt = QPushButton(f"Add {ADD_LABELS[sec]}")
                bt.clicked.connect(self.add_item)
                v.addWidget(bt)
            else:
                obj = self.model[sec][idx]
                title = f"{title}: {obj.get('name', '')}"
                f = FieldForm(S.LIST_SECTIONS[sec], obj, ctx)
                f.changed.connect(lambda n, s=sec, i=idx: self._item_changed(s, i, n))
                v.addWidget(f)
            if sec == "mesh_refinements":
                v.addWidget(self._mesh_buttons())
        elif sec == "results":
            v.addWidget(QLabel("Results are shown in the <b>Results</b> tab."))
            bt = QPushButton("Load results of this model")
            bt.clicked.connect(self.load_results)
            v.addWidget(bt)
        v.addStretch()
        self.settings_title.setText(f"Settings  ›  {title}")
        for lab in w.findChildren(QLabel):
            if lab.text().startswith("<p>") or len(lab.text()) > 60:
                lab.setWordWrap(True)
        # QScrollArea.setWidget() deletes the previous widget immediately; if a signal of one of its
        # editors triggered this rebuild, that is a use-after-free.  Detach it and delete it later.
        old = self.scroll.takeWidget()
        if old is not None:
            old.hide()
            old.deleteLater()
        self.scroll.setWidget(w)

    def _mesh_buttons(self):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 8, 0, 0)
        bt = QPushButton("Show mesh")
        bt.setObjectName("primary")
        bt.setToolTip("Draw the computational mesh in the Geometry tab (F6)")
        bt.clicked.connect(self.show_mesh)
        lay.addWidget(bt)
        hint = QLabel("<p><b>Along z / Along y</b> retains the existing full-domain graded bands. "
                      "Choose <b>2D</b>, select rectangle or polygon in <b>Region shape</b>, "
                      "and enter its coordinates or parameter expressions. Set <b>Cell size inside</b> "
                      "and <b>Safety buffer</b>. Protect possible cathode-return space as well as the beam. "
                      "Domain &amp; Mesh sets the background resolution. <b>Show mesh</b> previews "
                      "the actual triangles; the mesh is frozen before PIC. Regions are resolution "
                      "controls, not particle walls. Electrostatic solver only; smoothing must be 0.</p>")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        lay.addWidget(hint)
        return box

    def _draw_local_refinement(self, shape):
        """Interactive pre-run region entry; does not start or remesh a PIC run."""
        from matplotlib.widgets import RectangleSelector, PolygonSelector
        self.geo_timer.stop()
        previous = getattr(self, "_refine_selector", None)
        if previous is not None:
            previous.set_active(False)
            previous.disconnect_events()
        self.tabs.setCurrentIndex(0)
        self.draw_geometry()
        ax = self.geo_plot.axes[0]
        target = self.selected
        self.statusBar().showMessage("Drag a rectangle" if shape == "rectangle" else
                                     "Click polygon corners; click the first point again to finish. Escape cancels.")

        def finish(points):
            selector = getattr(self, "_refine_selector", None)
            if selector is not None:
                selector.set_active(False)
                selector.disconnect_events()
            self._refine_selector = None
            key = getattr(self, "_refine_cancel_key", None)
            if key is not None:
                self.geo_plot.canvas.mpl_disconnect(key)
                self._refine_cancel_key = None
            d = S.defaults(S.MESH_REFINE)
            if target and target[0] == "mesh_refinements" and target[1] is not None:
                index = target[1]
                d.update(self.model["mesh_refinements"][index])
            else:
                index = len(self.model["mesh_refinements"])
                self.model["mesh_refinements"].append(d)
                ev = M.evaluator(self.model)
                dom = self.model["domain"]
                h = min((ev.num(dom["z_max"])-ev.num(dom["z_min"]))/ev.num(dom["nz"]),
                        (ev.num(dom["y_max"])-ev.num(dom["y_min"]))/ev.num(dom["ny"]))
                d.update(name=f"local region {index+1}", size=f"{h/4:.12g}")
            fmt = lambda v: f"{v:.12g}"
            d.update(axis="local 2D", enabled=True, region_shape=shape)
            if shape == "rectangle":
                lo, hi = np.min(points, axis=0), np.max(points, axis=0)
                d.update(z0=fmt(lo[0]), z1=fmt(hi[0]), y0=fmt(lo[1]), y1=fmt(hi[1]))
            else:
                d["points"] = "; ".join(fmt(z)+","+fmt(y) for z,y in points)
            self.model["mesh_refinements"][index] = d
            self.mark_dirty()
            self.populate_tree(select=("mesh_refinements", index))
            self.chk_mesh.setChecked(True)
            self.draw_geometry()
            self.statusBar().showMessage("Local region added. Set cell size / buffer, preview, then run.", 8000)

        def rectangle(a, b):
            if None not in (a.xdata, a.ydata, b.xdata, b.ydata) and a.xdata != b.xdata and a.ydata != b.ydata:
                finish([(a.xdata,a.ydata),(b.xdata,b.ydata)])

        def polygon(points):
            if len(points) >= 3: finish(points)

        def cancel(event):
            if event.key != "escape": return
            selector = getattr(self, "_refine_selector", None)
            if selector is not None:
                selector.set_active(False)
                selector.disconnect_events()
            self._refine_selector = None
            self.geo_plot.canvas.mpl_disconnect(self._refine_cancel_key)
            self._refine_cancel_key = None
            self.draw_geometry()
            self.statusBar().showMessage("Local region drawing cancelled.", 3000)

        self._refine_selector = (RectangleSelector(ax, rectangle, button=[1], useblit=True)
                                 if shape == "rectangle" else PolygonSelector(ax, polygon, useblit=True))
        self._refine_cancel_key = self.geo_plot.canvas.mpl_connect("key_press_event", cancel)

    def show_mesh(self):
        self.chk_mesh.setChecked(True)
        self.tabs.setCurrentIndex(0)
        self.draw_geometry()

    def _single_changed(self, sec, name):
        self.mark_dirty()
        if sec == "simulation" and name == "geometry":
            g = self.model["simulation"]["geometry"]
            if g == "2D-RZ":
                self.model["boundaries"]["y_min"]["field"] = "axis"
                self.model["domain"]["y_min"] = "0"
            elif self.model["boundaries"]["y_min"]["field"] == "axis":
                self.model["boundaries"]["y_min"]["field"] = "conductor"
        self.geo_timer.start(300)

    def _item_changed(self, sec, idx, name):
        self.mark_dirty()
        # Relabel the tree item in place.  Never rebuild the tree / settings panel from here: this slot
        # runs inside the editor's own signal (e.g. editingFinished on focus-out), and destroying that
        # editor while Qt is still delivering its event crashes the application (segmentation fault).
        if name in ("name", "shape", "material", "model", "type", "preset"):
            it = self._tree_item(sec, idx)
            obj = self.model[sec][idx]
            extra = obj.get("shape") or obj.get("model") or obj.get("type") or obj.get("preset") or ""
            if sec == "objects":
                extra = f"{obj.get('material')} {obj.get('shape')}"
            label = obj.get("name", f"{sec}{idx}")
            if it is not None:
                it.setText(0, f"{label}  ({extra})" if extra else label)
            if name == "name" and self.selected == (sec, idx):
                self.settings_title.setText(f"Settings  ›  {S.SECTION_TITLES.get(sec, sec)}: {label}")
        self.geo_timer.start(300)

    def _tree_item(self, sec, idx):
        root = self.tree.topLevelItem(0)
        if root is None:
            return None
        for i in range(root.childCount()):
            top = root.child(i)
            for j in range(top.childCount()):
                ch = top.child(j)
                if ch.data(0, Qt.UserRole) == (sec, idx):
                    return ch
        return None

    # ======================================================================
    def draw_geometry(self):
        ax = self.geo_plot.axes[0]
        try:
            P.plot_geometry(ax, self.model, show_mesh=self.chk_mesh.isChecked(),
                            painted=self.chk_paint.isChecked(), selected=self.selected)
            from ..post.ticks import unit_axes
            unit_axes(ax, "m", "" if self.model["simulation"]["geometry"] == "1D-Z" else "m")
            from ..mesh.grid import grid_from_model
            mc = M.complete(self.model)
            g = grid_from_model(mc, M.evaluator(mc))
            nm = "r" if g.rz else "y"

            def rng(h, a):
                return (f"d{a} = {h[0]:.3g} m" if np.allclose(h, h[0], rtol=1e-9)
                        else f"d{a} = {h.min():.3g}…{h.max():.3g} m")
            if g.dim == 1:
                self.geo_info.setText(f"mesh: {g.nz} cells, {rng(g.hz, 'z')}")
            elif hasattr(g, "local_mesh"):
                mesh = g.local_mesh
                self.geo_info.setText(f"fixed local mesh: {len(mesh.points):,} nodes, "
                                      f"{len(mesh.triangles):,} triangles; "
                                      f"minimum triangle altitude = {mesh.hmin:.3g} m")
            else:
                self.geo_info.setText(f"mesh: {g.nz} × {g.ny} = {g.nz * g.ny:,} cells, {rng(g.hz, 'z')}, "
                                      f"{rng(g.hy, nm)}")
        except Exception as exc:  # noqa: BLE001
            ax.clear()
            ax.text(0.5, 0.5, f"Cannot draw geometry:\n{exc}", ha="center", va="center", transform=ax.transAxes)
        self.geo_plot.draw()

    # ======================================================================
    # file handling
    # ======================================================================
    def mark_dirty(self, *_):
        self.dirty = True
        self.update_title()

    def update_title(self):
        nm = os.path.basename(self.path) if self.path else "untitled"
        self.setWindowTitle(f"{APP_NAME} {__version__} - {nm}{' *' if self.dirty else ''}")

    def maybe_save(self):
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to the current model?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def _can_open_files(self):
        if self.proc is not None:
            self.statusBar().showMessage("Stop the current run before opening another model or results folder.", 5000)
            return False
        return True

    def _set_file_open_enabled(self, enabled):
        for action in self._file_open_actions:
            action.setEnabled(enabled)
        self.results.open_button.setEnabled(enabled)

    def new_model(self, g, s):
        if not self._can_open_files():
            return
        if not self.maybe_save():
            return
        self.model = M.new_model(g, s)
        self.path = None
        self.dirty = False
        self._clear_previous()
        self.results.clear()
        self.refresh_all()

    def open_dialog(self):
        if not self._can_open_files():
            return
        if not self.maybe_save():
            return
        fn, _ = QFileDialog.getOpenFileName(self, "Open model", self.settings.value("lastdir", ""),
                                            f"{APP_NAME} models (*.pic.json *.json)")
        if fn:
            self.open_file(fn)

    def open_file(self, fn):
        if not self._can_open_files():
            return
        try:
            self.model = M.load(fn)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.path = os.path.abspath(fn)
        self.settings.setValue("lastdir", os.path.dirname(self.path))
        self.dirty = False
        self._ok_example_edit = False
        self._clear_previous()
        self.refresh_all()
        self.msg(f"Opened {fn}")
        out = self.output_dir()
        if out and os.path.exists(os.path.join(out, "history.csv")):
            self.results.load(out)

    def open_example(self, p):
        if not self._can_open_files():
            return
        if not self.maybe_save():
            return
        self.model = M.load(p)
        self.model.pop("_path", None)
        self.path = None
        self.dirty = True
        self._clear_previous()
        self.results.clear()
        self.refresh_all()
        self.msg(f"Loaded example {os.path.basename(p)} (use Save as... to keep a copy)")

    def save(self):
        if not self.path:
            return self.save_as()
        ex_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
        if os.path.dirname(os.path.abspath(self.path)) == ex_dir and not getattr(self, "_ok_example_edit", False):
            r = QMessageBox.question(
                self, "Shipped example",
                "This file is one of the shipped examples.\n\nSave your changes as a new file "
                "(recommended), so the original example stays intact?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if r == QMessageBox.Cancel:
                return False
            if r == QMessageBox.Yes:
                return self.save_as()
            self._ok_example_edit = True
        try:
            M.save(self.model, self.path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.model["_path"] = self.path
        self.dirty = False
        self.update_title()
        self.msg(f"Saved {self.path}")
        return True

    def save_as(self):
        nm = (self.model.get("name") or "model").replace(" ", "_").replace("/", "_")
        fn, _ = QFileDialog.getSaveFileName(self, "Save model",
                                            os.path.join(self.settings.value("lastdir", ""), nm + ".pic.json"),
                                            f"{APP_NAME} models (*.pic.json)")
        if not fn:
            return False
        if not fn.endswith(".json"):
            fn += ".pic.json"
        self.path = fn
        self.settings.setValue("lastdir", os.path.dirname(fn))
        return self.save()

    def output_dir(self):
        if not self.path:
            return None
        return os.path.join(os.path.dirname(self.path), self.model["diagnostics"]["output_dir"] or "output")

    # ======================================================================
    # checks & run
    # ======================================================================
    def msg(self, text):
        self.log.appendPlainText(text)

    def check_model(self):
        msgs = M.validate(self.model)
        for lvl, m in msgs:
            self.msg(f"[{lvl}] {m}")
        errs = [m for lvl, m in msgs if lvl == "error"]
        if errs:
            QMessageBox.warning(self, "Model check", "\n".join(errs))
        return not errs

    def mesh_info(self):
        try:
            from ..core.simulation import Simulation
            sim = Simulation(self.model, out_dir=os.devnull, write_output=False, log=self.msg)
            self.msg(f"dt = {sim.dt:.4g} s, steps = {sim.n_steps}, particles at t=0: "
                     f"{sum(s.n for s in sim.species)}")
        except Exception as exc:  # noqa: BLE001
            self.msg(f"[error] {exc}")

    def _clear_previous(self):
        """A different model was loaded: drop the live plots of the previous one."""
        if self.proc is None:
            self.live_timer.stop()
            self.run_out = None
            self.live.clear()
            self.progress.setValue(0)

    def _launch(self, args, out_dir):
        if self.proc is not None:
            QMessageBox.information(self, "Busy", "A simulation is already running.")
            return
        import time as _time
        self.launch_time = _time.time()
        if out_dir:              # never show the previous run's live file for the new run
            try:
                os.remove(os.path.join(out_dir, "live.npz"))
            except OSError:
                pass
            self.live.clear("Starting simulation...")
        self.proc = QProcess(self)
        self._set_file_open_enabled(False)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_proc)
        self.proc.finished.connect(self._proc_done)
        self.proc.errorOccurred.connect(self._proc_error)
        self.run_out = out_dir
        env = self.proc.processEnvironment()
        from PySide6.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        env.insert("PYTHONPATH", root + os.pathsep + env.value("PYTHONPATH", ""))
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.msg("$ " + " ".join(args))
        self.proc.start(args[0], args[1:])
        self.progress.setValue(0)
        self.status.setText("Running...")
        if out_dir:
            self.live_timer.start(1000)
            self.tabs.setCurrentWidget(self.live)

    def _prepare_run(self):
        if not self.check_model():
            return None
        if not self.path:
            QMessageBox.information(self, "Save first", "Please save the model before running.")
            if not self.save_as():
                return None
        elif self.dirty:
            self.save()
        return self.output_dir()

    def run(self):
        if self.model.get("parallel", {}).get("mode") == "mpi":
            n = int(self.model["parallel"].get("ranks", 4) or 4)
            if n > 1:
                return self.run_mpi(n)
        out = self._prepare_run()
        if out:
            self._launch([sys.executable, "-m", "picstudio", "run", self.path, "--out", out], out)

    @staticmethod
    def _mpi_launcher():
        """mpiexec next to this Python first (pip 'mpich'/'openmpi' wheels, conda), then on the PATH."""
        here = os.path.dirname(sys.executable)
        for name in ("mpiexec", "mpirun", "mpiexec.exe", "mpiexec.hydra"):
            c = os.path.join(here, name)
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return shutil.which("mpiexec") or shutil.which("mpirun")

    def _mpi_ready(self):
        import importlib.util
        exe = self._mpi_launcher()
        has_mpi4py = importlib.util.find_spec("mpi4py") is not None
        if exe and has_mpi4py:
            return exe
        missing = []
        if not has_mpi4py:
            missing.append("the Python package mpi4py")
        if not exe:
            missing.append("an MPI library with mpiexec")
        QMessageBox.warning(
            self, "MPI not available",
            "MPI needs " + " and ".join(missing) + f".<br><br>Install both into the {APP_NAME} Python "
            f"environment, then restart {APP_NAME}:<br><tt>python -m pip install mpi4py mpich</tt><br>"
            "(or <tt>brew install open-mpi</tt> on macOS / Microsoft MPI on Windows, plus "
            "<tt>pip install mpi4py</tt>).<br><br>Until then use Parallel mode = threads (all cores in one "
            "process).")
        return None

    def run_mpi(self, n=None):
        out = self._prepare_run()
        if not out:
            return
        exe = self._mpi_ready()
        if not exe:
            return
        if n is None:
            n0 = int(self.model["parallel"].get("ranks", 4) or 4)
            n, ok = QInputDialog.getInt(self, "MPI ranks", "Number of MPI processes:", n0, 1, 1024)
            if not ok:
                return
        self._launch([exe, "-n", str(int(n)), sys.executable, "-m", "picstudio", "run", self.path, "--out", out,
                      "--mode", "mpi"], out)

    def run_sweep(self):
        if not self.model["sweep"].get("parameter"):
            QMessageBox.information(self, "Sweep", "Set the sweep parameter in 'Parametric Sweep' first.")
            return
        out = self._prepare_run()
        if out:
            self._launch([sys.executable, "-m", "picstudio", "sweep", self.path], None)

    def calibrate_q(self):
        lossy = [o for o in self.model["objects"] if o.get("material") == "lossy"]
        if not lossy:
            QMessageBox.information(self, "Calibrate Q", "Add a geometry object with material 'lossy' "
                                    "(loss specified by Q) and a probe inside the resonator first.")
            return
        if not self.model["probes"]:
            QMessageBox.information(self, "Calibrate Q", "Add a probe inside the resonator (e.g. a cavity) "
                                    "- its field component is used to excite and observe the mode.")
            return
        if not self._prepare_run_path():
            return
        self._after_proc = self._reload_after_qcal
        self.msg("Cold-test Q calibration (field-only ring-down; a few minutes)...")
        self._launch([sys.executable, "-m", "picstudio", "qcal", self.path], None)

    def _prepare_run_path(self):
        if not self.path:
            QMessageBox.information(self, "Save first", "Please save the model first.")
            if not self.save_as():
                return False
        elif self.dirty:
            self.save()
        return True

    def _reload_after_qcal(self):
        try:
            self.model = M.load(self.path)
            self.dirty = False
            self.refresh_all()
            self.msg("Loss port calibrated - 'fill' updated in the model.")
        except Exception as exc:  # noqa: BLE001
            self.msg(f"[error] {exc}")

    def toggle_pause(self):
        """Pause / resume the running simulation (the solver waits while the file PAUSE exists)."""
        if self.proc is None or not self.run_out:
            return
        fp = os.path.join(self.run_out, "PAUSE")
        try:
            if os.path.exists(fp):
                os.remove(fp)
                self._pause_look(False)
                self.msg("Resume requested")
            else:
                open(fp, "w").close()
                self._pause_look(True)
                self.msg("Pause requested (the run waits; press Resume to continue or Stop to end it)")
        except OSError as exc:
            self.msg(f"[error] pause: {exc}")

    def _pause_look(self, paused):
        from .theme import action_icon
        self.act_pause.setText("Resume" if paused else "Pause")
        self.act_pause.setIcon(action_icon("run" if paused else "pause"))

    def _reset_pause(self):
        self._pause_look(False)
        if self.run_out:
            try:
                os.remove(os.path.join(self.run_out, "PAUSE"))
            except OSError:
                pass

    def stop(self):
        if self.proc is None:
            return
        self._user_stop = True
        if self.run_out:
            try:
                open(os.path.join(self.run_out, "STOP"), "w").close()
            except OSError:
                pass
        self._reset_pause()
        # force-kill only THIS run, and only if it has not finished cleanly after a grace period
        # (before, a pending timer could kill the NEXT run if Run was pressed again quickly)
        p = self.proc
        QTimer.singleShot(30000, lambda p=p: self._kill_if_running(p))
        self.msg("Stop requested")

    def _kill_if_running(self, p=None):
        if p is None or self.proc is not p:
            return
        if p.state() != QProcess.NotRunning:
            self.msg("The run did not stop within 30 s - terminating it")
            p.kill()

    def _read_proc(self):
        if self.proc is None:
            return
        txt = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in txt.splitlines():
            if line.startswith("PAUSED"):
                self.status.setText(f"Paused at step {line.split()[1]}  (press Resume to continue)")
                self.live_timer.stop()
                self._poll_live()
            elif line.startswith("RESUMED"):
                self.status.setText("Running...")
                self.live_timer.start(1000)
            elif line.startswith("PROGRESS"):
                try:
                    _, s, n, t, npart, el = line.split()
                    self.progress.setMaximum(int(n))
                    self.progress.setValue(int(s))
                    self.status.setText(f"step {s}/{n}   t = {float(t):.4g} s   particles = {npart}   "
                                        f"elapsed {float(el):.1f} s")
                except ValueError:
                    pass
            elif line.strip():
                self.msg(line)

    def _proc_error(self, error):
        if error == QProcess.FailedToStart and self.proc is not None:
            self.msg(f"[error] Could not start solver: {self.proc.errorString()}")
            self._proc_done(-1, QProcess.NormalExit)

    def _proc_done(self, code, _status):
        self._read_proc()
        user_stop = getattr(self, "_user_stop", False)
        self._user_stop = False
        crashed = _status == QProcess.CrashExit or code in (9, 137, -9)
        if user_stop:
            self.msg("Run stopped by the user (results up to this time are saved)")
        elif crashed:
            self.msg(f"[error] The solver process was killed (exit code {code}). Usual causes: the "
                     "operating system ran out of memory (too many MPI ranks x threads x particles), "
                     "or the process was terminated from outside. Try fewer MPI ranks or threads, "
                     "fewer particles per cell, or Parallel mode = threads.")
        else:
            self.msg(f"Process finished with exit code {code}")
        self.status.setText("Stopped" if user_stop else ("Finished" if code == 0 else f"Finished (code {code})"))
        self._pause_look(False)
        self.live_timer.stop()
        self._poll_live()
        self.proc = None
        self._set_file_open_enabled(True)
        if self.run_out:
            self.results.load(self.run_out)
        self.run_out = None
        cb = getattr(self, "_after_proc", None)
        self._after_proc = None
        if cb is not None and code == 0:
            cb()

    def _poll_live(self):
        if not self.run_out:
            return
        from ..post.reader import Results
        fn = os.path.join(self.run_out, "live.npz")
        try:            # ignore a file left over from an earlier run
            if os.path.getmtime(fn) < getattr(self, "launch_time", 0) - 1:
                return
        except OSError:
            return
        d = Results.__new__(Results)
        d.dir = self.run_out
        live = Results.live(d)
        if live is not None:
            self.live.update_from(live)

    def load_results(self):
        if not self._can_open_files():
            return
        out = self.output_dir()
        if out and os.path.isdir(out):
            self.results.load(out)
            self.tabs.setCurrentWidget(self.results)
        else:
            QMessageBox.information(self, "Results", "No results found for this model (run it first).")

    # ======================================================================
    def show_validation(self):
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "results",
                                         "VALIDATION_REPORT.md"))
        from PySide6.QtWidgets import QTextBrowser
        w = QTextBrowser()
        w.setSearchPaths([os.path.dirname(p)])
        if os.path.exists(p):
            w.setMarkdown(open(p, encoding="utf-8").read())
        else:
            w.setPlainText("Run 'python benchmarks/run_all.py' to generate the validation report.")
        w.setWindowTitle(f"{APP_NAME} validation report")
        w.resize(1000, 800)
        w.show()
        self._val = w

    def show_guide(self, name="USER_GUIDE.md"):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", name))
        txt = open(root, encoding="utf-8").read() if os.path.exists(root) else f"docs/{name} not found"
        from PySide6.QtWidgets import QTextBrowser
        w = QTextBrowser()
        w.setMarkdown(txt)
        w.setWindowTitle(f"{APP_NAME} - " + name)
        w.resize(900, 700)
        w.show()
        self._guide = w

    def about(self):
        logo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "app_logo.svg")
        box = QMessageBox(self)
        box.setWindowTitle(f"About {APP_NAME}")
        from PySide6.QtGui import QIcon
        box.setIconPixmap(QIcon(logo).pixmap(96, 96))
        box.setText(f"<h2>{APP_NAME} {__version__}</h2><p><b>{APP_TAGLINE}</b><br>{APP_SUBTITLE}</p>"
                    "<p>Particle-in-cell simulation of diodes, beams, plasmas and vacuum electron devices in 1D (z), "
                    "2D Cartesian (z, y) and 2D axisymmetric (z, r) geometry. Electrostatic solver: finite elements "
                    "(scikit-fem). Electromagnetic solver: Yee FDTD with charge-conserving deposition.</p>"
                    f"<p><b>Author:</b> {AUTHOR}<br>{AFFILIATION}<br>"
                    "<b>Website:</b> <a href='https://www.arifuzzaman.info'>www.arifuzzaman.info</a></p>"
                    f"<p>{COPYRIGHT}. Released under the {LICENSE}.<br>"
                    "Particle kernels follow WarpX (BSD-3-Clause-LBNL); see THIRD_PARTY_NOTICES.md.</p>"
                    f"<p><a href='{URL}'>{URL}</a><br>DOI: <a href='https://doi.org/{DOI}'>{DOI}</a></p>"
                    "<p>If you use VEXPACE in published work, please cite it (Help &rarr; How to cite VEXPACE).</p>")
        box.setTextFormat(Qt.RichText)
        box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        box.exec()

    def cite(self):
        text = CITATION
        bib = ("@software{faisal_vexpace,\n"
               "  author    = {Faisal, Md Arifuzzaman},\n"
               "  title     = {{VEXPACE}: An Open-Source Particle-in-Cell Simulation Framework for Plasma and Vacuum Electronics},\n"
               "  year      = {2026},\n"
               "  publisher = {Zenodo},\n"
               f"  doi       = {{{DOI}}},\n"
               f"  url       = {{https://doi.org/{DOI}}}\n}}")
        box = QMessageBox(self)
        box.setWindowTitle("How to cite VEXPACE")
        box.setText("<p>If VEXPACE contributes to published work, please cite:</p>"
                    f"<p>{text}</p><p>BibTeX:</p><pre>{bib}</pre>")
        box.setTextFormat(Qt.RichText)
        box.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.exec()

    def closeEvent(self, ev):
        if self.proc is not None:
            if QMessageBox.question(self, "Running", "A simulation is running. Stop it and quit?") != \
                    QMessageBox.Yes:
                ev.ignore()
                return
            self.proc.kill()
        if self.maybe_save():
            ev.accept()
        else:
            ev.ignore()
