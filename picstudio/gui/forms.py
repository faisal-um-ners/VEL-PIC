"""Schema-driven property editors (the 'Settings' panel of the model builder)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..expr import Evaluator
from ..model import schema as S


class FieldForm(QWidget):
    """Form for one dict according to a list of schema Fields."""
    changed = Signal(str)          # field name
    structure_changed = Signal()   # visibility may have changed (choice field)

    def __init__(self, fields, values: dict, context=None, parent=None, explain=True):
        super().__init__(parent)
        self.fields = fields
        self.values = values
        self.context = context or {}
        self.widgets = {}
        self.rows = {}
        self.form = QFormLayout(self)
        self.form.setLabelAlignment(Qt.AlignRight)
        self.form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        for f in fields:
            w = self._make(f)
            lab = QLabel(f.label + (f"  [{f.unit}]" if f.unit else ""))
            if f.help:
                lab.setToolTip(f.help)
                w.setToolTip(f.help)
            self.form.addRow(lab, w)
            self.widgets[f.name] = w
            self.rows[f.name] = lab
        self.material_summary = None
        if any(f.name == "material" for f in fields):
            self.material_summary = QLabel()
            self.material_summary.setWordWrap(True)
            self.material_summary.setTextFormat(Qt.RichText)
            self.material_summary.setObjectName("materialSummary")
            self.material_summary.setStyleSheet("QLabel#materialSummary { font-size: 11px; padding-top: 8px; }")
            self.form.addRow(self.material_summary)
        self.explain = None
        if explain:
            self.explain = QLabel()
            self.explain.setWordWrap(True)
            self.explain.setTextFormat(Qt.RichText)
            self.explain.setObjectName("formExplain")
            self.explain.setStyleSheet("QLabel#formExplain { font-size: 11px; padding-top: 10px; }")
            self.form.addRow(self.explain)
        self.update_visibility()

    def _make(self, f):
        v = self.values.get(f.name, f.default)
        if f.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(v))
            w.toggled.connect(lambda val, n=f.name: self._set(n, bool(val)))
            return w
        if f.kind == "int":
            w = QSpinBox()
            w.setRange(-1000000, 100000000)
            try:
                w.setValue(int(v))
            except (TypeError, ValueError):
                w.setValue(0)
            w.valueChanged.connect(lambda val, n=f.name: self._set(n, int(val)))
            return w
        if f.kind == "choice" or f.name in ("species",) and self.context.get("species") is not None:
            w = QComboBox()
            choices = list(f.choices)
            if f.name == "species":
                choices = list(self.context.get("species", []))
            if f.name == "object" and self.context.get("objects") is not None:
                choices = list(self.context.get("objects"))
            if v not in choices and v not in (None, ""):
                choices.append(str(v))
            for c in choices:
                w.addItem(f.choice_labels.get(c,str(c)),c)
            w.setCurrentIndex(choices.index(v) if v in choices else -1)
            w.currentIndexChanged.connect(lambda index, n=f.name, ww=w:
                self._set(n, ww.itemData(index), structural=True) if index>=0 else None)
            return w
        if f.name == "object" and self.context.get("objects") is not None:
            w = QComboBox()
            w.setEditable(True)
            w.addItems(list(self.context["objects"]))
            w.setCurrentText(str(v))
            w.currentTextChanged.connect(lambda val, n=f.name: self._set(n, val))
            return w
        if f.kind == "vec3":
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            vals = list(v) if isinstance(v, (list, tuple)) else ["0", "0", "0"]
            edits = []
            comp = self.context.get("components", ("z", "y", "x"))
            for k in range(3):
                e = QLineEdit(str(vals[k]))
                e.setPlaceholderText(comp[k])
                e.setToolTip(f"{comp[k]} component")
                edits.append(e)
                h.addWidget(QLabel(comp[k]))
                h.addWidget(e)

            def upd(_=None, n=f.name, es=edits):
                self._set(n, [x.text() for x in es])
            for e in edits:
                e.editingFinished.connect(upd)
            return box
        if f.kind == "text":
            w = QPlainTextEdit(str(v))
            w.setMaximumHeight(80)
            w.textChanged.connect(lambda n=f.name, ww=w: self._set(n, ww.toPlainText(), quiet=True))
            return w
        w = QLineEdit("" if v is None else str(v))
        w.editingFinished.connect(lambda n=f.name, ww=w: self._set(n, ww.text()))
        if f.kind == "expr":
            w.setPlaceholderText("number or expression")
        return w

    def _set(self, name, val, structural=False, quiet=False):
        if self.values.get(name) == val:
            return
        self.values[name] = val
        if structural:
            self.update_visibility()
            self.structure_changed.emit()
        self.changed.emit(name)

    def update_visibility(self):
        for f in self.fields:
            vis = S.is_visible(f, self.values)
            self.widgets[f.name].setVisible(vis)
            self.rows[f.name].setVisible(vis)
        if self.material_summary is not None:
            material = self.values.get("material", "conductor")
            lines = S.MATERIAL_SUMMARIES.get(material)
            self.material_summary.setVisible(lines is not None)
            if lines:
                self.material_summary.setText("<b>" + material.capitalize() + "</b>: " +
                                              lines[0] + "<br>" + lines[1])
        if self.explain is not None:
            self.explain.setText(explain_html(self.fields, self.values))


def explain_html(fields, values, title="Meaning of the settings"):
    """Paragraph describing every visible input of a form (shown at the end of the form)."""
    import html
    parts = []
    for f in fields:
        if not S.is_visible(f, values) or not f.help:
            continue
        unit = f" [{f.unit}]" if f.unit else ""
        parts.append(f"<b>{html.escape(f.label)}</b>{html.escape(unit)}: {html.escape(f.help)}")
    if not parts:
        return ""
    return f"<hr><p><b>{title}</b></p><p>" + "<br>".join(parts) + "</p>"


class ParameterTable(QWidget):
    changed = Signal()

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<b>Global parameters</b> - usable in every numeric field "
                             "(constants: c, e, me, mp, eps0, mu0, kB, pi)."))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Expression", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.table)
        h = QHBoxLayout()
        b1 = QPushButton("Add parameter")
        b2 = QPushButton("Remove selected")
        b1.clicked.connect(self.add_row)
        b2.clicked.connect(self.remove_row)
        h.addWidget(b1)
        h.addWidget(b2)
        h.addStretch()
        lay.addLayout(h)
        self.msg = QLabel("")
        self.msg.setStyleSheet("color: #b00")
        lay.addWidget(self.msg)
        self._filling = True
        for k, v in params.items():
            self._append(k, v)
        self._filling = False
        self.table.itemChanged.connect(self._edited)
        self.refresh_values()

    def _append(self, k, v):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(k))
        self.table.setItem(r, 1, QTableWidgetItem(str(v)))
        it = QTableWidgetItem("")
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 2, it)

    def add_row(self):
        self._filling = True
        n = 1
        while f"p{n}" in self.params:
            n += 1
        self._append(f"p{n}", "1")
        self._filling = False
        self._edited()

    def remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self._edited()

    def _edited(self, *_):
        if self._filling:
            return
        new = {}
        for r in range(self.table.rowCount()):
            k = self.table.item(r, 0)
            v = self.table.item(r, 1)
            if k and k.text().strip():
                new[k.text().strip()] = v.text() if v else "0"
        self.params.clear()
        self.params.update(new)
        self.refresh_values()
        self.changed.emit()

    def refresh_values(self):
        self._filling = True
        try:
            ev = Evaluator(self.params)
            self.msg.setText("")
            for r in range(self.table.rowCount()):
                k = self.table.item(r, 0).text().strip()
                val = ev.params.get(k)
                txt = f"{val:.6g}" if isinstance(val, float) else ("" if val is None else str(val)[:40])
                self.table.item(r, 2).setText(txt)
        except Exception as exc:  # noqa: BLE001
            self.msg.setText(str(exc))
        self._filling = False


class BoundaryPanel(QWidget):
    changed = Signal()

    def __init__(self, boundaries: dict, geometry: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        names = {"z_min": "z = z_min", "z_max": "z = z_max", "y_min": "y = y_min  (r = r_min / axis)",
                 "y_max": "y = y_max  (r = r_max)"}
        sides = ("z_min", "z_max") if geometry == "1D-Z" else S.SIDES
        self.forms = []
        for s in sides:
            g = QGroupBox(names[s])
            v = QVBoxLayout(g)
            f = FieldForm(S.BOUNDARY, boundaries[s], explain=False)
            f.changed.connect(lambda *_: self.changed.emit())
            v.addWidget(f)
            lay.addWidget(g)
            self.forms.append(f)
        ex = QLabel(explain_html(S.BOUNDARY, {"field": "conductor"})
                    + "<p><b>Absorbing layer cells</b> (absorbing sides only): " + S.BOUNDARY[-1].help + "</p>")
        ex.setWordWrap(True)
        ex.setTextFormat(Qt.RichText)
        ex.setStyleSheet("font-size: 11px; padding-top: 6px;")
        lay.addWidget(ex)
        lay.addStretch()
