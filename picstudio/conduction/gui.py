"""Standalone AC/DC contact study dialog; does not modify PIC models or runner."""
from pathlib import Path
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog,
    QLabel, QHBoxLayout, QMessageBox, QPlainTextEdit
)


class ConductionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VEXPACE - AC/DC Contact Conduction (FEM)")
        self.resize(690, 430)
        self.process = None
        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Independent 2D anisotropic FEM with a parallel R||C contact interface.\n"
            "Open a JSON contact configuration. Existing PIC models are not modified."
        ))
        form = QFormLayout()
        root.addLayout(form)
        default_example = Path(__file__).resolve().parents[2] / "examples" / "contact_acdc_reference.json"
        self.config = QLineEdit(str(default_example))
        self.out = QLineEdit(str(Path.cwd() / "contact_results"))
        self.fmin = QLineEdit("1e3")
        self.fmax = QLineEdit("1e12")
        self.nfreq = QLineEdit("31")
        row = QHBoxLayout()
        row.addWidget(self.config)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.choose_config)
        row.addWidget(browse)
        form.addRow("Contact JSON:", row)
        outrow=QHBoxLayout()
        outrow.addWidget(self.out)
        choose=QPushButton("Choose...")
        choose.clicked.connect(self.choose_out)
        outrow.addWidget(choose)
        form.addRow("Output folder:", outrow)
        form.addRow("AC start (Hz):",self.fmin)
        form.addRow("AC end (Hz):",self.fmax)
        form.addRow("Log points:",self.nfreq)
        self.run_button=QPushButton("Run DC + AC frequency sweep")
        self.run_button.clicked.connect(self.run_study)
        root.addWidget(self.run_button)
        self.status=QPlainTextEdit()
        self.status.setReadOnly(True)
        root.addWidget(self.status)

    def choose_config(self):
        p,_=QFileDialog.getOpenFileName(self,"Contact configuration",self.config.text(),"JSON (*.json)")
        if p:self.config.setText(p)

    def choose_out(self):
        p=QFileDialog.getExistingDirectory(self,"Results folder",self.out.text())
        if p:self.out.setText(p)

    def run_study(self):
        try:
            fmin=float(self.fmin.text())
            fmax=float(self.fmax.text())
            n=int(self.nfreq.text())
            if fmin<=0 or fmax<=fmin or n<2:
                raise ValueError("Require 0 < start < end, and at least 2 frequencies")
            if not Path(self.config.text()).is_file():
                raise ValueError("Select an existing configuration JSON")
        except ValueError as exc:
            QMessageBox.warning(self,"Invalid contact study",str(exc))
            return
        self.status.clear()
        self.run_button.setEnabled(False)
        self.process=QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([
            "-m","picstudio.conduction",self.config.text(),"--out",self.out.text(),
            "--fmin",str(fmin),"--fmax",str(fmax),"--points",str(n)
        ])
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(
            lambda:self.status.insertPlainText(bytes(self.process.readAllStandardOutput()).decode(errors="replace")))
        self.process.finished.connect(self.finished)
        self.process.start()

    def finished(self, exitcode, _status):
        self.run_button.setEnabled(True)
        self.status.appendPlainText(
            "\nCompleted. Plots and CSV outputs are in: "+self.out.text()
            if exitcode==0 else "\nConduction solver failed. Check the messages above."
        )


def open_dialog(parent):
    dlg=ConductionDialog(parent)
    dlg.exec()
