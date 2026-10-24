"""Run with QT_QPA_PLATFORM=offscreen in the tfgpu environment."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    from picstudio.gui.main_window import MainWindow
except ImportError:
    MainWindow = None


@unittest.skipIf(MainWindow is None, "optional GUI dependencies unavailable")
class FileOpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.draw = patch.object(MainWindow, 'draw_geometry')
        self.draw.start()
        self.window = MainWindow()
        self.lastdir = self.window.settings.value('lastdir')

    def tearDown(self):
        w = self.window
        if w.proc is not None:
            w.proc.kill()
            w.proc.waitForFinished(3000)
            self.wait_until(lambda: w.proc is None)
        w.dirty = False
        w.close()
        if self.lastdir is None:
            w.settings.remove('lastdir')
        else:
            w.settings.setValue('lastdir', self.lastdir)
        self.draw.stop()

    def wait_until(self, condition):
        deadline = time.monotonic() + 10
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
        self.assertTrue(condition())

    def assert_open_enabled(self, enabled):
        for a in self.window._file_open_actions:
            self.assertEqual(a.isEnabled(), enabled, a.text())
        self.assertEqual(self.window.results.open_button.isEnabled(), enabled)

    def test_running_paused_and_finished(self):
        w = self.window
        self.assert_open_enabled(True)
        with tempfile.TemporaryDirectory() as directory:
            w._launch([sys.executable, '-c', 'import sys; sys.stdin.read()'], directory)
            self.wait_until(lambda: w.proc.state() == QProcess.Running)
            proc = w.proc
            original_model, original_path = w.model, w.path
            for paused in (False, True):
                if paused:
                    w.toggle_pause()
                    self.assertTrue((Path(directory)/'PAUSE').exists())
                self.assert_open_enabled(False)
                with patch.object(w, 'maybe_save') as save, \
                     patch('picstudio.gui.main_window.M.load') as load, \
                     patch('picstudio.gui.results.QFileDialog.getExistingDirectory') as choose:
                    w.new_model('1D-Z', 'electrostatic')
                    w.open_dialog()
                    w.open_file('must-not-be-opened.pic.json')
                    w.open_example('must-not-be-opened.pic.json')
                    w.load_results()
                    w.results.choose_dir()
                    w.results.load(directory)
                    save.assert_not_called()
                    load.assert_not_called()
                    choose.assert_not_called()
                self.assertIs(w.model, original_model)
                self.assertEqual(w.path, original_path)
                self.assertIsNone(w.results.res)
                self.assertIs(w.proc, proc)
            w.toggle_pause()
            self.assertFalse((Path(directory)/'PAUSE').exists())
            self.assert_open_enabled(False)
            proc.closeWriteChannel()
            self.wait_until(lambda: w.proc is None)
            self.assert_open_enabled(True)
            self.assertIsNotNone(w.results.res)  # finished-run results still load
            self.assertEqual(w.results.res.dir, directory)

    def test_failed_start_restores_opening(self):
        w = self.window
        w._launch(['/definitely/not/an/executable/vexpace'], None)
        self.wait_until(lambda: w.proc is None)
        self.assert_open_enabled(True)
        self.assertIn('Could not start solver', w.log.toPlainText())


if __name__ == '__main__':
    unittest.main()
