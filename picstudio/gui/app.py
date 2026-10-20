"""GUI entry point."""
from __future__ import annotations

import sys


def main(path=None):
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is required for the GUI:  pip install PySide6")
        return 1
    from .main_window import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    from .. import APP_NAME
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    import os
    from PySide6.QtGui import QIcon
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "app_logo.svg")))
    w = MainWindow(path)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
