#!/usr/bin/env python3
"""Launch the ETROYL electromagnetic design and optimization GUI."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.theme import apply_theme
from PySide6.QtCore import QSettings


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ETROYL Electromagnetic Design")
    app.setOrganizationName("ETROYL")
    # Use a consistent cross-platform widget style and ETROYL's own default
    # appearance instead of inheriting the desktop environment theme.
    app.setStyle("Fusion")
    settings = QSettings("ETROYL", "ETROYL Electromagnetic Design")
    apply_theme(app, str(settings.value("appearance/theme", "dark")))
    window = MainWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
