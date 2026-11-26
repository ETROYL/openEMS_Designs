#!/usr/bin/env python3
"""Launch the ETROYL electromagnetic design and optimization GUI."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ETROYL Electromagnetic Design")
    app.setOrganizationName("ETROYL")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
