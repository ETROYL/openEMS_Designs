"""Application-wide ETROYL appearance themes for the desktop GUI."""
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor


DARK = {
    "window": "#171b22",
    "surface": "#202630",
    "surface_alt": "#252d38",
    "input": "#151a21",
    "border": "#3a4655",
    "text": "#e7edf5",
    "muted": "#a5b1c0",
    "accent": "#3995ff",
    "accent_hover": "#58a6ff",
    "selection": "#244d78",
}

LIGHT = {
    "window": "#f3f5f8",
    "surface": "#ffffff",
    "surface_alt": "#e9eef4",
    "input": "#ffffff",
    "border": "#cbd4df",
    "text": "#202b38",
    "muted": "#667487",
    "accent": "#176fc1",
    "accent_hover": "#0f5da6",
    "selection": "#cfe4fa",
}


def apply_theme(app, theme_name: str = "dark") -> str:
    """Apply a complete application palette and stylesheet; return chosen theme."""
    name = "light" if str(theme_name).lower() == "light" else "dark"
    c = LIGHT if name == "light" else DARK

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(c["window"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(c["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(c["input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(c["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(c["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(c["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(c["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(c["surface"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(c["text"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(c["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Mid, QColor(c["border"]))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(c["border"]))
    app.setPalette(palette)

    app.setStyleSheet("""
        QWidget {
            color: %(text)s;
            selection-background-color: %(selection)s;
            selection-color: %(text)s;
        }
        QMainWindow, QDialog { background-color: %(window)s; }
        QToolBar {
            background: %(surface)s;
            border: 0;
            border-bottom: 1px solid %(border)s;
            spacing: 7px;
            padding: 6px 10px;
        }
        QToolBar::separator {
            background: %(border)s;
            width: 1px;
            margin: 4px 6px;
        }
        QTabWidget::pane {
            background: %(window)s;
            border: 1px solid %(border)s;
            top: -1px;
        }
        QTabBar::tab {
            background: %(surface_alt)s;
            color: %(muted)s;
            border: 1px solid %(border)s;
            padding: 9px 16px;
            margin-right: 3px;
        }
        QTabBar::tab:selected {
            background: %(surface)s;
            color: %(accent)s;
            border-bottom: 2px solid %(accent)s;
        }
        QTabBar::tab:hover:!selected { color: %(text)s; }
        QGroupBox {
            background: %(surface)s;
            border: 1px solid %(border)s;
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px 8px 8px 8px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: %(muted)s;
        }
        QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: %(input)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            border-radius: 4px;
            padding: 5px 7px;
        }
        QComboBox::drop-down {
            border: 0;
            width: 22px;
        }
        QComboBox QAbstractItemView {
            background: %(surface)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            selection-background-color: %(selection)s;
        }
        QPushButton {
            background: %(surface_alt)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            border-radius: 5px;
            padding: 7px 12px;
        }
        QPushButton:hover {
            background: %(selection)s;
            border-color: %(accent)s;
        }
        QPushButton:pressed { background: %(surface)s; }
        QPushButton:disabled { color: %(muted)s; background: %(surface)s; }
        QCheckBox, QRadioButton { spacing: 7px; }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 15px; height: 15px;
        }
        QProgressBar {
            background: %(surface_alt)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }
        QScrollArea { border: 0; background: %(window)s; }
        QScrollBar:vertical {
            background: %(surface)s; width: 12px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: %(border)s; min-height: 24px; border-radius: 5px;
        }
        QScrollBar:horizontal {
            background: %(surface)s; height: 12px; margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: %(border)s; min-width: 24px; border-radius: 5px;
        }
        QStatusBar {
            background: %(surface)s;
            color: %(muted)s;
            border-top: 1px solid %(border)s;
        }
        QMenu {
            background: %(surface)s;
            color: %(text)s;
            border: 1px solid %(border)s;
        }
        QMenu::item { padding: 6px 24px 6px 10px; }
        QMenu::item:selected { background: %(selection)s; }
        QToolTip {
            background: %(surface_alt)s;
            color: %(text)s;
            border: 1px solid %(border)s;
            padding: 4px;
        }
        QSplitter::handle { background: %(border)s; }
    """ % c)
    return name
