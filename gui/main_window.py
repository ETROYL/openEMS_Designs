"""Initial PySide6 application shell and YAML project editor."""
from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Top-level application window.

    This first milestone establishes the GUI shell and safe YAML editing.
    Optimizer, CAD, simulation and post-processing adapters are integrated
    in subsequent milestones; the existing command-line tools remain intact.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ETROYL | Electromagnetic Design & Optimization")
        self.resize(1180, 780)
        self.current_config: Path | None = None

        toolbar = QToolBar("Project")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QPushButton("Open YAML…")
        open_action.clicked.connect(self.open_config)
        toolbar.addWidget(open_action)

        save_action = QPushButton("Save YAML")
        save_action.clicked.connect(self.save_config)
        toolbar.addWidget(save_action)

        validate_action = QPushButton("Validate YAML")
        validate_action.clicked.connect(self.validate_config)
        toolbar.addWidget(validate_action)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "Open a design YAML file or start entering a configuration here…"
        )
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self._editor_page(), "Project & YAML")

        self.tabs.addTab(
            self._placeholder_page(
                "Optimization",
                "Strategy selection and optimization controls will be connected "
                "to the ETROYL discrete optimizer, Optuna and SciPy adapters."
            ),
            "Optimization",
        )
        self.tabs.addTab(
            self._placeholder_page(
                "CAD Preview",
                "The existing view_cad geometry builder will be integrated here. "
                "The preview will use the currently loaded YAML configuration."
            ),
            "CAD Preview",
        )
        self.tabs.addTab(
            self._placeholder_page(
                "Simulation",
                "Standalone openEMS simulation controls, frequency settings, "
                "mesh settings and execution-resource selection will be added here."
            ),
            "Simulation",
        )
        self.tabs.addTab(
            self._placeholder_page(
                "Results",
                "S-parameter, impedance, VSWR, far-field and optimization-history "
                "visualizations will be integrated here."
            ),
            "Results",
        )

        self.statusBar().showMessage("Ready — open a YAML design to begin")

    def _editor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.path_label = QLabel("No design loaded")
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.path_label)
        layout.addWidget(self.editor)
        return page

    @staticmethod
    def _placeholder_page(title: str, description: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        message = QLabel(description)
        message.setWordWrap(True)
        message.setStyleSheet("color: palette(mid);")
        layout.addWidget(heading)
        layout.addWidget(message)
        layout.addStretch(1)
        return page

    def open_config(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open electromagnetic design",
            str(self.current_config.parent if self.current_config else Path.cwd()),
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if not filename:
            return

        path = Path(filename)
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        self.current_config = path
        self.editor.setPlainText(content)
        self.path_label.setText(str(path))
        self.statusBar().showMessage(f"Loaded {path.name}")

    def save_config(self) -> None:
        if self.current_config is None:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save electromagnetic design",
                str(Path.cwd() / "design.yaml"),
                "YAML files (*.yaml *.yml)",
            )
            if not filename:
                return
            self.current_config = Path(filename)

        try:
            self.current_config.write_text(
                self.editor.toPlainText(), encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        self.path_label.setText(str(self.current_config))
        self.statusBar().showMessage(f"Saved {self.current_config.name}")

    def validate_config(self) -> None:
        try:
            parsed = yaml.safe_load(self.editor.toPlainText())
            if not isinstance(parsed, dict):
                raise ValueError("The YAML root must be a mapping/object.")
        except (yaml.YAMLError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            self.statusBar().showMessage("YAML validation failed")
            return

        self.statusBar().showMessage("YAML syntax and root structure are valid")
        QMessageBox.information(
            self,
            "YAML validation",
            "The YAML is syntactically valid and its root is a mapping. "
            "Design-schema validation will be added in a later milestone.",
        )
