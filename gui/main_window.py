"""PySide6 application shell with YAML editing and CAD preview."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    """Desktop interface; engineering workflows are connected incrementally."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ETROYL | Electromagnetic Design & Optimization")
        self.resize(1180, 780)
        self.current_config: Path | None = None
        self._preview_config_path: Path | None = None
        self._cad_process = QProcess(self)
        self._cad_process.readyReadStandardOutput.connect(self._cad_stdout)
        self._cad_process.readyReadStandardError.connect(self._cad_stderr)
        self._cad_process.finished.connect(self._cad_finished)
        self._cad_process.errorOccurred.connect(self._cad_error)

        toolbar = QToolBar("Project")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for label, callback in (
            ("Open YAML…", self.open_config),
            ("Save YAML", self.save_config),
            ("Validate YAML", self.validate_config),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Open a design YAML file or enter a configuration…")
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self._editor_page(), "Project & YAML")
        self.tabs.addTab(self._placeholder_page(
            "Optimization",
            "Strategy selection and controls for the ETROYL discrete optimizer, Optuna and SciPy will be integrated next."
        ), "Optimization")
        self.tabs.addTab(self._cad_page(), "CAD Preview")
        self.tabs.addTab(self._placeholder_page(
            "Simulation",
            "Standalone openEMS simulation controls, frequency settings, mesh settings and execution resources."
        ), "Simulation")
        self.tabs.addTab(self._placeholder_page(
            "Results",
            "S-parameters, impedance, VSWR, far-field and optimization-history visualizations."
        ), "Results")
        self.statusBar().showMessage("Ready — open a YAML design to begin")

    def _editor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.path_label = QLabel("No design loaded")
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        layout.addWidget(self.editor)
        return page

    def _cad_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("3D geometry preview")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        description = QLabel(
            "Build the CSXCAD geometry from the current YAML editor contents and open it in AppCSXCAD. "
            "Unsaved editor changes are included."
        )
        description.setWordWrap(True)
        self.preview_button = QPushButton("Generate and open 3D CAD preview")
        self.preview_button.clicked.connect(self.preview_cad)
        self.preview_log = QPlainTextEdit()
        self.preview_log.setReadOnly(True)
        self.preview_log.setPlaceholderText("CAD preview output will appear here…")
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(self.preview_button)
        layout.addWidget(self.preview_log, 1)
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
            self, "Open electromagnetic design",
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
                self, "Save electromagnetic design",
                str(Path.cwd() / "design.yaml"), "YAML files (*.yaml *.yml)",
            )
            if not filename:
                return
            self.current_config = Path(filename)
        try:
            self.current_config.write_text(self.editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.path_label.setText(str(self.current_config))
        self.statusBar().showMessage(f"Saved {self.current_config.name}")

    def validate_config(self) -> bool:
        try:
            parsed = yaml.safe_load(self.editor.toPlainText())
            if not isinstance(parsed, dict):
                raise ValueError("The YAML root must be a mapping/object.")
        except (yaml.YAMLError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            self.statusBar().showMessage("YAML validation failed")
            return False
        self.statusBar().showMessage("YAML syntax and root structure are valid")
        QMessageBox.information(
            self, "YAML validation",
            "The YAML is syntactically valid and its root is a mapping. "
            "Detailed electromagnetic design-schema validation will follow.",
        )
        return True

    def preview_cad(self) -> None:
        if self._cad_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "CAD preview", "A CAD preview is already running.")
            return
        if not self.validate_config_silently():
            return

        # Use the editor contents, not a possibly stale on-disk YAML file.
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="etroyl_gui_preview_",
            encoding="utf-8", delete=False,
        )
        try:
            handle.write(self.editor.toPlainText())
            handle.close()
            self._preview_config_path = Path(handle.name)
        except Exception:
            handle.close()
            raise

        self.preview_log.clear()
        self.preview_log.appendPlainText("Preparing CSXCAD geometry…")
        self.preview_button.setEnabled(False)
        self.statusBar().showMessage("Generating CAD preview…")
        self._cad_process.start(
            sys.executable,
            ["-m", "gui.cad_preview", str(self._preview_config_path)],
        )

    def validate_config_silently(self) -> bool:
        try:
            parsed = yaml.safe_load(self.editor.toPlainText())
            if not isinstance(parsed, dict):
                raise ValueError("The YAML root must be a mapping/object.")
        except (yaml.YAMLError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            return False
        return True

    def _cad_stdout(self) -> None:
        data = bytes(self._cad_process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self.preview_log.appendPlainText(data.rstrip())

    def _cad_stderr(self) -> None:
        data = bytes(self._cad_process.readAllStandardError()).decode(errors="replace")
        if data:
            self.preview_log.appendPlainText(data.rstrip())

    def _cad_finished(self, exit_code: int, _exit_status) -> None:
        self.preview_button.setEnabled(True)
        if self._preview_config_path:
            self._preview_config_path.unlink(missing_ok=True)
            self._preview_config_path = None
        if exit_code == 0:
            self.statusBar().showMessage("CAD preview closed")
        else:
            self.statusBar().showMessage(f"CAD preview process exited with code {exit_code}")
            self.preview_log.appendPlainText(f"Process exited with code {exit_code}.")

    def _cad_error(self, _error) -> None:
        self.preview_button.setEnabled(True)
        self.preview_log.appendPlainText(
            "Could not start the CAD preview process. Check Python environment and openEMS/CSXCAD imports."
        )
        self.statusBar().showMessage("CAD preview failed to start")


import sys
