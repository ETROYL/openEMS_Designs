"""PySide6 application shell with YAML editing, CAD preview and FDTD execution."""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    """Desktop interface; engineering workflows are connected incrementally."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ETROYL | Electromagnetic Design & Optimization")
        self.resize(1180, 780)
        self.current_config: Path | None = None
        self._preview_config_path: Path | None = None
        self._simulation_config_path: Path | None = None
        self.last_simulation_dir: Path | None = None
        self.last_simulation_config: Path | None = None
        self._simulation_stdout_buffer = ""
        self._s11_original_pixmap = QPixmap()
        self._radiation_original_pixmap = QPixmap()
        self._postprocess_process = QProcess(self)
        self._postprocess_process.readyReadStandardOutput.connect(self._postprocess_stdout)
        self._postprocess_process.readyReadStandardError.connect(self._postprocess_stderr)
        self._postprocess_process.finished.connect(self._postprocess_finished)
        self._postprocess_process.errorOccurred.connect(self._postprocess_error)

        self._cad_process = QProcess(self)
        self._cad_process.readyReadStandardOutput.connect(self._cad_stdout)
        self._cad_process.readyReadStandardError.connect(self._cad_stderr)
        self._cad_process.finished.connect(self._cad_finished)
        self._cad_process.errorOccurred.connect(self._cad_error)

        self._simulation_process = QProcess(self)
        self._simulation_process.readyReadStandardOutput.connect(self._simulation_stdout)
        self._simulation_process.readyReadStandardError.connect(self._simulation_stderr)
        self._simulation_process.finished.connect(self._simulation_finished)
        self._simulation_process.errorOccurred.connect(self._simulation_error)

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
        self.tabs.addTab(self._simulation_page(), "Simulation")
        self.tabs.addTab(self._results_page(), "Results")
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

    def _simulation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("Standalone FDTD simulation")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        description = QLabel(
            "Run one openEMS simulation using the current YAML editor contents. "
            "The run uses the simulation settings in the YAML, PML_8 boundaries and the "
            "thread count configured by parallel.py. Each run is written to a new output folder."
        )
        description.setWordWrap(True)
        self.simulation_settings_label = QLabel(
            "Load a design to review its frequency range and solver settings."
        )
        self.simulation_settings_label.setWordWrap(True)
        self.simulation_settings_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.run_simulation_button = QPushButton("Run FDTD simulation")
        self.run_simulation_button.clicked.connect(self.start_simulation)
        self.stop_simulation_button = QPushButton("Stop simulation")
        self.stop_simulation_button.setEnabled(False)
        self.stop_simulation_button.clicked.connect(self.stop_simulation)
        self.simulation_status = QLabel("No simulation has been run yet.")
        self._set_simulation_status("No simulation has been run yet.", "neutral")
        self.simulation_log = QPlainTextEdit()
        self.simulation_log.setReadOnly(True)
        self.simulation_log.setPlaceholderText("FDTD execution output will appear here…")
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(self.simulation_settings_label)
        layout.addWidget(self.run_simulation_button)
        layout.addWidget(self.stop_simulation_button)
        layout.addWidget(self.simulation_status)
        layout.addWidget(self.simulation_log, 1)
        return page

    def _results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("Simulation results")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        description = QLabel(
            "Post-process the most recent successful FDTD run using the design snapshot "
            "saved with that simulation. The existing postprocess.py computes port response, "
            "impedance, VSWR and NF2FF radiation results."
        )
        description.setWordWrap(True)
        self.results_status = QLabel("Run an FDTD simulation first.")
        self.results_status.setWordWrap(True)
        self.run_postprocess_button = QPushButton("Run post-processing")
        self.run_postprocess_button.setEnabled(False)
        self.run_postprocess_button.clicked.connect(self.start_postprocessing)
        self.results_log = QPlainTextEdit()
        self.results_log.setReadOnly(True)
        self.results_log.setMaximumHeight(150)
        self.results_log.setPlaceholderText("Post-processing output will appear here…")

        self.s11_plot_label = QLabel("S-parameter / impedance plot will appear here.")
        self.s11_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.s11_plot_label.setMinimumWidth(400)
        self.s11_plot_label.setStyleSheet("background: palette(alternate-base); border-radius: 6px;")
        self.radiation_plot_label = QLabel("Radiation / far-field plot will appear here.")
        self.radiation_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.radiation_plot_label.setMinimumWidth(400)
        self.radiation_plot_label.setStyleSheet("background: palette(alternate-base); border-radius: 6px;")

        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)
        plots_layout.setContentsMargins(18, 18, 18, 18)
        plots_layout.setSpacing(22)
        plots_layout.addWidget(self.s11_plot_label, 1)
        plots_layout.addWidget(self.radiation_plot_label, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(plots_widget)

        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(self.results_status)
        layout.addWidget(self.run_postprocess_button)
        layout.addWidget(self.results_log)
        layout.addWidget(scroll, 1)
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
        self._update_simulation_settings()
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
        self._update_simulation_settings()
        self.statusBar().showMessage("YAML syntax and root structure are valid")
        QMessageBox.information(
            self, "YAML validation",
            "The YAML is syntactically valid and its root is a mapping. "
            "Detailed electromagnetic design-schema validation will follow.",
        )
        return True

    def _current_config_is_valid(self) -> bool:
        try:
            parsed = yaml.safe_load(self.editor.toPlainText())
            if not isinstance(parsed, dict):
                raise ValueError("The YAML root must be a mapping/object.")
            return True
        except (yaml.YAMLError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid YAML", str(exc))
            return False

    def _write_temporary_config(self, prefix: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=prefix,
            encoding="utf-8", delete=False,
        )
        try:
            handle.write(self.editor.toPlainText())
            handle.close()
            return Path(handle.name)
        except Exception:
            handle.close()
            raise

    def preview_cad(self) -> None:
        if self._cad_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "CAD preview", "A CAD preview is already running.")
            return
        if not self._current_config_is_valid():
            return

        self._preview_config_path = self._write_temporary_config("etroyl_gui_preview_")
        self.preview_log.clear()
        self.preview_log.appendPlainText("Preparing CSXCAD geometry…")
        self.preview_button.setEnabled(False)
        self.statusBar().showMessage("Generating CAD preview…")
        self._cad_process.start(
            sys.executable,
            ["-m", "gui.cad_preview", str(self._preview_config_path)],
        )

    def _simulation_settings(self) -> str:
        cfg = yaml.safe_load(self.editor.toPlainText()) or {}
        sim = cfg.get("simulation", {})
        return (
            f"Frequency: {sim.get('fstart', '—')}–{sim.get('fstop', '—')} GHz    |    "
            f"Frequency points: {sim.get('n_pts', 301)}\n"
            f"End criteria: {sim.get('end_criteria', 1e-4)}    |    "
            f"Maximum timesteps: {sim.get('max_timesteps', 500000)}"
        )

    def _update_simulation_settings(self) -> None:
        try:
            self.simulation_settings_label.setText(self._simulation_settings())
        except (yaml.YAMLError, AttributeError, TypeError):
            self.simulation_settings_label.setText("Simulation settings could not be read from the current YAML.")

    def start_simulation(self) -> None:
        if self._simulation_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "FDTD simulation", "A simulation is already running.")
            return
        if not self._current_config_is_valid():
            return
        try:
            cfg = yaml.safe_load(self.editor.toPlainText())
            sim = cfg.get("simulation", {})
            fstart, fstop = float(sim["fstart"]), float(sim["fstop"])
            if fstart <= 0 or fstop <= fstart:
                raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range")
            if int(sim.get("max_timesteps", 500000)) <= 0:
                raise ValueError("simulation.max_timesteps must be positive")
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid simulation settings", str(exc))
            return

        default_parent = self.current_config.parent if self.current_config else Path.cwd()
        selected_parent = QFileDialog.getExistingDirectory(
            self, "Choose parent folder for simulation output", str(default_parent)
        )
        if not selected_parent:
            return

        if self._postprocess_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "FDTD simulation", "Wait for post-processing to finish first.")
            return
        self._simulation_config_path = self._write_temporary_config("etroyl_gui_simulation_")
        self._simulation_stdout_buffer = ""
        self.last_simulation_dir = None
        self.last_simulation_config = None
        self.run_postprocess_button.setEnabled(False)
        self.results_status.setText("Waiting for the FDTD simulation to complete successfully.")
        self._s11_original_pixmap = QPixmap()
        self._radiation_original_pixmap = QPixmap()
        self.s11_plot_label.setText("S-parameter / impedance plot will appear here.")
        self.s11_plot_label.setPixmap(QPixmap())
        self.radiation_plot_label.setText("Radiation / far-field plot will appear here.")
        self.radiation_plot_label.setPixmap(QPixmap())
        self.simulation_log.clear()
        self.simulation_log.appendPlainText("Launching standalone FDTD simulation…")
        self._set_simulation_status("SIMULATION RUNNING", "running")
        self.run_simulation_button.setEnabled(False)
        self.stop_simulation_button.setEnabled(True)
        self.statusBar().showMessage("FDTD simulation running…")
        self._simulation_process.start(
            sys.executable,
            ["-m", "gui.simulation_runner", str(self._simulation_config_path), selected_parent,
             self.current_config.stem if self.current_config else "design"],
        )

    def stop_simulation(self) -> None:
        if self._simulation_process.state() != QProcess.ProcessState.NotRunning:
            self.simulation_log.appendPlainText("Stop requested by user.")
            self._simulation_process.kill()

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

    def _simulation_stdout(self) -> None:
        data = bytes(self._simulation_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_simulation_stdout(data)

    def _consume_simulation_stdout(self, data: str, flush: bool = False) -> None:
        self._simulation_stdout_buffer += data
        lines = self._simulation_stdout_buffer.split("\n")
        self._simulation_stdout_buffer = "" if flush else lines.pop()
        if flush and lines and lines[-1] == "":
            lines.pop()
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            self.simulation_log.appendPlainText(line)
            if line.startswith("SIMULATION_OUTPUT_DIR="):
                self.last_simulation_dir = Path(line.split("=", 1)[1].strip())
            elif line.startswith("SIMULATION_CONFIG="):
                self.last_simulation_config = Path(line.split("=", 1)[1].strip())

    def _simulation_stderr(self) -> None:
        data = bytes(self._simulation_process.readAllStandardError()).decode(errors="replace")
        if data:
            self.simulation_log.appendPlainText(data.rstrip())

    def _simulation_finished(self, exit_code: int, _exit_status) -> None:
        remaining = bytes(self._simulation_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_simulation_stdout(remaining, flush=True)
        self.run_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
        if self._simulation_config_path:
            self._simulation_config_path.unlink(missing_ok=True)
            self._simulation_config_path = None
        if exit_code == 0:
            if self.last_simulation_dir and self.last_simulation_config and self.last_simulation_dir.is_dir() and self.last_simulation_config.is_file():
                self.run_postprocess_button.setEnabled(True)
                self.results_status.setText(f"Ready to post-process: {self.last_simulation_dir}")
                self._set_simulation_status("SIMULATION COMPLETED SUCCESSFULLY", "success")
                self.statusBar().showMessage("FDTD simulation completed")
            else:
                self.results_status.setText("Simulation completed, but its output snapshot could not be located.")
                self._set_simulation_status("SIMULATION FINISHED, BUT OUTPUT SNAPSHOT IS MISSING", "failure")
                self.statusBar().showMessage("FDTD simulation output could not be located")
        else:
            self._set_simulation_status(f"SIMULATION FAILED — PROCESS EXIT CODE {exit_code}", "failure")
            self.statusBar().showMessage(f"FDTD simulation exited with code {exit_code}")
            self.simulation_log.appendPlainText(f"Simulation process exited with code {exit_code}.")

    def start_postprocessing(self) -> None:
        if self._postprocess_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Post-processing", "Post-processing is already running.")
            return
        if not self.last_simulation_dir or not self.last_simulation_config:
            QMessageBox.warning(self, "Post-processing", "Run a successful FDTD simulation first.")
            return
        if not self.last_simulation_dir.is_dir() or not self.last_simulation_config.is_file():
            QMessageBox.warning(self, "Post-processing", "The saved simulation output or design snapshot is missing.")
            return

        self.results_log.clear()
        self.results_log.appendPlainText("Starting post-processing of the latest simulation…")
        self.run_postprocess_button.setEnabled(False)
        self.run_simulation_button.setEnabled(False)
        self.results_status.setText(f"Processing: {self.last_simulation_dir}")
        self._postprocess_process.start(
            sys.executable,
            [
                "-m", "postprocess",
                "--config", str(self.last_simulation_config),
                "--sim-dir", str(self.last_simulation_dir),
                "--no-show",
            ],
        )

    def _postprocess_stdout(self) -> None:
        data = bytes(self._postprocess_process.readAllStandardOutput()).decode(errors="replace")
        if data:
            self.results_log.appendPlainText(data.rstrip())

    def _postprocess_stderr(self) -> None:
        data = bytes(self._postprocess_process.readAllStandardError()).decode(errors="replace")
        if data:
            self.results_log.appendPlainText(data.rstrip())

    def _postprocess_finished(self, exit_code: int, _exit_status) -> None:
        self.run_simulation_button.setEnabled(True)
        self.run_postprocess_button.setEnabled(
            bool(self.last_simulation_dir and self.last_simulation_config)
        )
        if exit_code == 0 and self.last_simulation_dir:
            self._load_result_plot(self.s11_plot_label, self.last_simulation_dir / "s11_plot.png")
            self._load_result_plot(self.radiation_plot_label, self.last_simulation_dir / "radiation_plot.png")
            self.results_status.setText(f"Post-processing complete. Results saved in: {self.last_simulation_dir}")
            self.statusBar().showMessage("Post-processing completed")
        else:
            self.results_status.setText("Post-processing failed. See the log for details.")
            self.results_log.appendPlainText(f"Post-processing process exited with code {exit_code}.")
            self.statusBar().showMessage("Post-processing failed")

    def _load_result_plot(self, label: QLabel, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText(f"Could not load plot: {image_path}")
            return
        if label is self.s11_plot_label:
            self._s11_original_pixmap = pixmap
        else:
            self._radiation_original_pixmap = pixmap
        label.setText("")
        self._refresh_result_plots()

    def _refresh_result_plots(self) -> None:
        for label, original in (
            (self.s11_plot_label, self._s11_original_pixmap),
            (self.radiation_plot_label, self._radiation_original_pixmap),
        ):
            if original.isNull():
                continue
            available_width = max(1, label.width() - 24)
            available_height = max(1, self.height() - 260)
            label.setPixmap(original.scaled(
                available_width, available_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "s11_plot_label"):
            self._refresh_result_plots()

    def _set_simulation_status(self, message: str, state: str) -> None:
        styles = {
            "neutral": "background:#e8edf2; color:#344054;",
            "running": "background:#fff4d6; color:#805b00;",
            "success": "background:#d9f7e5; color:#146c36;",
            "failure": "background:#fde2e2; color:#a61b1b;",
        }
        self.simulation_status.setText(message)
        self.simulation_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.simulation_status.setStyleSheet(
            f"QLabel {{ {styles.get(state, styles['neutral'])} "
            "font-weight: 700; padding: 10px; border-radius: 6px; }}"
        )

    def _postprocess_error(self, _error) -> None:
        self.run_simulation_button.setEnabled(True)
        self.run_postprocess_button.setEnabled(
            bool(self.last_simulation_dir and self.last_simulation_config)
        )
        self.results_status.setText("Could not start post-processing. Check Python and Matplotlib dependencies.")
        self.results_log.appendPlainText(
            "Could not start the post-processing process. Check the Python environment and openEMS imports."
        )
        self.statusBar().showMessage("Post-processing failed to start")

    def _simulation_error(self, _error) -> None:
        self.run_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
        self._set_simulation_status("SIMULATION FAILED TO START", "failure")
        self.simulation_log.appendPlainText(
            "Could not start the simulation process. Check the Python environment and openEMS imports."
        )
        self.statusBar().showMessage("FDTD simulation failed to start")
