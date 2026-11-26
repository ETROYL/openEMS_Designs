"""PySide6 application shell with YAML editing, CAD preview and FDTD execution."""
from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QProcess
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSpinBox, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    """Desktop interface; engineering workflows are connected incrementally."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ETROYL | Electromagnetic Design & Optimization")
        self.resize(1180, 780)

        # Keep the brand asset package-relative so launch location does not matter.
        logo_path = Path(__file__).resolve().with_name("logo-mono.webp")
        self._logo_pixmap = QPixmap(str(logo_path))
        if not self._logo_pixmap.isNull():
            self.setWindowIcon(QIcon(self._logo_pixmap))
        self.current_config: Path | None = None
        self._preview_config_path: Path | None = None
        self._simulation_config_path: Path | None = None
        self.last_simulation_dir: Path | None = None
        self.last_simulation_config: Path | None = None
        self._simulation_stdout_buffer = ""
        self._optimization_stdout_buffer = ""
        self._optimization_steps: list[int] = []
        self._optimization_objectives: list[float] = []
        self._optimization_best_steps: list[int] = []
        self._optimization_best_values: list[float] = []
        self._optimization_run_dir = None
        self._optimization_stop_file = None
        self._optimization_pause_file = None
        self._optimization_metadata_path = None
        self._optimization_pause_requested = False
        self._optimization_was_completed = False
        self.last_best_design_path = None
        self._optimization_stop_requested = False
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
        self._optimization_process = QProcess(self)
        self._optimization_process.readyReadStandardOutput.connect(self._optimization_stdout)
        self._optimization_process.readyReadStandardError.connect(self._optimization_stderr)
        self._optimization_process.finished.connect(self._optimization_finished)
        self._optimization_process.errorOccurred.connect(self._optimization_error)

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

        # Branded application header above the existing workflow tabs.
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("etroylHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 24, 12)
        header_layout.setSpacing(18)

        if not self._logo_pixmap.isNull():
            logo_label = QLabel()
            logo_label.setObjectName("etroylLogo")
            logo_label.setPixmap(self._logo_pixmap.scaled(
                190, 54,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            logo_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            header_layout.addWidget(logo_label)

        brand_text = QWidget()
        brand_text_layout = QVBoxLayout(brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(2)
        product_title = QLabel("Electromagnetic Design & Optimization")
        product_title.setStyleSheet("font-size: 17px; font-weight: 650;")
        brand_caption = QLabel("SCIENCE  |  ENGINEERING  |  DEPLOYMENT")
        brand_caption.setStyleSheet("font-size: 9px; letter-spacing: 1.4px; color: palette(mid);")
        brand_text_layout.addWidget(product_title)
        brand_text_layout.addWidget(brand_caption)
        header_layout.addWidget(brand_text)
        header_layout.addStretch(1)
        header.setStyleSheet(
            "#etroylHeader { border-bottom: 1px solid palette(midlight); }"
        )

        central_layout.addWidget(header)

        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Open a design YAML file or enter a configuration…")
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.tabs.addTab(self._editor_page(), "Project & YAML")
        self.tabs.addTab(self._optimization_page(), "Optimization")
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

    def _optimization_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        heading = QLabel("Design optimization")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        description = QLabel(
            "Optimize the current YAML without modifying the source file. Each launch "
            "creates a timestamped run inside <design>_opt. ETROYL Discrete is available now; "
            "Optuna and SciPy will be connected next."
        )
        description.setWordWrap(True)
        settings = QFormLayout()
        self.optimization_strategy = QComboBox()
        self.optimization_strategy.addItems([
            "ETROYL Discrete (Sobol + GP + Expected Improvement)",
            "Optuna (TPE) — coming next",
            "SciPy (SHGO + Nelder–Mead) — coming next",
        ])
        self.optimization_strategy.currentIndexChanged.connect(self._optimization_strategy_changed)
        self.optimization_objective = QComboBox()
        self.optimization_objective.addItem("Maximize matched bandwidth", "maximize_bandwidth_in_band")
        self.optimization_objective.addItem("Minimize worst-case S11", "minimize_max_s11_in_band")
        self.optimization_objective.addItem("Minimize S11 magnitude", "minimize_s11_in_band")
        self.optimization_objective.addItem("Maximize impedance smoothness", "maximize_impedance_smoothness")
        self.optimization_objective.addItem("Combined: S11 + impedance smoothness", "multi_objective_combined")
        self.optimization_objective.setCurrentIndex(4)
        self.optimization_objective.currentIndexChanged.connect(self._optimization_objective_changed)
        self.optimization_weight_s11 = QDoubleSpinBox()
        self.optimization_weight_s11.setDecimals(6)
        self.optimization_weight_s11.setRange(0.0, 1000000.0)
        self.optimization_weight_s11.setSingleStep(0.1)
        self.optimization_weight_s11.setValue(1.0)
        self.optimization_weight_smoothness = QDoubleSpinBox()
        self.optimization_weight_smoothness.setDecimals(6)
        self.optimization_weight_smoothness.setRange(0.0, 1000000.0)
        self.optimization_weight_smoothness.setSingleStep(0.001)
        self.optimization_weight_smoothness.setValue(0.0053)
        self.optimization_budget = QSpinBox()
        self.optimization_budget.setRange(1, 100000)
        self.optimization_budget.setValue(30)
        self.optimization_initial_samples = QSpinBox()
        self.optimization_initial_samples.setRange(1, 100000)
        self.optimization_initial_samples.setValue(8)
        self.optimization_pool_size = QSpinBox()
        self.optimization_pool_size.setRange(1, 10000000)
        self.optimization_pool_size.setValue(4096)
        self.optimization_seed = QSpinBox()
        self.optimization_seed.setRange(-1, 2147483647)
        self.optimization_seed.setValue(42)
        settings.addRow("Strategy", self.optimization_strategy)
        settings.addRow("Objective function", self.optimization_objective)
        settings.addRow("Combined weight — S11", self.optimization_weight_s11)
        settings.addRow("Combined weight — impedance smoothness", self.optimization_weight_smoothness)
        settings.addRow("New FDTD evaluation budget", self.optimization_budget)
        settings.addRow("Initial Sobol samples", self.optimization_initial_samples)
        settings.addRow("Candidate pool size", self.optimization_pool_size)
        settings.addRow("Random seed (-1 = random)", self.optimization_seed)
        controls = QHBoxLayout()
        self.start_optimization_button = QPushButton("Start optimization")
        self.start_optimization_button.clicked.connect(self.start_optimization)
        self.pause_optimization_button = QPushButton("Pause")
        self.pause_optimization_button.setEnabled(False)
        self.pause_optimization_button.clicked.connect(self.pause_optimization)
        self.resume_optimization_button = QPushButton("Resume")
        self.resume_optimization_button.setEnabled(False)
        self.resume_optimization_button.clicked.connect(self.resume_optimization)
        self.stop_optimization_button = QPushButton("Stop")
        self.stop_optimization_button.setEnabled(False)
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.load_best_design_button = QPushButton("Load best design into YAML editor")
        self.load_best_design_button.setEnabled(False)
        self.load_best_design_button.clicked.connect(self.load_best_design)
        controls.addWidget(self.start_optimization_button)
        controls.addWidget(self.pause_optimization_button)
        controls.addWidget(self.resume_optimization_button)
        controls.addWidget(self.stop_optimization_button)
        controls.addWidget(self.load_best_design_button)
        self.optimization_status = QLabel("No optimization has been run yet.")
        self._set_optimization_status("No optimization has been run yet.", "neutral")
        self.optimization_progress = QProgressBar()
        self.optimization_progress.setRange(0, 100)
        self.optimization_progress.setValue(0)
        self.optimization_progress_label = QLabel("Evaluations: 0 / 0")
        self.optimization_best_label = QLabel("Best objective: —")
        self.optimization_output_label = QLabel("Output: <design>_opt/run_<timestamp>")
        self.optimization_output_label.setWordWrap(True)
        self.optimization_log = QPlainTextEdit()
        self.optimization_log.setReadOnly(True)
        self.optimization_log.setPlaceholderText("Optimization progress and evaluation output will appear here…")
        self.optimization_log.setMinimumWidth(420)

        # Embedded live convergence chart, displayed beside the textual console.
        self.optimization_figure = Figure(figsize=(5.0, 3.8), tight_layout=True)
        self.optimization_canvas = FigureCanvas(self.optimization_figure)
        self.optimization_canvas.setMinimumWidth(340)
        self.optimization_axes = self.optimization_figure.add_subplot(111)
        self._reset_optimization_plot()

        output_panel = QWidget()
        output_layout = QHBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(12)
        output_layout.addWidget(self.optimization_log, 3)
        output_layout.addWidget(self.optimization_canvas, 2)

        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addLayout(settings)
        layout.addLayout(controls)
        layout.addWidget(self.optimization_status)
        layout.addWidget(self.optimization_progress)
        layout.addWidget(self.optimization_progress_label)
        layout.addWidget(self.optimization_best_label)
        layout.addWidget(self.optimization_output_label)
        layout.addWidget(output_panel, 1)
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

    def _reset_optimization_plot(self) -> None:
        self._optimization_steps = []
        self._optimization_objectives = []
        self._optimization_best_steps = []
        self._optimization_best_values = []
        self.optimization_axes.clear()
        self.optimization_axes.set_title("Live optimization convergence")
        self.optimization_axes.set_xlabel("Optimization evaluation")
        self.optimization_axes.set_ylabel("Objective cost (lower is better)")
        self.optimization_axes.grid(True, alpha=0.25)
        self.optimization_axes.text(
            0.5, 0.5, "Waiting for the first completed evaluation",
            transform=self.optimization_axes.transAxes,
            ha="center", va="center", color="gray",
        )
        self.optimization_canvas.draw_idle()

    def _set_optimization_history(self, history: list[dict]) -> None:
        self._reset_optimization_plot()
        for item in history:
            self._append_optimization_plot_point(
                int(item.get("step", 0)),
                item.get("status", "failed"),
                item.get("objective"),
                item.get("best_objective"),
            )

    def _append_optimization_plot_point(
        self, step: int, status: str, objective, best_objective
    ) -> None:
        try:
            value = float(objective)
            if status == "ok" and math.isfinite(value):
                self._optimization_steps.append(step)
                self._optimization_objectives.append(value)
        except (TypeError, ValueError):
            pass
        try:
            best = float(best_objective)
            if math.isfinite(best):
                self._optimization_best_steps.append(step)
                self._optimization_best_values.append(best)
        except (TypeError, ValueError):
            pass

        self.optimization_axes.clear()
        self.optimization_axes.set_title("Live optimization convergence")
        self.optimization_axes.set_xlabel("Optimization evaluation")
        self.optimization_axes.set_ylabel("Objective cost (lower is better)")
        self.optimization_axes.grid(True, alpha=0.25)
        if self._optimization_steps:
            self.optimization_axes.plot(
                self._optimization_steps, self._optimization_objectives,
                marker="o", markersize=4, linewidth=1.1,
                label="Objective per evaluation",
            )
        if self._optimization_best_steps:
            self.optimization_axes.plot(
                self._optimization_best_steps, self._optimization_best_values,
                marker=".", linewidth=2.0, label="Best so far",
            )
        if self._optimization_steps or self._optimization_best_steps:
            self.optimization_axes.legend(loc="best", fontsize=8)
        else:
            self.optimization_axes.text(
                0.5, 0.5, "Waiting for the first completed evaluation",
                transform=self.optimization_axes.transAxes,
                ha="center", va="center", color="gray",
            )
        self.optimization_canvas.draw_idle()

    def _optimization_strategy_changed(self, index: int) -> None:
        enabled = index == 0
        self.start_optimization_button.setEnabled(
            enabled and self._optimization_process.state() == QProcess.ProcessState.NotRunning
        )
        if not enabled:
            self._set_optimization_status(
                "This strategy is not connected yet. ETROYL Discrete is currently available.",
                "neutral",
            )

    def _optimization_objective_changed(self, _index: int) -> None:
        combined = self.optimization_objective.currentData() == "multi_objective_combined"
        self.optimization_weight_s11.setEnabled(combined)
        self.optimization_weight_smoothness.setEnabled(combined)

    def start_optimization(self) -> None:
        if self._optimization_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Optimization", "An optimization is already running.")
            return
        if self.optimization_strategy.currentIndex() != 0:
            QMessageBox.information(self, "Strategy not available", "ETROYL Discrete is the first integrated strategy.")
            return
        if self.current_config is None:
            QMessageBox.warning(
                self, "Design file required",
                "Open a YAML design file first. The optimization folder is created beside that file.",
            )
            return
        if not self._current_config_is_valid():
            return
        if self._simulation_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Optimization", "Stop the standalone FDTD simulation first.")
            return
        if self._postprocess_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Optimization", "Wait for post-processing to finish first.")
            return

        try:
            from optimizer import DiscreteParameterSpace
            config = yaml.safe_load(self.editor.toPlainText()) or {}
            space = DiscreteParameterSpace.from_config(config)
            if space.dimension == 0:
                raise ValueError("No active discrete optimization variables were found in the YAML.")
            if self.optimization_initial_samples.value() > self.optimization_budget.value():
                raise ValueError("Initial Sobol samples cannot exceed the evaluation budget.")
            objective_name = str(self.optimization_objective.currentData())
            weight_s11 = self.optimization_weight_s11.value()
            weight_smoothness = self.optimization_weight_smoothness.value()
            if objective_name == "multi_objective_combined" and weight_s11 == 0 and weight_smoothness == 0:
                raise ValueError("At least one combined-objective weight must be greater than zero.")
        except Exception as exc:
            QMessageBox.warning(self, "Optimization configuration", str(exc))
            return

        root_dir = self.current_config.parent / (self.current_config.stem + "_opt")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root_dir / ("run_" + timestamp)
        suffix = 1
        while run_dir.exists():
            run_dir = root_dir / ("run_" + timestamp + "_" + str(suffix).zfill(2))
            suffix += 1
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            snapshot = run_dir / "config.yaml"
            snapshot.write_text(self.editor.toPlainText(), encoding="utf-8")
            metadata_path = run_dir / "run_settings.json"
            metadata = {
                "budget": self.optimization_budget.value(),
                "initial_samples": self.optimization_initial_samples.value(),
                "pool_size": self.optimization_pool_size.value(),
                "seed": self.optimization_seed.value(),
                "objective": objective_name,
                "weight_s11": weight_s11,
                "weight_smoothness": weight_smoothness,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Optimization output", str(exc))
            return

        self._optimization_run_dir = run_dir
        self._optimization_config_path = snapshot
        self._optimization_metadata_path = metadata_path
        self._optimization_stop_file = run_dir / "STOP_REQUESTED"
        self._optimization_pause_file = run_dir / "PAUSE_REQUESTED"
        self._optimization_stop_file.unlink(missing_ok=True)
        self._optimization_pause_file.unlink(missing_ok=True)
        self.last_best_design_path = None
        self._optimization_stop_requested = False
        self._optimization_pause_requested = False
        self._optimization_was_completed = False
        self._optimization_stdout_buffer = ""
        self.optimization_log.clear()
        self._reset_optimization_plot()
        self.optimization_progress.setRange(0, self.optimization_budget.value())
        self.optimization_progress.setValue(0)
        self.optimization_progress_label.setText("Evaluations: 0 / " + str(self.optimization_budget.value()))
        self.optimization_best_label.setText("Best objective: —")
        self.optimization_output_label.setText("Output: " + str(run_dir))
        self.load_best_design_button.setEnabled(False)
        self.resume_optimization_button.setEnabled(False)
        self.optimization_log.appendPlainText("Objective: " + objective_name)
        if objective_name == "multi_objective_combined":
            self.optimization_log.appendPlainText(
                f"Combined weights: S11={weight_s11:g}, impedance smoothness={weight_smoothness:g}"
            )
        self._launch_optimization_process()

    def _launch_optimization_process(self) -> None:
        if not self._optimization_metadata_path or not self._optimization_config_path:
            QMessageBox.warning(self, "Optimization", "The saved optimization run settings are missing.")
            return
        try:
            metadata = json.loads(self._optimization_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Optimization", "Could not read run settings: " + str(exc))
            return

        self._optimization_stdout_buffer = ""
        self.start_optimization_button.setEnabled(False)
        self.pause_optimization_button.setEnabled(True)
        self.resume_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(True)
        self._set_optimization_status("OPTIMIZATION RUNNING", "running")
        self.optimization_log.appendPlainText("Launching / resuming ETROYL discrete optimizer…")
        self.statusBar().showMessage("Optimization running…")
        self._optimization_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self._optimization_process.start(
            sys.executable,
            [
                "-m", "gui.optimization_runner",
                "--config", str(self._optimization_config_path),
                "--run-dir", str(self._optimization_run_dir),
                "--budget", str(metadata["budget"]),
                "--initial-samples", str(metadata["initial_samples"]),
                "--pool-size", str(metadata["pool_size"]),
                "--seed", str(metadata["seed"]),
                "--objective", str(metadata["objective"]),
                "--weight-s11", str(metadata["weight_s11"]),
                "--weight-smoothness", str(metadata["weight_smoothness"]),
                "--stop-file", str(self._optimization_stop_file),
                "--pause-file", str(self._optimization_pause_file),
            ],
        )

    def pause_optimization(self) -> None:
        if self._optimization_process.state() == QProcess.ProcessState.NotRunning:
            return
        try:
            self._optimization_pause_file.write_text("pause requested by GUI", encoding="utf-8")
        except OSError as exc:
            self.optimization_log.appendPlainText("Could not write pause request: " + str(exc))
            return
        self._optimization_pause_requested = True
        self.pause_optimization_button.setEnabled(False)
        self._set_optimization_status(
            "PAUSE REQUESTED — waiting for the current FDTD evaluation to finish",
            "running",
        )
        self.optimization_log.appendPlainText(
            "Pause requested. The current FDTD evaluation will finish, then the run will be saved for resumption."
        )

    def resume_optimization(self) -> None:
        if self._optimization_process.state() != QProcess.ProcessState.NotRunning:
            return
        if not self._optimization_run_dir or not self._optimization_pause_file:
            QMessageBox.warning(self, "Resume optimization", "There is no paused optimization run to resume.")
            return
        if self._optimization_stop_file and self._optimization_stop_file.exists():
            QMessageBox.information(self, "Resume optimization", "This run was stopped, not paused. Start a new run instead.")
            return
        if not self._optimization_pause_file.exists():
            QMessageBox.information(self, "Resume optimization", "The run is not in a paused state.")
            return
        self._optimization_pause_file.unlink(missing_ok=True)
        self._optimization_pause_requested = False
        self._launch_optimization_process()

    def stop_optimization(self) -> None:
        if self._optimization_process.state() == QProcess.ProcessState.NotRunning:
            return
        try:
            self._optimization_stop_file.write_text("stop requested by GUI", encoding="utf-8")
        except OSError as exc:
            self.optimization_log.appendPlainText("Could not write stop request: " + str(exc))
            return
        self._optimization_stop_requested = True
        self.pause_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        self._set_optimization_status(
            "STOP REQUESTED — waiting for the current FDTD evaluation to finish",
            "running",
        )
        self.optimization_log.appendPlainText(
            "Stop requested. The current FDTD evaluation will finish; this run will not be resumable from this GUI session."
        )

    def load_best_design(self) -> None:
        if not self.last_best_design_path or not self.last_best_design_path.is_file():
            QMessageBox.warning(self, "Best design", "The saved best-design YAML could not be found.")
            return
        try:
            content = self.last_best_design_path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Best design", str(exc))
            return
        self.editor.setPlainText(content)
        self.path_label.setText(
            "Best design loaded (source file preserved): " + str(self.last_best_design_path)
        )
        self._update_simulation_settings()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage("Best optimized design loaded into the YAML editor")

    def _optimization_stdout(self) -> None:
        data = bytes(self._optimization_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_optimization_output(data)

    def _consume_optimization_output(self, data: str, flush: bool = False) -> None:
        self._optimization_stdout_buffer += data
        lines = self._optimization_stdout_buffer.split("\n")
        self._optimization_stdout_buffer = "" if flush else lines.pop()
        if flush and lines and lines[-1] == "":
            lines.pop()
        for raw_line in lines:
            line = raw_line.rstrip("\r")
            if not line:
                continue
            if line.startswith("__OPTIMIZATION_HISTORY__="):
                try:
                    history = json.loads(line.split("=", 1)[1])
                    self._set_optimization_history(history)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self.optimization_log.appendPlainText("Could not restore convergence history: " + str(exc))
                continue
            if line.startswith("__OPTIMIZATION_PROGRESS__="):
                try:
                    payload = json.loads(line.split("=", 1)[1])
                    completed = int(payload["completed"])
                    budget = int(payload["budget"])
                    self._append_optimization_plot_point(
                        int(payload.get("step", completed)),
                        str(payload.get("status", "ok")),
                        payload.get("objective"),
                        payload.get("best_objective"),
                    )
                    self.optimization_progress.setRange(0, max(1, budget))
                    self.optimization_progress.setValue(min(completed, budget))
                    self.optimization_progress_label.setText(
                        "Evaluations: " + str(completed) + " / " + str(budget)
                    )
                    best = payload.get("best_objective")
                    if best is not None:
                        self.optimization_best_label.setText(
                            "Best objective: " + format(float(best), ".10g")
                            + "    |    Successful: " + str(payload.get("successful", 0))
                            + "    Failed: " + str(payload.get("failed", 0))
                        )
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self.optimization_log.appendPlainText("Could not parse progress update: " + str(exc))
                continue
            self.optimization_log.appendPlainText(line)
            if line == "OPTIMIZATION_PAUSED=1":
                self._optimization_pause_requested = True
            elif line == "OPTIMIZATION_STOPPED=1":
                self._optimization_stop_requested = True
            elif line == "OPTIMIZATION_COMPLETED=1":
                self._optimization_was_completed = True
            elif line.startswith("BEST_DESIGN_PATH="):
                self.last_best_design_path = Path(line.split("=", 1)[1].strip())

    def _optimization_stderr(self) -> None:
        data = bytes(self._optimization_process.readAllStandardError()).decode(errors="replace")
        if data:
            self.optimization_log.appendPlainText(data.rstrip())

    def _optimization_finished(self, exit_code: int, _exit_status) -> None:
        remaining = bytes(self._optimization_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_optimization_output(remaining, flush=True)
        self.start_optimization_button.setEnabled(self.optimization_strategy.currentIndex() == 0)
        self.pause_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        if self.last_best_design_path and self.last_best_design_path.is_file():
            self.load_best_design_button.setEnabled(True)
        if self._optimization_pause_requested:
            self.start_optimization_button.setEnabled(False)
            self.resume_optimization_button.setEnabled(True)
            self._set_optimization_status("OPTIMIZATION PAUSED — READY TO RESUME", "running")
            self.statusBar().showMessage("Optimization paused; run history preserved")
        elif self._optimization_stop_requested:
            self.resume_optimization_button.setEnabled(False)
            self._set_optimization_status("OPTIMIZATION STOPPED BY USER", "neutral")
            self.statusBar().showMessage("Optimization stopped")
        elif self._optimization_was_completed and self.last_best_design_path and self.last_best_design_path.is_file():
            self.resume_optimization_button.setEnabled(False)
            self._set_optimization_status("OPTIMIZATION COMPLETED SUCCESSFULLY", "success")
            self.statusBar().showMessage("Optimization completed")
        else:
            self.resume_optimization_button.setEnabled(False)
            self._set_optimization_status("OPTIMIZATION FAILED — PROCESS EXIT CODE " + str(exit_code), "failure")
            self.statusBar().showMessage("Optimization failed")
            self.optimization_log.appendPlainText("Optimization process exited with code " + str(exit_code) + ".")

    def _optimization_error(self, _error) -> None:
        self.start_optimization_button.setEnabled(self.optimization_strategy.currentIndex() == 0)
        self.pause_optimization_button.setEnabled(False)
        self.resume_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        self._set_optimization_status("OPTIMIZATION FAILED TO START", "failure")
        self.optimization_log.appendPlainText(
            "Could not start the optimizer. Check the Python environment and repository dependencies."
        )
        self.statusBar().showMessage("Optimization failed to start")

    def _set_optimization_status(self, message: str, state: str) -> None:
        styles = {
            "neutral": "background:#e8edf2; color:#344054;",
            "running": "background:#fff4d6; color:#805b00;",
            "success": "background:#d9f7e5; color:#146c36;",
            "failure": "background:#fde2e2; color:#a61b1b;",
        }
        self.optimization_status.setText(message)
        self.optimization_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.optimization_status.setStyleSheet(
            "QLabel { " + styles.get(state, styles["neutral"])
            + " font-weight: 700; padding: 10px; border-radius: 6px; }"
        )

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
