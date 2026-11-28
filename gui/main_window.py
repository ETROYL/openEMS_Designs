"""PySide6 application shell with YAML editing, CAD preview and FDTD execution."""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import sys
import tempfile
import sqlite3
from datetime import datetime
from time import monotonic
from pathlib import Path

import yaml

from gui.cpu_resources import detect_cpu_resources, validate_allocation
from gui.optimization_compatibility import find_incompatible_settings
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSpinBox, QSizePolicy, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)


class MainWindow(QMainWindow):
    """Desktop interface; engineering workflows are connected incrementally."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ETROYL | Electromagnetic Design & Optimization")

        # Start inside the usable desktop area, while retaining normal window
        # resizing/maximizing. Content is constrained inside its tab pages so
        # status updates cannot force the outer window to grow.
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            window_width = min(1180, max(320, available.width() - 24))
            window_height = min(780, max(320, available.height() - 24))
            self.setMinimumSize(min(640, window_width), min(480, window_height))
            self.resize(window_width, window_height)
            self.move(
                available.center().x() - window_width // 2,
                available.center().y() - window_height // 2,
            )
        else:
            self.resize(1180, 780)

        # Keep the brand asset package-relative so launch location does not matter.
        logo_path = Path(__file__).resolve().with_name("logo-mono.webp")
        self._logo_pixmap = QPixmap(str(logo_path))
        if not self._logo_pixmap.isNull():
            self.setWindowIcon(QIcon(self._logo_pixmap))
        self.current_config: Path | None = None
        self._last_valid_simulation_settings_text: str | None = None
        self._preview_config_path: Path | None = None
        self._simulation_config_path: Path | None = None
        self.last_simulation_dir: Path | None = None
        self.last_simulation_config: Path | None = None
        self.last_simulation_multiband = False
        self._simulation_stdout_buffer = ""
        self._simulation_started_at: float | None = None
        self._simulation_elapsed_seconds = 0.0
        self._simulation_interval_total = 0
        self._simulation_interval_completed = 0
        self._simulation_interval_runtimes: list[float] = []
        self._simulation_workers = 1
        self._simulation_progress_timer = QTimer(self)
        self._simulation_progress_timer.setInterval(1000)
        self._simulation_progress_timer.timeout.connect(self._update_simulation_elapsed)
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
        self._optimization_kill_requested = False
        self._s11_original_pixmap = QPixmap()
        self._radiation_original_pixmap = QPixmap()
        self._result_plot_catalog: list[dict[str, str]] = []
        self._result_plot_directory: Path | None = None
        self._pattern_update_only = False
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
        # Keep every YAML-derived view synchronized while the project editor changes.
        # Optimization's manual evaluation override remains intentionally independent.
        self.editor.textChanged.connect(self._update_simulation_settings)
        self._update_simulation_settings()
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        heading = QLabel("Design optimization")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        description = QLabel(
            "Optimize the current YAML without modifying the source file. Each launch "
            "creates a timestamped run inside <design>_opt. ETROYL Discrete and Optuna TPE "
            "share the same discrete parameter space, FDTD workers and evaluation history."
        )
        description.setWordWrap(True)
        # The page heading already names this section; omit a second group-box title.
        settings_box = QGroupBox()
        settings_layout = QHBoxLayout(settings_box)
        settings_layout.setContentsMargins(8, 6, 8, 6)
        settings_layout.setSpacing(18)
        left_settings = QFormLayout()
        right_settings = QFormLayout()
        for form in (left_settings, right_settings):
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(4)
            form.setContentsMargins(0, 0, 0, 0)

        self.optimization_strategy = QComboBox()
        self.optimization_strategy.addItems([
            "ETROYL Discrete (Sobol + GP + Expected Improvement)",
            "Optuna (TPE)",
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
        self.optimization_initial_label = QLabel("Initial Sobol samples")
        self.optimization_pool_size = QSpinBox()
        self.optimization_pool_size.setRange(1, 10000000)
        self.optimization_pool_size.setValue(4096)
        self.optimization_seed = QSpinBox()
        self.optimization_seed.setRange(-1, 2147483647)
        self.optimization_seed.setValue(42)

        # Six editable controls on the left.
        left_settings.addRow("Strategy", self.optimization_strategy)
        left_settings.addRow("Objective function", self.optimization_objective)
        left_settings.addRow("Combined weight — S11", self.optimization_weight_s11)
        left_settings.addRow("Combined weight — impedance smoothness", self.optimization_weight_smoothness)
        left_settings.addRow("New / additional FDTD budget", self.optimization_budget)
        left_settings.addRow(self.optimization_initial_label, self.optimization_initial_samples)

        self.cpu_resources = detect_cpu_resources()
        self.optimization_resource_mode = QComboBox()
        self.optimization_resource_mode.addItem("Default (4 workers × 2 threads)", "default")
        self.optimization_resource_mode.addItem("Manual", "manual")
        self.optimization_resource_mode.currentIndexChanged.connect(self._optimization_resource_mode_changed)
        self.optimization_workers = QSpinBox()
        self.optimization_workers.setRange(1, int(self.cpu_resources["physical_cores"]))
        self.optimization_workers.setValue(min(4, int(self.cpu_resources["physical_cores"])))
        self.optimization_threads = QSpinBox()
        # Benchmarking on the target workstation showed that more than four
        # threads per FDTD worker reduces throughput; keep the GUI within the
        # tested operating range.
        self.optimization_threads.setRange(1, min(4, int(self.cpu_resources["logical_threads"])))
        self.optimization_threads.setValue(min(2, int(self.cpu_resources["logical_threads"])))
        self.optimization_workers.valueChanged.connect(self._optimization_resource_mode_changed)
        self.optimization_threads.valueChanged.connect(self._optimization_resource_mode_changed)
        self.optimization_cpu_capacity = QLabel(
            f"Available to this process: {self.cpu_resources['physical_cores']} physical cores / "
            f"{self.cpu_resources['logical_threads']} logical CPUs "
            f"({max(0, int(self.cpu_resources['logical_threads']) - 1)} usable; 1 reserved)"
        )
        self.optimization_allocation_status = QLabel("Default allocation: 4 workers × 2 threads per worker")
        # Keep resource diagnostics visible and let their text wrap naturally
        # across the full width of the right-hand settings column.
        self.optimization_cpu_capacity.setWordWrap(True)
        self.optimization_allocation_status.setWordWrap(True)
        for diagnostic_label in (
            self.optimization_cpu_capacity,
            self.optimization_allocation_status,
        ):
            diagnostic_label.setMinimumHeight(28)
            diagnostic_label.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )

        # Five editable controls on the right, followed by full-width diagnostics.
        right_settings.addRow("Candidate pool size", self.optimization_pool_size)
        right_settings.addRow("Random seed (-1 = random)", self.optimization_seed)
        right_settings.addRow("Hardware allocation", self.optimization_resource_mode)
        right_settings.addRow("Workers", self.optimization_workers)
        right_settings.addRow("Threads per worker", self.optimization_threads)
        right_settings.addRow(self.optimization_cpu_capacity)
        right_settings.addRow(self.optimization_allocation_status)

        settings_layout.addLayout(left_settings, 1)
        settings_layout.addLayout(right_settings, 1)
        self._optimization_resource_mode_changed()

        # Candidate evaluation can either inherit the YAML simulation profile
        # or use a GUI-only override saved with this optimization run.
        evaluation_box = QGroupBox("FDTD parameters used during optimization")
        evaluation_layout = QVBoxLayout(evaluation_box)
        self.optimization_multiband_checkbox = QCheckBox(
            "Use multi-band simulation"
        )
        self.optimization_multiband_checkbox.setChecked(True)
        self.optimization_multiband_checkbox.setToolTip(
            "Prepare one fixed interval plan from the YAML default geometry, "
            "then reuse it for every candidate and resumed evaluation."
        )
        evaluation_layout.addWidget(self.optimization_multiband_checkbox)
        self.use_simulation_parameters_radio = QRadioButton(
            "Use simulation parameters for Optimization"
        )
        self.define_optimization_parameters_radio = QRadioButton(
            "Define optimization-only parameters"
        )
        self.use_simulation_parameters_radio.setChecked(True)
        self.use_simulation_parameters_radio.toggled.connect(
            self._optimization_evaluation_mode_changed
        )
        evaluation_layout.addWidget(self.use_simulation_parameters_radio)
        evaluation_layout.addWidget(self.define_optimization_parameters_radio)

        evaluation_form = QFormLayout()
        self.optimization_fstart = QDoubleSpinBox()
        self.optimization_fstart.setDecimals(6)
        self.optimization_fstart.setRange(0.000001, 1.0e6)
        self.optimization_fstart.setSuffix(" GHz")
        self.optimization_fstop = QDoubleSpinBox()
        self.optimization_fstop.setDecimals(6)
        self.optimization_fstop.setRange(0.000001, 1.0e6)
        self.optimization_fstop.setSuffix(" GHz")
        self.optimization_n_pts = QSpinBox()
        self.optimization_n_pts.setRange(2, 1000000)
        self.optimization_end_criteria_db = QDoubleSpinBox()
        self.optimization_end_criteria_db.setDecimals(2)
        self.optimization_end_criteria_db.setRange(-200.0, -0.01)
        self.optimization_end_criteria_db.setSuffix(" dB")
        self.optimization_max_timesteps = QSpinBox()
        self.optimization_max_timesteps.setRange(1, 100000000)
        evaluation_form.addRow("Start frequency", self.optimization_fstart)
        evaluation_form.addRow("Stop frequency", self.optimization_fstop)
        evaluation_form.addRow("Number of frequency points", self.optimization_n_pts)
        evaluation_form.addRow("Energy-decay criterion", self.optimization_end_criteria_db)
        evaluation_form.addRow("Maximum timesteps", self.optimization_max_timesteps)
        evaluation_layout.addLayout(evaluation_form)
        self._optimization_evaluation_controls = [
            self.optimization_fstart, self.optimization_fstop,
            self.optimization_n_pts, self.optimization_end_criteria_db,
            self.optimization_max_timesteps,
        ]
        self._update_optimization_evaluation_defaults()
        self._optimization_evaluation_mode_changed()
        self.editor.textChanged.connect(self._update_optimization_evaluation_defaults)
        self.optimization_multiband_checkbox.toggled.connect(self._refresh_continue_eligibility)
        self.use_simulation_parameters_radio.toggled.connect(self._refresh_continue_eligibility)
        for control in self._optimization_evaluation_controls:
            control.valueChanged.connect(self._refresh_continue_eligibility)
        self.editor.textChanged.connect(self._refresh_continue_eligibility)
        self._refresh_continue_eligibility()


        controls = QHBoxLayout()
        self.start_optimization_button = QPushButton("Start new optimization")
        self.start_optimization_button.clicked.connect(self.start_optimization)
        self.continue_optimization_button = QPushButton("Continue optimization")
        self.continue_optimization_button.setEnabled(False)
        self.continue_optimization_button.clicked.connect(self.continue_optimization)
        self.pause_optimization_button = QPushButton("Pause")
        self.pause_optimization_button.setEnabled(False)
        self.pause_optimization_button.clicked.connect(self.pause_optimization)
        self.resume_optimization_button = QPushButton("Resume")
        self.resume_optimization_button.setEnabled(False)
        self.resume_optimization_button.clicked.connect(self.resume_optimization)
        self.stop_optimization_button = QPushButton("Stop")
        self.stop_optimization_button.setEnabled(False)
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.kill_optimization_button = QPushButton("Kill")
        self.kill_optimization_button.setEnabled(False)
        self.kill_optimization_button.clicked.connect(self.kill_optimization)
        self.load_best_design_button = QPushButton("Load best design into YAML editor")
        self.load_best_design_button.setEnabled(False)
        self.load_best_design_button.clicked.connect(self.load_best_design)
        controls.addWidget(self.start_optimization_button)
        controls.addWidget(self.continue_optimization_button)
        controls.addWidget(self.pause_optimization_button)
        controls.addWidget(self.resume_optimization_button)
        controls.addWidget(self.stop_optimization_button)
        controls.addWidget(self.kill_optimization_button)
        controls.addWidget(self.load_best_design_button)
        self.optimization_status = QLabel("No optimization has been run yet.")
        self._set_optimization_status("No optimization has been run yet.", "neutral")
        self.optimization_progress = QProgressBar()
        self.optimization_progress.setRange(0, 100)
        self.optimization_progress.setValue(0)
        self.optimization_progress_label = QLabel("Evaluations: 0 / 0")
        self.optimization_best_label = QLabel("Best objective: —")
        self.optimization_output_label = QLabel("Output: <design>_opt/run_<timestamp>")
        self.optimization_output_label.setWordWrap(False)
        self.optimization_output_label.setToolTip(self.optimization_output_label.text())
        self.optimization_log = QPlainTextEdit()
        self.optimization_log.setReadOnly(True)
        self.optimization_log.setPlaceholderText("Optimization progress and evaluation output will appear here…")
        self.optimization_log.setMinimumWidth(240)
        self.optimization_log.setMinimumHeight(150)
        self.optimization_log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )

        # Embedded live convergence chart, displayed beside the textual console.
        self.optimization_figure = Figure(figsize=(5.0, 3.0), tight_layout=True)
        self.optimization_canvas = FigureCanvas(self.optimization_figure)
        self.optimization_canvas.setMinimumWidth(240)
        self.optimization_canvas.setMinimumHeight(150)
        self.optimization_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.optimization_axes = self.optimization_figure.add_subplot(111)
        self._reset_optimization_plot()

        output_panel = QWidget()
        output_layout = QHBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)
        output_layout.addWidget(self.optimization_log, 3)
        output_layout.addWidget(self.optimization_canvas, 2)

        # Put run metadata on one line rather than spending three rows on it.
        run_info = QHBoxLayout()
        run_info.setSpacing(18)
        self.optimization_progress_label.setMinimumWidth(125)
        self.optimization_best_label.setMinimumWidth(170)
        self.optimization_output_label.setMinimumWidth(240)
        self.optimization_output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        run_info.addWidget(self.optimization_progress_label)
        run_info.addWidget(self.optimization_best_label)
        run_info.addWidget(self.optimization_output_label, 1)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)
        settings_row.addWidget(settings_box, 3)
        settings_row.addWidget(evaluation_box, 2)

        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addLayout(settings_row)
        layout.addLayout(controls)
        layout.addWidget(self.optimization_status)
        layout.addWidget(self.optimization_progress)
        layout.addLayout(run_info)
        layout.addWidget(output_panel, 1)

        # The optimization controls are intentionally allowed to scroll within
        # the tab on shorter displays instead of forcing the main window larger.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)
        return scroll

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
            "Run a standalone FDTD simulation using the current YAML editor contents. "
            "Choose standard single-band execution or the accurate multi-band workflow. "
            "Each run is written to a new output folder."
        )
        description.setWordWrap(True)
        self.simulation_multiband_checkbox = QCheckBox("Use multi-band simulation")
        self.simulation_multiband_checkbox.setChecked(True)
        self.simulation_multiband_checkbox.setToolTip(
            "Prepare intervals for this design, then run the accurate profile "
            "(-40 dB, up to 1,000,000 timesteps, 1,024 output points)."
        )
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
        simulation_actions = QHBoxLayout()
        simulation_actions.addWidget(self.run_simulation_button)
        simulation_actions.addWidget(self.stop_simulation_button)
        simulation_actions.addStretch(1)
        self.simulation_status = QLabel("No simulation has been run yet.")
        self._set_simulation_status("No simulation has been run yet.", "neutral")
        self.simulation_progress = QProgressBar()
        self.simulation_progress.setRange(0, 1)
        self.simulation_progress.setValue(0)
        self.simulation_progress.setFormat("Waiting for simulation")
        self.simulation_progress_label = QLabel("Intervals: —")
        self.simulation_elapsed_label = QLabel("Elapsed: 00:00")
        self.simulation_eta_label = QLabel("Remaining: —")
        self.simulation_stage_label = QLabel("Stage: idle")
        self.simulation_stage_label.setWordWrap(True)
        simulation_progress_info = QHBoxLayout()
        simulation_progress_info.addWidget(self.simulation_progress_label)
        simulation_progress_info.addWidget(self.simulation_elapsed_label)
        simulation_progress_info.addWidget(self.simulation_eta_label)
        simulation_progress_info.addStretch(1)
        self.simulation_log = QPlainTextEdit()
        self.simulation_log.setReadOnly(True)
        self.simulation_log.setPlaceholderText("FDTD execution output will appear here…")
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(self.simulation_multiband_checkbox)
        layout.addWidget(self.simulation_settings_label)
        layout.addLayout(simulation_actions)
        layout.addWidget(self.simulation_status)
        layout.addWidget(self.simulation_progress)
        layout.addLayout(simulation_progress_info)
        layout.addWidget(self.simulation_stage_label)
        layout.addWidget(self.simulation_log, 1)
        return page

    def _results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        heading = QLabel("Simulation results")
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.results_status = QLabel("Run an FDTD simulation first.")
        self.results_status.setWordWrap(True)
        self.results_status.setMaximumHeight(38)
        self.results_summary = QLabel("Performance metrics will appear after post-processing.")
        self.results_summary.setWordWrap(True)
        self.results_summary.setMaximumHeight(58)
        self.results_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        results_actions = QHBoxLayout()
        self.open_results_button = QPushButton("Open existing results folder…")
        self.open_results_button.setToolTip("Load a previously completed simulation output directory.")
        self.open_results_button.clicked.connect(self.open_existing_results)
        self.run_postprocess_button = QPushButton("Run post-processing")
        self.run_postprocess_button.setEnabled(False)
        self.run_postprocess_button.clicked.connect(self.start_postprocessing)
        results_actions.addWidget(self.open_results_button)
        results_actions.addWidget(self.run_postprocess_button)
        results_actions.addStretch(1)

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Plot:"))
        self.results_plot_combo = QComboBox()
        self.results_plot_combo.setMinimumWidth(260)
        self.results_plot_combo.addItem("Waiting for simulation results…", "")
        self.results_plot_combo.setEnabled(False)
        self.results_plot_combo.currentIndexChanged.connect(self._display_selected_result_plot)
        plot_controls.addWidget(self.results_plot_combo)
        plot_controls.addStretch(1)

        pattern_controls = QHBoxLayout()
        pattern_controls.addWidget(QLabel("Radiation pattern frequency:"))
        self.pattern_frequency_spin = QDoubleSpinBox()
        self.pattern_frequency_spin.setDecimals(6)
        self.pattern_frequency_spin.setRange(0.000001, 1.0e6)
        self.pattern_frequency_spin.setSingleStep(0.01)
        self.pattern_frequency_spin.setSuffix(" GHz")
        self.pattern_frequency_spin.setToolTip(
            "Choose any frequency covered by the recorded far-field simulation bands."
        )
        self.update_pattern_button = QPushButton("Update pattern")
        self.update_pattern_button.setEnabled(False)
        self.update_pattern_button.setToolTip(
            "Recalculate only the selected far-field pattern and refresh the 2×4 overview."
        )
        self.update_pattern_button.clicked.connect(self._request_radiation_pattern_update)
        pattern_controls.addWidget(self.pattern_frequency_spin)
        pattern_controls.addWidget(self.update_pattern_button)
        pattern_controls.addStretch(1)

        self.results_log = QPlainTextEdit()
        self.results_log.setReadOnly(True)
        self.results_log.setMaximumHeight(64)
        self.results_log.setPlaceholderText("Post-processing output…")

        self.s11_plot_label = QLabel("Integrated results overview will appear here.")
        self.s11_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.s11_plot_label.setMinimumSize(320, 260)
        self.s11_plot_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.s11_plot_label.setStyleSheet(
            "background: #171b22; border: 1px solid #343d4a; border-radius: 6px;"
        )
        # Retained as a compatibility target for older single-band result paths.
        self.radiation_plot_label = QLabel()
        self.radiation_plot_label.hide()

        layout.addWidget(heading)
        layout.addLayout(results_actions)
        layout.addWidget(self.results_status)
        layout.addWidget(self.results_summary)
        layout.addLayout(plot_controls)
        layout.addLayout(pattern_controls)
        layout.addWidget(self.results_log)
        layout.addWidget(self.s11_plot_label, 1)
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
        enabled = index in (0, 1)
        self.optimization_initial_label.setText(
            "Initial Sobol samples" if index == 0 else "TPE startup trials"
        )
        self.start_optimization_button.setEnabled(
            enabled and self._optimization_process.state() == QProcess.ProcessState.NotRunning
        )

    def _optimization_resource_mode_changed(self, _index: int = 0) -> None:
        manual = self.optimization_resource_mode.currentData() == "manual"
        self.optimization_workers.setEnabled(manual)
        self.optimization_threads.setEnabled(manual)
        if not manual:
            self.optimization_allocation_status.setText("Default allocation: 4 workers × 2 threads per worker")
            self.optimization_allocation_status.setStyleSheet("color: palette(text);")
            return
        try:
            validate_allocation(self.optimization_workers.value(), self.optimization_threads.value())
            total = self.optimization_workers.value() * self.optimization_threads.value()
            self.optimization_allocation_status.setText(
                f"Valid: {self.optimization_workers.value()} × {self.optimization_threads.value()} = "
                f"{total} logical CPUs"
            )
            self.optimization_allocation_status.setStyleSheet("color: #16803c;")
        except ValueError as exc:
            self.optimization_allocation_status.setText(str(exc))
            self.optimization_allocation_status.setStyleSheet("color: #b42318;")

    def _optimization_evaluation_mode_changed(self, _checked: bool = False) -> None:
        enabled = self.define_optimization_parameters_radio.isChecked()
        for control in self._optimization_evaluation_controls:
            control.setEnabled(enabled)

    def _update_optimization_evaluation_defaults(self) -> None:
        """Refresh override defaults only from a fully valid YAML simulation block."""
        if self.define_optimization_parameters_radio.isChecked():
            return
        try:
            from gui.simulation_settings import parse_simulation_settings

            settings = parse_simulation_settings(self.editor.toPlainText())
        except (yaml.YAMLError, AttributeError, TypeError, ValueError, OverflowError):
            # Keep every control at its last valid value while the editor is invalid.
            return

        self.optimization_fstart.setValue(
            0.1 if settings["fstart"] is None else settings["fstart"]
        )
        self.optimization_fstop.setValue(
            1.0 if settings["fstop"] is None else settings["fstop"]
        )
        self.optimization_n_pts.setValue(settings["n_pts"])
        self.optimization_end_criteria_db.setValue(settings["end_criteria_db"])
        self.optimization_max_timesteps.setValue(settings["max_timesteps"])

    def _optimization_evaluation_settings(self) -> dict | None:
        if self.use_simulation_parameters_radio.isChecked():
            return None
        settings = {
            "fstart": self.optimization_fstart.value(),
            "fstop": self.optimization_fstop.value(),
            "n_pts": self.optimization_n_pts.value(),
            "end_criteria_db": self.optimization_end_criteria_db.value(),
            "max_timesteps": self.optimization_max_timesteps.value(),
        }
        if settings["fstart"] <= 0 or settings["fstop"] <= settings["fstart"]:
            raise ValueError("Optimization frequency range must be positive and increasing.")
        return settings

    def _optimization_objective_changed(self, _index: int) -> None:
        combined = self.optimization_objective.currentData() == "multi_objective_combined"
        self.optimization_weight_s11.setEnabled(combined)
        self.optimization_weight_smoothness.setEnabled(combined)


    def _continue_compatibility_mismatches(self, metadata: dict) -> list[str]:
        """Compare current evaluation controls with the immutable run profile."""
        config = yaml.safe_load(self.editor.toPlainText()) or {}
        simulation = config.get("simulation") or {}
        current_override = (
            None if self.use_simulation_parameters_radio.isChecked()
            else self._optimization_evaluation_settings()
        )
        return find_incompatible_settings(
            metadata,
            current_simulation_settings=simulation,
            current_evaluation_settings=current_override,
            current_use_multiband=self.optimization_multiband_checkbox.isChecked(),
        )

    def _refresh_continue_eligibility(self, *_args) -> None:
        """Keep Continue disabled whenever current controls differ from the saved run."""
        button = getattr(self, "continue_optimization_button", None)
        if button is None:
            return
        reason = "Continue is available only for a run with stored evaluations."
        enabled = False
        if (
            self._optimization_process.state() == QProcess.ProcessState.NotRunning
            and self._has_stored_optimization_history()
            and self._optimization_metadata_path
            and not (self._optimization_pause_file and self._optimization_pause_file.exists())
        ):
            try:
                metadata = json.loads(self._optimization_metadata_path.read_text(encoding="utf-8"))
                mismatches = self._continue_compatibility_mismatches(metadata)
                if mismatches:
                    reason = (
                        "Continue is disabled because these evaluation settings differ "
                        "from the saved run: " + ", ".join(mismatches)
                        + ". Start a new optimization to change them."
                    )
                else:
                    enabled = True
                    reason = "Evaluation settings match the saved run."
            except (OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
                reason = "Continue is unavailable: " + str(exc)
        button.setEnabled(enabled)
        button.setToolTip(reason)

    def start_optimization(self) -> None:
        if self._optimization_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Optimization", "An optimization is already running.")
            return
        strategy_index = self.optimization_strategy.currentIndex()
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
            if (
                strategy_index == 0
                and self.optimization_initial_samples.value() > self.optimization_budget.value()
            ):
                raise ValueError("Initial Sobol samples cannot exceed the evaluation budget.")
            resource_mode = str(self.optimization_resource_mode.currentData())
            if resource_mode == "default":
                workers, threads_per_worker = 4, 2
                validate_allocation(workers, threads_per_worker)
            else:
                workers = self.optimization_workers.value()
                threads_per_worker = self.optimization_threads.value()
                validate_allocation(workers, threads_per_worker)
            objective_name = str(self.optimization_objective.currentData())
            weight_s11 = self.optimization_weight_s11.value()
            weight_smoothness = self.optimization_weight_smoothness.value()
            if objective_name == "multi_objective_combined" and weight_s11 == 0 and weight_smoothness == 0:
                raise ValueError("At least one combined-objective weight must be greater than zero.")
            evaluation_settings = self._optimization_evaluation_settings()
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
                "strategy": "etroyl_discrete" if strategy_index == 0 else "optuna_tpe",
                "budget": self.optimization_budget.value(),
                "initial_samples": self.optimization_initial_samples.value(),
                "pool_size": self.optimization_pool_size.value(),
                "seed": self.optimization_seed.value(),
                "objective": objective_name,
                "weight_s11": weight_s11,
                "weight_smoothness": weight_smoothness,
                "resource_mode": resource_mode,
                "workers": workers,
                "threads_per_worker": threads_per_worker,
                "simulation_settings": dict(config.get("simulation") or {}),
                "use_simulation_parameters": self.use_simulation_parameters_radio.isChecked(),
                "use_multiband": self.optimization_multiband_checkbox.isChecked(),
                "evaluation_settings": evaluation_settings,
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
        self._optimization_kill_requested = False
        self.last_best_design_path = None
        self._optimization_stop_requested = False
        self._optimization_pause_requested = False
        self._optimization_was_completed = False
        self._optimization_stdout_buffer = ""
        self.continue_optimization_button.setEnabled(False)
        self.optimization_log.clear()
        self._reset_optimization_plot()
        self.optimization_progress.setRange(0, self.optimization_budget.value())
        self.optimization_progress.setValue(0)
        self.optimization_progress_label.setText("Evaluations: 0 / " + str(self.optimization_budget.value()))
        self.optimization_best_label.setText("Best objective: —")
        self.optimization_output_label.setText("Output: " + str(run_dir))
        self.load_best_design_button.setEnabled(False)
        self.resume_optimization_button.setEnabled(False)
        self.optimization_log.appendPlainText(
            "Strategy: " + ("ETROYL Discrete" if strategy_index == 0 else "Optuna TPE")
        )
        self.optimization_log.appendPlainText("Objective: " + objective_name)
        self.optimization_log.appendPlainText(
            "Simulation engine: " + (
                "multi-band" if self.optimization_multiband_checkbox.isChecked()
                else "standard single-band"
            )
        )
        self.optimization_log.appendPlainText(
            f"Requested allocation: {workers} worker(s) × {threads_per_worker} thread(s) per worker"
        )
        self.optimization_log.appendPlainText(
            f"FDTD execution: {workers} process worker(s) × {threads_per_worker} thread(s) per worker."
        )
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
        self.kill_optimization_button.setEnabled(True)
        self._set_optimization_status("OPTIMIZATION RUNNING", "running")
        strategy = str(metadata.get("strategy", "etroyl_discrete"))
        strategy_label = "Optuna TPE" if strategy == "optuna_tpe" else "ETROYL Discrete"
        self.optimization_log.appendPlainText(f"Launching / resuming {strategy_label} optimizer…")
        self.statusBar().showMessage("Optimization running…")
        self._optimization_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        # Run in a dedicated session so spawned FDTD workers share a killable process group.
        setsid = shutil.which("setsid")
        program = setsid or sys.executable
        arguments = [
            sys.executable, "-m", "gui.optimization_runner",
            "--config", str(self._optimization_config_path),
            "--run-dir", str(self._optimization_run_dir),
            "--budget", str(metadata["budget"]),
            "--strategy", strategy,
            "--initial-samples", str(metadata["initial_samples"]),
            "--pool-size", str(metadata["pool_size"]),
            "--seed", str(metadata["seed"]),
            "--objective", str(metadata["objective"]),
            "--weight-s11", str(metadata["weight_s11"]),
            "--weight-smoothness", str(metadata["weight_smoothness"]),
            "--resource-mode", str(metadata.get("resource_mode", "default")),
            "--workers", str(metadata.get("workers", 4)),
            "--threads-per-worker", str(metadata.get("threads_per_worker", 2)),
            "--stop-file", str(self._optimization_stop_file),
            "--pause-file", str(self._optimization_pause_file),
        ]
        if setsid:
            self._optimization_process.start(program, arguments)
        else:
            self.optimization_log.appendPlainText("Warning: setsid unavailable; Kill may not terminate spawned workers on this platform.")
            self._optimization_process.start(program, arguments[1:])

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
            "PAUSE REQUESTED — waiting for active FDTD evaluations to finish",
            "running",
        )
        self.optimization_log.appendPlainText(
            "Pause requested. All FDTD evaluations already in progress will finish; no new candidates will be launched. The run will remain available to Resume."
        )

    def continue_optimization(self) -> None:
        """Continue the selected run with an additional evaluation budget."""
        if self._optimization_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Continue optimization", "An optimization is already running.")
            return
        if not self._optimization_run_dir or not self._optimization_metadata_path:
            QMessageBox.warning(self, "Continue optimization", "There is no existing optimization run to continue.")
            return
        if self._optimization_pause_file and self._optimization_pause_file.exists():
            QMessageBox.information(
                self, "Continue optimization",
                "This run is paused. Use Resume to finish its current budget before adding more evaluations.",
            )
            return
        if self._optimization_stop_file and self._optimization_stop_file.exists():
            # A user-stopped run can still be continued; remove the old stop marker below.
            pass
        try:
            metadata = json.loads(self._optimization_metadata_path.read_text(encoding="utf-8"))
            mismatches = self._continue_compatibility_mismatches(metadata)
            if mismatches:
                raise ValueError(
                    "The current evaluation settings do not match this run: "
                    + ", ".join(mismatches)
                    + ". Restore the original settings or start a new optimization."
                )
            database = self._optimization_run_dir / "evaluations.sqlite"
            if not database.is_file():
                raise FileNotFoundError("The run's evaluations.sqlite database is missing.")
            with sqlite3.connect(str(database)) as connection:
                completed = int(connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])
            additional_budget = int(self.optimization_budget.value())
            if additional_budget <= 0:
                raise ValueError("The additional evaluation budget must be greater than zero.")
            if completed <= 0:
                raise ValueError("This run has no stored evaluations. Start a new optimization instead.")
            target_budget = completed + additional_budget
            resource_mode = str(self.optimization_resource_mode.currentData())
            if resource_mode == "default":
                workers, threads_per_worker = 4, 2
            else:
                workers = int(self.optimization_workers.value())
                threads_per_worker = int(self.optimization_threads.value())
            validate_allocation(workers, threads_per_worker)
            metadata["budget"] = target_budget
            metadata["last_additional_budget"] = additional_budget
            metadata["resource_mode"] = resource_mode
            metadata["workers"] = workers
            metadata["threads_per_worker"] = threads_per_worker
            self._optimization_metadata_path.write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        except (OSError, sqlite3.Error, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Continue optimization", str(exc))
            return

        self._optimization_stop_file.unlink(missing_ok=True)
        self._optimization_pause_file.unlink(missing_ok=True)
        self._optimization_stop_requested = False
        self._optimization_pause_requested = False
        self._optimization_kill_requested = False
        self._optimization_was_completed = False
        self.continue_optimization_button.setEnabled(False)
        self.start_optimization_button.setEnabled(False)
        self.resume_optimization_button.setEnabled(False)
        self.optimization_progress.setRange(0, target_budget)
        self.optimization_progress.setValue(completed)
        self.optimization_progress_label.setText(
            f"Evaluations: {completed} / {target_budget}"
        )
        self.optimization_log.appendPlainText(
            f"\nContinuing existing run: {completed} stored evaluations + "
            f"{additional_budget} additional evaluations (target {target_budget})."
        )
        self.optimization_log.appendPlainText(
            f"New allocation: {workers} worker(s) × {threads_per_worker} thread(s) per worker."
        )
        self._launch_optimization_process()

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
            "STOP REQUESTED — waiting for active FDTD evaluations to finish",
            "running",
        )
        self.optimization_log.appendPlainText(
            "Stop requested. All FDTD evaluations already in progress will finish; no new candidates will be launched. Completed results are retained and can be continued later with an additional evaluation budget."
        )

    def kill_optimization(self) -> None:
        """Immediately terminate the runner and spawned FDTD workers."""
        if self._optimization_process.state() == QProcess.ProcessState.NotRunning:
            return
        answer = QMessageBox.question(
            self, "Kill optimization",
            "Terminate the optimization immediately, including all active FDTD workers? "
            "The current evaluations may be lost; completed SQLite history will be retained.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        pid = int(self._optimization_process.processId())
        if pid <= 0:
            return
        self._optimization_kill_requested = True
        self._optimization_pause_requested = False
        self._optimization_stop_requested = False
        self.pause_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        self.kill_optimization_button.setEnabled(False)
        self._set_optimization_status("KILL REQUESTED — TERMINATING RUNNER AND FDTD WORKERS", "failure")
        self.optimization_log.appendPlainText("Immediate termination requested. Active evaluations may be interrupted; stored SQLite results are preserved.")
        try:
            if shutil.which("setsid"):
                os.killpg(pid, signal.SIGKILL)
            else:
                self._optimization_process.kill()
        except ProcessLookupError:
            pass
        except OSError as exc:
            self.optimization_log.appendPlainText("Process-group termination failed: " + str(exc))
            self._optimization_process.kill()
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

    def _has_stored_optimization_history(self) -> bool:
        if not self._optimization_run_dir:
            return False
        database = self._optimization_run_dir / "evaluations.sqlite"
        if not database.is_file():
            return False
        try:
            with sqlite3.connect(str(database)) as connection:
                row = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()
            return bool(row and int(row[0]) > 0)
        except sqlite3.Error:
            return False

    def _optimization_finished(self, exit_code: int, _exit_status) -> None:
        remaining = bytes(self._optimization_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_optimization_output(remaining, flush=True)
        self.start_optimization_button.setEnabled(True)
        self.continue_optimization_button.setEnabled(False)
        self.pause_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        self.kill_optimization_button.setEnabled(False)
        if self.last_best_design_path and self.last_best_design_path.is_file():
            self.load_best_design_button.setEnabled(True)
        if self._optimization_kill_requested:
            self.resume_optimization_button.setEnabled(False)
            self._refresh_continue_eligibility()
            self.start_optimization_button.setEnabled(True)
            self._set_optimization_status("OPTIMIZATION KILLED — STORED HISTORY PRESERVED", "failure")
            self.statusBar().showMessage("Optimization killed; completed evaluations remain available")
            self.optimization_log.appendPlainText("Optimization process group terminated.")
        elif self._optimization_pause_requested:
            self.start_optimization_button.setEnabled(False)
            self.resume_optimization_button.setEnabled(True)
            self._set_optimization_status("OPTIMIZATION PAUSED — READY TO RESUME", "running")
            self.statusBar().showMessage("Optimization paused; run history preserved")
        elif self._optimization_stop_requested:
            self.resume_optimization_button.setEnabled(False)
            self._refresh_continue_eligibility()
            self._set_optimization_status("OPTIMIZATION STOPPED — HISTORY PRESERVED", "neutral")
            self.statusBar().showMessage("Optimization stopped; its history can be continued")
        elif self._optimization_was_completed and self.last_best_design_path and self.last_best_design_path.is_file():
            self.resume_optimization_button.setEnabled(False)
            self._refresh_continue_eligibility()
            self._set_optimization_status("OPTIMIZATION COMPLETED SUCCESSFULLY", "success")
            self.statusBar().showMessage("Optimization completed")
        else:
            self.resume_optimization_button.setEnabled(False)
            self._refresh_continue_eligibility()
            self._set_optimization_status("OPTIMIZATION FAILED — PROCESS EXIT CODE " + str(exit_code), "failure")
            self.statusBar().showMessage("Optimization failed")
            self.optimization_log.appendPlainText("Optimization process exited with code " + str(exit_code) + ".")

    def _optimization_error(self, _error) -> None:
        self.start_optimization_button.setEnabled(True)
        self._refresh_continue_eligibility()
        self.pause_optimization_button.setEnabled(False)
        self.resume_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(False)
        self.kill_optimization_button.setEnabled(False)
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
            from gui.simulation_settings import parse_simulation_settings

            parse_simulation_settings(self.editor.toPlainText())
        except (yaml.YAMLError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            QMessageBox.warning(self, "Invalid YAML or simulation settings", str(exc))
            self.statusBar().showMessage("YAML validation failed")
            return False
        self._update_simulation_settings()
        self.statusBar().showMessage("YAML syntax and simulation settings are valid")
        QMessageBox.information(
            self, "YAML validation",
            "The YAML syntax, root structure, and simulation settings are valid. "
            "Detailed electromagnetic design-schema validation will follow.",
        )
        return True

    def _current_config_is_valid(self) -> bool:
        try:
            from gui.simulation_settings import parse_simulation_settings

            parse_simulation_settings(self.editor.toPlainText())
            return True
        except (yaml.YAMLError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            QMessageBox.warning(self, "Invalid YAML or simulation settings", str(exc))
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
        from gui.simulation_settings import format_simulation_settings

        return format_simulation_settings(self.editor.toPlainText())

    def _update_simulation_settings(self) -> None:
        """Refresh the Simulation tab from the current valid YAML editor contents."""
        try:
            settings_text = self._simulation_settings()
        except (yaml.YAMLError, AttributeError, TypeError, ValueError, OverflowError):
            previous = self._last_valid_simulation_settings_text
            if previous:
                self.simulation_settings_label.setText(
                    "YAML is temporarily invalid — showing the last valid settings:\n"
                    + previous
                )
            else:
                self.simulation_settings_label.setText(
                    "YAML is not yet valid. Simulation settings will appear when it is corrected."
                )
            return

        self._last_valid_simulation_settings_text = settings_text
        self.simulation_settings_label.setText(settings_text)

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
        self._simulation_started_at = monotonic()
        self._simulation_elapsed_seconds = 0.0
        self._simulation_interval_total = 0
        self._simulation_interval_completed = 0
        self._simulation_interval_runtimes = []
        self._simulation_workers = 1
        self.simulation_progress.setRange(0, 0)
        self.simulation_progress.setFormat("Preparing simulation…")
        self.simulation_progress_label.setText("Intervals: planning")
        self.simulation_elapsed_label.setText("Elapsed: 00:00")
        self.simulation_eta_label.setText("Remaining: estimating…")
        self.simulation_stage_label.setText("Stage: preparing simulation")
        self._simulation_progress_timer.start()
        self.last_simulation_dir = None
        self.last_simulation_config = None
        self.last_simulation_multiband = self.simulation_multiband_checkbox.isChecked()
        self.run_postprocess_button.setEnabled(False)
        self.results_status.setText("Waiting for the FDTD simulation to complete successfully.")
        self.results_summary.setText("Performance metrics will appear after post-processing.")
        self._s11_original_pixmap = QPixmap()
        self._radiation_original_pixmap = QPixmap()
        self._set_result_plot_catalog([])
        self.s11_plot_label.setText("S11 magnitude (dB) will appear here.")
        self.s11_plot_label.setPixmap(QPixmap())
        self.simulation_log.clear()
        self.simulation_log.appendPlainText("Launching standalone FDTD simulation…")
        self._set_simulation_status("SIMULATION RUNNING", "running")
        self.run_simulation_button.setEnabled(False)
        self.stop_simulation_button.setEnabled(True)
        self.statusBar().showMessage("FDTD simulation running…")
        simulation_args = [
            "-m", "gui.simulation_runner", str(self._simulation_config_path), selected_parent,
            self.current_config.stem if self.current_config else "design",
        ]
        if self.last_simulation_multiband:
            simulation_args.append("--multiband")
        self._simulation_process.start(sys.executable, simulation_args)

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
            if line.startswith("__SIMULATION_PROGRESS__="):
                try:
                    payload = json.loads(line.split("=", 1)[1])
                    self._handle_simulation_progress(payload)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    self.simulation_log.appendPlainText("Could not parse simulation progress update: " + str(exc))
                continue
            self.simulation_log.appendPlainText(line)
            if line.startswith("Building CSXCAD"):
                self.simulation_stage_label.setText("Stage: building geometry and mesh")
                self.simulation_progress.setFormat("Building geometry and mesh…")
            elif line.startswith("Starting openEMS FDTD solver"):
                self.simulation_stage_label.setText(
                    "Stage: openEMS solver running (solver progress is not exposed reliably)"
                )
                self.simulation_progress.setFormat("FDTD solver running…")
                self.simulation_eta_label.setText("Remaining: unavailable for single-band runs")
            elif line.startswith("FDTD simulation completed successfully"):
                self.simulation_stage_label.setText("Stage: FDTD solver completed")
            if line.startswith("SIMULATION_OUTPUT_DIR="):
                self.last_simulation_dir = Path(line.split("=", 1)[1].strip())
            elif line.startswith("SIMULATION_CONFIG="):
                self.last_simulation_config = Path(line.split("=", 1)[1].strip())

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
            return "—"
        value = int(round(seconds))
        hours, remainder = divmod(value, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def _update_simulation_elapsed(self) -> None:
        if self._simulation_started_at is None:
            return
        self._simulation_elapsed_seconds = monotonic() - self._simulation_started_at
        self.simulation_elapsed_label.setText(
            "Elapsed: " + self._format_duration(self._simulation_elapsed_seconds)
        )

    def _handle_simulation_progress(self, payload: dict) -> None:
        phase = str(payload.get("phase", "running"))
        total = int(payload.get("total", self._simulation_interval_total or 0))
        completed = int(payload.get("completed", self._simulation_interval_completed))
        if total > 0:
            self._simulation_interval_total = total
            self._simulation_interval_completed = min(max(completed, 0), total)
            self._simulation_workers = max(1, int(payload.get("workers", self._simulation_workers)))
            self.simulation_progress.setRange(0, total)
            self.simulation_progress.setValue(self._simulation_interval_completed)
            self.simulation_progress.setFormat(
                f"Intervals: {self._simulation_interval_completed} / {total} (%p%)"
            )
            self.simulation_progress_label.setText(
                f"Intervals: {self._simulation_interval_completed} / {total}"
            )
        if payload.get("runtime_seconds") is not None:
            self._simulation_interval_runtimes.append(float(payload["runtime_seconds"]))
        if phase == "planning":
            self.simulation_stage_label.setText("Stage: preparing frequency intervals")
            self.simulation_progress.setRange(0, 0)
            self.simulation_progress.setFormat("Preparing frequency intervals…")
            self.simulation_eta_label.setText("Remaining: estimating…")
        elif phase == "intervals":
            interval = payload.get("interval")
            if interval is not None:
                low = payload.get("fstart_ghz")
                high = payload.get("fstop_ghz")
                detail = (f" — interval {interval}: {float(low):g}–{float(high):g} GHz"
                          if low is not None and high is not None else f" — interval {interval}")
                self.simulation_stage_label.setText("Stage: running FDTD intervals" + detail)
            else:
                self.simulation_stage_label.setText("Stage: running FDTD intervals")
            remaining = self._simulation_interval_total - self._simulation_interval_completed
            if len(self._simulation_interval_runtimes) >= 1 and remaining > 0:
                mean_runtime = sum(self._simulation_interval_runtimes) / len(self._simulation_interval_runtimes)
                eta = mean_runtime * remaining / self._simulation_workers
                self.simulation_eta_label.setText("Remaining: ~" + self._format_duration(eta))
            elif remaining == 0:
                self.simulation_eta_label.setText("Remaining: finalising…")
            else:
                self.simulation_eta_label.setText("Remaining: estimating…")
        elif phase == "reconstructing":
            self.simulation_stage_label.setText("Stage: reconstructing broadband response and saving results")
            self.simulation_progress.setFormat("FDTD intervals complete — finalising results…")
            self.simulation_eta_label.setText("Remaining: finalising…")
        elif phase == "complete":
            self.simulation_stage_label.setText("Stage: simulation and result preparation complete")
            if total > 0:
                self.simulation_progress.setRange(0, total)
                self.simulation_progress.setValue(total)
                self.simulation_progress.setFormat("Simulation complete — %p%")
            self.simulation_eta_label.setText("Remaining: 00:00")

    def _simulation_stderr(self) -> None:
        data = bytes(self._simulation_process.readAllStandardError()).decode(errors="replace")
        if data:
            self.simulation_log.appendPlainText(data.rstrip())

    def _simulation_finished(self, exit_code: int, _exit_status) -> None:
        remaining = bytes(self._simulation_process.readAllStandardOutput()).decode(errors="replace")
        self._consume_simulation_stdout(remaining, flush=True)
        self._simulation_progress_timer.stop()
        self._update_simulation_elapsed()
        self._simulation_started_at = None
        self.run_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
        if self._simulation_config_path:
            self._simulation_config_path.unlink(missing_ok=True)
            self._simulation_config_path = None
        if exit_code == 0:
            if self.last_simulation_dir and self.last_simulation_config and self.last_simulation_dir.is_dir() and self.last_simulation_config.is_file():
                if self.last_simulation_multiband:
                    response_csv = self.last_simulation_dir / "broadband_response.csv"
                    self.run_postprocess_button.setEnabled(response_csv.is_file())
                    self.run_postprocess_button.setToolTip(
                        "Calculate matching metrics and, when available, full far-field characteristics."
                    )
                    run_manifest = self.last_simulation_dir / "multiband_run.json"
                    far_field_available = False
                    if run_manifest.is_file():
                        try:
                            far_field_available = bool(json.loads(
                                run_manifest.read_text(encoding="utf-8")
                            ).get("far_field_recorded"))
                        except (OSError, ValueError, TypeError):
                            pass
                    initial_plot = self.last_simulation_dir / "s11_plot.png"
                    if initial_plot.is_file():
                        self._set_result_plot_catalog([{
                            "label": "|S11| magnitude (dB)",
                            "path": str(initial_plot),
                        }])
                    self.results_status.setText(
                        "Broadband response is ready. Run post-processing for matching metrics"
                        + (" and full far-field characteristics." if far_field_available
                           else ". Far-field data was not recorded for this run.")
                        + f" Numerical response data: {self.last_simulation_dir}"
                    )
                else:
                    self.run_postprocess_button.setEnabled(True)
                    self.run_postprocess_button.setToolTip("")
                    self.results_status.setText(f"Ready to post-process: {self.last_simulation_dir}")
                self._set_simulation_status("SIMULATION COMPLETED SUCCESSFULLY", "success")
                if self._simulation_interval_total == 0:
                    self.simulation_progress.setRange(0, 1)
                    self.simulation_progress.setValue(1)
                    self.simulation_progress.setFormat("Simulation complete")
                self.simulation_stage_label.setText("Stage: simulation completed successfully")
                self.simulation_eta_label.setText("Remaining: 00:00")
                self.statusBar().showMessage("FDTD simulation completed")
            else:
                self.results_status.setText("Simulation completed, but its output snapshot could not be located.")
                self._set_simulation_status("SIMULATION FINISHED, BUT OUTPUT SNAPSHOT IS MISSING", "failure")
                self.statusBar().showMessage("FDTD simulation output could not be located")
        else:
            self._set_simulation_status(f"SIMULATION FAILED — PROCESS EXIT CODE {exit_code}", "failure")
            self.simulation_stage_label.setText("Stage: simulation failed or was stopped")
            self.simulation_eta_label.setText("Remaining: —")
            self.statusBar().showMessage(f"FDTD simulation exited with code {exit_code}")
            self.simulation_log.appendPlainText(f"Simulation process exited with code {exit_code}.")

    def open_existing_results(self) -> None:
        """Restore a completed multi-band simulation from its output directory."""
        start_dir = (
            str(self.last_simulation_dir)
            if self.last_simulation_dir and self.last_simulation_dir.is_dir()
            else str(self.current_config.parent if self.current_config else Path.cwd())
        )
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Open existing simulation results", start_dir
        )
        if not selected_dir:
            return

        output_dir = Path(selected_dir).expanduser().resolve()
        response_csv = output_dir / "broadband_response.csv"
        config_snapshot = output_dir / "design.yaml"
        if not response_csv.is_file():
            QMessageBox.warning(
                self, "Results not found",
                "The selected folder does not contain broadband_response.csv."
            )
            return
        # Broadband analysis is driven by the CSV and does not require the
        # original YAML snapshot. Keep the snapshot when one is available.
        self.last_simulation_dir = output_dir
        self.last_simulation_config = config_snapshot if config_snapshot.is_file() else None
        self.last_simulation_multiband = True
        self.run_postprocess_button.setEnabled(
            self._postprocess_process.state() == QProcess.ProcessState.NotRunning
        )
        self.run_postprocess_button.setToolTip(
            "Calculate broadband matching metrics and, when recorded, full far-field characteristics."
        )
        run_manifest = output_dir / "multiband_run.json"
        far_field_available = False
        if run_manifest.is_file():
            try:
                far_field_available = bool(json.loads(run_manifest.read_text(encoding="utf-8")).get("far_field_recorded"))
            except (OSError, ValueError, TypeError):
                far_field_available = False
        self._load_result_catalog(output_dir)

        metrics_path = output_dir / "multiband_metrics.json"
        if metrics_path.is_file():
            self._show_multiband_metrics(metrics_path)
        else:
            self.results_summary.setText(
                "Broadband response found. Run post-processing to calculate "
                "matching bandwidth, resonances, VSWR and impedance metrics."
            )
        self.results_status.setText(
            f"Loaded existing broadband simulation results: {output_dir}"
        )
        self.results_log.appendPlainText(
            f"Loaded existing results folder: {output_dir}"
        )
        self.statusBar().showMessage("Existing simulation results loaded")

    def start_postprocessing(self) -> None:
        if self._postprocess_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Post-processing", "Post-processing is already running.")
            return
        if not self.last_simulation_dir or not self.last_simulation_dir.is_dir():
            QMessageBox.warning(self, "Post-processing", "Open an existing results folder or run an FDTD simulation first.")
            return
        if self.last_simulation_multiband:
            if not (self.last_simulation_dir / "broadband_response.csv").is_file():
                QMessageBox.warning(
                    self, "Post-processing",
                    "The selected results folder no longer contains broadband_response.csv."
                )
                return
        elif not self.last_simulation_config or not self.last_simulation_config.is_file():
            QMessageBox.warning(
                self, "Post-processing",
                "The single-band design snapshot is missing; select a valid simulation output."
            )
            return

        self.results_log.clear()
        self.results_log.appendPlainText("Starting post-processing of the latest simulation…")
        self.run_postprocess_button.setEnabled(False)
        self.run_simulation_button.setEnabled(False)
        self.results_status.setText(f"Processing: {self.last_simulation_dir}")
        if self.last_simulation_multiband:
            response_csv = self.last_simulation_dir / "broadband_response.csv"
            if not response_csv.is_file():
                self.results_status.setText("Broadband response CSV is missing; post-processing cannot continue.")
                self.run_postprocess_button.setEnabled(False)
                self.run_simulation_button.setEnabled(True)
                return
            process_args = [
                "-m", "multiband_fdtd.results",
                "--csv", str(response_csv),
                "--output-dir", str(self.last_simulation_dir),
            ]
            run_manifest = self.last_simulation_dir / "multiband_run.json"
            if run_manifest.is_file():
                try:
                    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
                    if manifest.get("far_field_recorded"):
                        process_args.extend(["--simulation-dir", str(self.last_simulation_dir)])
                except (OSError, ValueError, TypeError):
                    pass
        else:
            process_args = [
                "-m", "postprocess",
                "--config", str(self.last_simulation_config),
                "--sim-dir", str(self.last_simulation_dir),
                "--no-show",
            ]
        self._postprocess_process.start(sys.executable, process_args)

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
        can_process = bool(
            self.last_simulation_dir and self.last_simulation_dir.is_dir()
            and self.last_simulation_config and self.last_simulation_config.is_file()
        )
        if self.last_simulation_multiband and self.last_simulation_dir:
            can_process = (self.last_simulation_dir / "broadband_response.csv").is_file()
        self.run_postprocess_button.setEnabled(can_process)
        if exit_code == 0 and self.last_simulation_dir:
            if self.last_simulation_multiband:
                self._load_result_catalog(self.last_simulation_dir)
                self._show_multiband_metrics(self.last_simulation_dir / "multiband_metrics.json")
                self.results_status.setText(
                    f"Broadband post-processing complete. Results: {self.last_simulation_dir}"
                )
            else:
                legacy_plots = []
                for label, filename in (
                    ("S-parameter response", "s11_plot.png"),
                    ("Radiation pattern", "radiation_plot.png"),
                ):
                    if (self.last_simulation_dir / filename).is_file():
                        legacy_plots.append({
                            "label": label, "path": str(self.last_simulation_dir / filename)
                        })
                self._set_result_plot_catalog(legacy_plots)
                self.results_status.setText(f"Post-processing complete. Results: {self.last_simulation_dir}")
            self.statusBar().showMessage("Post-processing completed")
        else:
            self.results_status.setText("Post-processing failed. See the log for details.")
            self.results_log.appendPlainText(f"Post-processing process exited with code {exit_code}.")
            self.statusBar().showMessage("Post-processing failed")

    def _show_multiband_metrics(self, metrics_path: Path) -> None:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            z = metrics["impedance_at_minimum_s11_ohm"]
            bands10 = metrics["bands_below_minus_10_db"]
            bands20 = metrics["bands_below_minus_20_db"]
            lines = [
                (
                    f"{metrics['frequency_start_ghz']:.4g}–{metrics['frequency_stop_ghz']:.4g} GHz "
                    f"({metrics['sample_count']} pts)  |  "
                    f"Min |S11|: {metrics['minimum_s11_db']:.2f} dB "
                    f"@ {metrics['minimum_s11_frequency_ghz']:.4g} GHz  |  "
                    f"VSWR: {metrics['vswr_at_minimum_s11']:.2f}"
                ),
                (
                    f"Bandwidth: −10 dB {metrics['total_bandwidth_below_minus_10_db_ghz']:.3g} GHz "
                    f"({len(bands10)} bands)  |  "
                    f"−20 dB {metrics['total_bandwidth_below_minus_20_db_ghz']:.3g} GHz "
                    f"({len(bands20)} bands)  |  "
                    f"Z@min: {z['real']:.1f}{z['imag']:+.1f}j Ω"
                ),
            ]
            far_field_path = metrics_path.parent / "far_field_metrics.json"
            if far_field_path.is_file():
                far_field = json.loads(far_field_path.read_text(encoding="utf-8"))
                frequencies = far_field["frequency_ghz"]
                directivity = far_field["directivity_dbi"]
                gain = far_field["realized_gain_dbi"]
                efficiency = far_field["total_efficiency_percent"]
                best_dir = max(range(len(directivity)), key=lambda i: directivity[i])
                best_gain = max(range(len(gain)), key=lambda i: gain[i])
                resonance_eval = min(
                    range(len(frequencies)),
                    key=lambda i: abs(frequencies[i] - metrics["minimum_s11_frequency_ghz"]),
                )
                lines.append(
                    f"Peak directivity {directivity[best_dir]:.1f} dBi  |  "
                    f"Peak gain {gain[best_gain]:.1f} dBi  |  "
                    f"Total efficiency at resonance {efficiency[resonance_eval]:.1f}% "
                    "(radiation efficiency assumed 100% for ideal PEC)"
                )
            self.results_summary.setText("\n".join(lines))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.results_summary.setText(f"Could not load broadband performance metrics: {exc}")

    def _load_result_catalog(self, results_dir: Path) -> None:
        catalog_path = results_dir / "plot_catalog.json"
        entries = []
        if catalog_path.is_file():
            try:
                raw = json.loads(catalog_path.read_text(encoding="utf-8"))
                for item in raw:
                    image_path = Path(item.get("path") or (results_dir / item["filename"]))
                    if not image_path.is_absolute():
                        image_path = results_dir / image_path
                    if image_path.is_file():
                        entries.append({"label": str(item["label"]), "path": str(image_path)})
            except (OSError, ValueError, TypeError, KeyError):
                entries = []
        if not entries:
            for label, filename in (
                ("|S11| magnitude (dB)", "s11_plot.png"),
                ("Integrated overview (legacy)", "multiband_results.png"),
                ("Integrated overview (2×2)", "integrated_overview.png"),
                ("Radiation results", "radiation_plot.png"),
            ):
                image_path = results_dir / filename
                if image_path.is_file():
                    entries.append({"label": label, "path": str(image_path)})
        self._set_result_plot_catalog(entries)

    def _set_result_plot_catalog(self, entries: list[dict[str, str]]) -> None:
        self._result_plot_catalog = list(entries)
        self._result_plot_directory = (
            Path(entries[0]["path"]).parent if entries else None
        )
        self.results_plot_combo.blockSignals(True)
        self.results_plot_combo.clear()
        if entries:
            for item in entries:
                self.results_plot_combo.addItem(item["label"], item["path"])
            self.results_plot_combo.setEnabled(True)
            # The first catalog item is always the S11 magnitude plot.
            self.results_plot_combo.setCurrentIndex(0)
        else:
            self.results_plot_combo.addItem("No plots available", "")
            self.results_plot_combo.setEnabled(False)
        self.results_plot_combo.blockSignals(False)
        self._display_selected_result_plot()

    def _display_selected_result_plot(self, *_args) -> None:
        image_path = self.results_plot_combo.currentData()
        pixmap = QPixmap(str(image_path)) if image_path else QPixmap()
        self._s11_original_pixmap = pixmap
        if pixmap.isNull():
            self.s11_plot_label.setPixmap(QPixmap())
            self.s11_plot_label.setText(
                "Select a completed simulation or run post-processing to generate plots."
            )
        else:
            self.s11_plot_label.setText("")
        self._refresh_result_plots()

    def _load_result_plot(self, label: QLabel, image_path: Path) -> None:
        """Compatibility loader for legacy single-band result images."""
        pixmap = QPixmap(str(image_path))
        self._s11_original_pixmap = pixmap
        if pixmap.isNull():
            self.s11_plot_label.setText(f"Could not load plot: {image_path}")
        else:
            self.s11_plot_label.setText("")
        self._refresh_result_plots()

    def _refresh_result_plots(self) -> None:
        original = self._s11_original_pixmap
        if original.isNull() or not hasattr(self, "s11_plot_label"):
            return
        available_width = max(1, self.s11_plot_label.width() - 20)
        available_height = max(1, self.s11_plot_label.height() - 20)
        self.s11_plot_label.setPixmap(original.scaled(
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
        can_process = bool(
            self.last_simulation_dir and self.last_simulation_dir.is_dir()
            and (
                (self.last_simulation_multiband
                 and (self.last_simulation_dir / "broadband_response.csv").is_file())
                or
                (not self.last_simulation_multiband
                 and self.last_simulation_config and self.last_simulation_config.is_file())
            )
        )
        self.run_postprocess_button.setEnabled(can_process)
        self.results_status.setText("Could not start post-processing. Check Python and Matplotlib dependencies.")
        self.results_log.appendPlainText(
            "Could not start the post-processing process. Check the Python environment and openEMS imports."
        )
        self.statusBar().showMessage("Post-processing failed to start")

    def _simulation_error(self, _error) -> None:
        self._simulation_progress_timer.stop()
        self._simulation_started_at = None
        self.run_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
        self.simulation_stage_label.setText("Stage: could not start simulation process")
        self.simulation_eta_label.setText("Remaining: —")
        self._set_simulation_status("SIMULATION FAILED TO START", "failure")
        self.simulation_log.appendPlainText(
            "Could not start the simulation process. Check the Python environment and openEMS imports."
        )
        self.statusBar().showMessage("FDTD simulation failed to start")
