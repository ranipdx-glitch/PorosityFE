#!/usr/bin/env python3
"""Porosity FE Analysis — PyQt6 GUI.

Provides a standalone desktop application for configuring and running
porosity defect analysis on composite laminates. Layout mirrors the
WrinkleFE GUI (left sidebar + central plots + bottom text output).

Usage
-----
>>> from porosity_gui import launch
>>> launch()  # Opens the GUI application
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import sys
import threading
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton,
        QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter,
        QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog,
        QProgressBar, QFormLayout, QScrollArea, QSizePolicy,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QElapsedTimer
    from PyQt6.QtGui import QAction
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


def parse_layup(text: str) -> list:
    """Parse a layup string like '[0/45/-45/90]_3s' to a flat angle list.

    Raises ValueError on malformed input (empty, non-numeric tokens,
    invalid repeat counts) rather than silently substituting a default.
    Pure function, kept module-level so it is testable without Qt.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Layup string is empty.")

    cleaned = text.replace("[", "").replace("]", "")

    symmetric = cleaned.endswith("s")
    if symmetric:
        cleaned = cleaned[:-1]

    repeat = 1
    if "_" in cleaned:
        cleaned, repeat_token = cleaned.rsplit("_", 1)
        try:
            repeat = int(repeat_token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid repeat count {repeat_token!r} in layup {text!r}; "
                f"expected an integer after the '_' (e.g. '[0/90]_3s')."
            ) from exc
        if repeat < 1:
            raise ValueError(
                f"Repeat count must be >= 1, got {repeat} in layup {text!r}."
            )

    sep = "/" if "/" in cleaned else ","
    tokens = [a.strip() for a in cleaned.split(sep) if a.strip()]
    if not tokens:
        raise ValueError(f"No ply angles found in layup {text!r}.")
    try:
        angles = [float(a) for a in tokens]
    except ValueError as exc:
        bad = next((a for a in tokens if not _is_float(a)), tokens[0])
        raise ValueError(
            f"Invalid ply angle {bad!r} in layup {text!r}; "
            f"expected numeric degrees (e.g. '[0/45/-45/90]_3s')."
        ) from exc

    angles = angles * repeat
    if symmetric:
        angles = angles + list(reversed(angles))
    return angles


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL_QT = True
except ImportError:
    HAS_MPL_QT = False

import numpy as np


def _check_pyqt6() -> None:
    """Raise ImportError with helpful message if PyQt6 is missing."""
    if not HAS_PYQT6:
        raise ImportError(
            "Porosity FE GUI requires PyQt6. Install with:\n"
            "  pip install porosity-fe[gui]\n"
            "Or, if you already have the source checkout:\n"
            "  pip install PyQt6\n"
            "Or run the analysis script directly without the GUI:\n"
            "  python porosity_fe_analysis.py"
        )


# ======================================================================
# Result export helpers (module-level so they're testable without Qt)
# ======================================================================

def build_export_payload(result: dict) -> dict:
    """Flatten an analysis result into the export payload structure.

    Shared by the JSON and CSV writers so both formats describe the same
    fields. ``result`` is the dict produced by ``AnalysisWorker`` and stored
    on the main window as ``self._result``.
    """
    cfg = result["config"]
    emp = result["empirical"]
    payload = {
        "config": {
            "material": cfg["material_name"],
            "n_plies": cfg["n_plies"],
            "t_ply": cfg["t_ply"],
            "Vp_percent": cfg["Vp"],
            "distribution": cfg["distribution"],
            "void_shape": cfg["void_shape"],
            "mesh": f"{cfg['nx']}x{cfg['ny']}x{cfg['nz']}",
        },
        "empirical": {},
    }
    for mode in emp:
        payload["empirical"][mode] = {}
        for model in emp[mode]:
            r = emp[mode][model]
            payload["empirical"][mode][model] = {
                "failure_stress_MPa": r["failure_stress"],
                "knockdown": r["knockdown"],
            }
    return payload


def write_results_json(filepath: str, payload: dict) -> None:
    # Prepend the schema envelope (#20) while keeping the payload keys
    # flat at the top level for backward compatibility.
    from porosity_fe_analysis import JSON_SCHEMA_VERSION, FORMAT_EMPIRICAL_SWEEP
    envelope = {
        "schema_version": JSON_SCHEMA_VERSION,
        "format": FORMAT_EMPIRICAL_SWEEP,
        **payload,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2)


def write_results_csv(filepath: str, payload: dict) -> None:
    """Write the export payload as a flat CSV.

    Configuration metadata is written as comment lines prefixed with ``#``;
    pandas (``read_csv(comment='#')``) and most CSV viewers handle this
    cleanly, while Excel ignores the comments and treats the table as data.
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        # Config preamble as comment lines
        for key, value in payload["config"].items():
            f.write(f"# {key}: {value}\n")
        writer = csv.writer(f)
        writer.writerow(["mode", "model", "failure_stress_MPa", "knockdown"])
        for mode in payload["empirical"]:
            for model in payload["empirical"][mode]:
                r = payload["empirical"][mode][model]
                writer.writerow([
                    mode, model,
                    r["failure_stress_MPa"],
                    r["knockdown"],
                ])


# ======================================================================
# Analysis worker thread
# ======================================================================

if HAS_PYQT6:

    class AnalysisWorker(QThread):
        """Background thread for running porosity analysis."""

        finished = pyqtSignal(object)  # results dict
        error = pyqtSignal(str)
        progress = pyqtSignal(str)

        def __init__(self, config: dict) -> None:
            super().__init__()
            self.config = config
            # threading.Event has built-in memory-fence semantics; a plain
            # bool can be missed by the worker if the write hasn't propagated
            # across cores before the next checkpoint read.
            self._stop_event = threading.Event()

        def request_stop(self) -> None:
            self._stop_event.set()

        @property
        def _stop_requested(self) -> bool:
            """Back-compat accessor for the six checkpoint reads in run()."""
            return self._stop_event.is_set()

        def run(self) -> None:
            try:
                from porosity_fe_analysis import (
                    MATERIALS, PorosityField, CompositeMesh,
                    EmpiricalSolver,
                    FESolver, FieldResults,
                )

                cfg = self.config

                # --- Material ---
                self.progress.emit("Setting up material and laminate...")
                if cfg["material_name"] not in MATERIALS:
                    raise ValueError(
                        f"Unknown material {cfg['material_name']!r}. "
                        f"Available presets: {sorted(MATERIALS)}."
                    )
                material = MATERIALS[cfg["material_name"]]
                material = dataclasses.replace(
                    material,
                    t_ply=cfg["t_ply"],
                    n_plies=cfg["n_plies"],
                )

                if self._stop_requested:
                    return

                # --- Porosity field ---
                self.progress.emit("Creating porosity field...")
                pf_kwargs = {
                    "distribution": cfg["distribution"],
                    "void_shape": cfg["void_shape"],
                }
                if cfg["distribution"] == "clustered":
                    pf_kwargs["cluster_location"] = cfg["cluster_location"]

                porosity_field = PorosityField(
                    material,
                    cfg["Vp"] / 100.0,  # convert % to fraction
                    **pf_kwargs,
                )

                if self._stop_requested:
                    return

                # --- Mesh ---
                self.progress.emit("Generating mesh...")
                mesh = CompositeMesh(
                    porosity_field, material,
                    nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"],
                    ply_angles=cfg["angles"],
                )

                if self._stop_requested:
                    return

                # --- Solvers ---
                self.progress.emit("Running empirical solver (Judd-Wright, Power Law, Linear)...")
                empirical = EmpiricalSolver(mesh, material, ply_angles=cfg["angles"])
                emp_results = empirical.get_all_failure_loads()

                if self._stop_requested:
                    return

                # --- FE Solver ---
                # FESolver BCs don't support ILSS short-beam shear today; rather
                # than silently substituting compression and reporting a knockdown
                # the user didn't ask for (#9), skip the FE pass entirely for
                # ILSS and let the empirical bar stand alone in the comparison.
                loading_mode = cfg["loading_mode"]
                fe_supported = loading_mode in ("compression", "tension", "shear")

                fe_solver = None
                fe_field = None
                fe_loading = None
                if fe_supported:
                    applied_strain = -0.01 if loading_mode == "compression" else 0.01
                    fe_loading = loading_mode

                    self.progress.emit("Assembling stiffness matrix (FE)...")
                    fe_solver = FESolver(mesh, material, porosity_field, ply_angles=cfg["angles"])

                    if self._stop_requested:
                        return

                    self.progress.emit("Solving FE system...")
                    fe_field = fe_solver.solve(
                        loading=fe_loading,
                        applied_strain=applied_strain,
                        verbose=False,
                    )

                    if self._stop_requested:
                        return

                    self.progress.emit("Recovering FE stresses...")
                else:
                    self.progress.emit(
                        f"Skipping FE solve (mode '{loading_mode}' not supported "
                        f"by FE BCs; empirical results only)."
                    )

                self.progress.emit("Analysis complete.")

                results = {
                    "config": cfg,
                    "material": material,
                    "porosity_field": porosity_field,
                    "mesh": mesh,
                    "empirical_solver": empirical,
                    "empirical": emp_results,
                    "fe_solver": fe_solver,
                    "fe_field": fe_field,
                    "fe_loading": fe_loading,
                    "fe_skipped_reason": (
                        None if fe_supported
                        else f"FE solver does not support '{loading_mode}' boundary conditions"
                    ),
                    "f_md": empirical.f_md,
                }

                self.finished.emit(results)

            except Exception as e:
                logger.exception("Analysis worker raised")
                self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ======================================================================
# Main Window
# ======================================================================

if HAS_PYQT6:

    class PorosityFEMainWindow(QMainWindow):
        """Main application window for Porosity FE Analysis.

        Layout
        ------
        +----------------------------------+---------------------------+
        |  Left Sidebar (350px)            |  Central Area             |
        |  - Material & Laminate           |  - Plot tabs              |
        |  - Porosity Parameters           |    (Profile, Mesh,        |
        |  - Mesh                          |     Results)              |
        |  - Analysis Controls             |                           |
        |                                  +---------------------------+
        |                                  |  Bottom: Text Output      |
        +----------------------------------+---------------------------+
        """

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("PorosityFE - Composite Laminate Porosity Analysis")
            self.setMinimumSize(1200, 800)

            self._result = None
            self._worker = None

            self._setup_menu_bar()
            self._setup_central_widget()
            self._setup_status_bar()

        # ----------------------------------------------------------
        # Menu bar
        # ----------------------------------------------------------

        def _setup_menu_bar(self) -> None:
            menubar = self.menuBar()

            # File menu
            file_menu = menubar.addMenu("&File")

            new_action = QAction("&New Analysis", self)
            new_action.triggered.connect(self._on_new)
            file_menu.addAction(new_action)

            export_action = QAction("&Export Results...", self)
            export_action.triggered.connect(self._on_export)
            file_menu.addAction(export_action)

            file_menu.addSeparator()

            quit_action = QAction("&Quit", self)
            quit_action.triggered.connect(self.close)
            file_menu.addAction(quit_action)

            # Help menu
            help_menu = menubar.addMenu("&Help")
            about_action = QAction("&About", self)
            about_action.triggered.connect(self._on_about)
            help_menu.addAction(about_action)

        # ----------------------------------------------------------
        # Central widget
        # ----------------------------------------------------------

        def _setup_central_widget(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)

            main_layout = QHBoxLayout(central)

            # Left sidebar: configuration panels
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_panel.setMaximumWidth(350)

            left_layout.addWidget(self._create_material_group())
            left_layout.addWidget(self._create_porosity_group())
            left_layout.addWidget(self._create_mesh_group())
            left_layout.addWidget(self._create_analysis_group())
            left_layout.addStretch()

            # Right area: splitter with plots on top, text on bottom
            right_splitter = QSplitter(Qt.Orientation.Vertical)

            # Plot tabs with matplotlib canvases
            self.plot_tabs = QTabWidget()

            if HAS_MPL_QT:
                self.profile_canvas = FigureCanvas(Figure(figsize=(8, 5)))
                self.mesh_canvas = FigureCanvas(Figure(figsize=(8, 5)))
                self.results_canvas = FigureCanvas(Figure(figsize=(8, 5)))
                self.stress_canvas = FigureCanvas(Figure(figsize=(8, 5)))

                for canvas in (self.profile_canvas, self.mesh_canvas,
                               self.results_canvas, self.stress_canvas):
                    canvas.setSizePolicy(
                        QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Expanding,
                    )

                self.plot_tabs.addTab(self.profile_canvas, "Profile")
                self.plot_tabs.addTab(self.mesh_canvas, "Mesh")
                self.plot_tabs.addTab(self.results_canvas, "Results")

                # Stress tab: wrap canvas + dropdown in a container widget
                stress_container = QWidget()
                stress_layout = QVBoxLayout(stress_container)
                stress_layout.setContentsMargins(4, 4, 4, 4)

                stress_ctrl_layout = QHBoxLayout()
                stress_ctrl_layout.addWidget(QLabel("Stress component:"))
                self.stress_component_combo = QComboBox()
                self.stress_component_combo.addItems([
                    "\u03c3\u2081\u2081 (fiber)",
                    "\u03c3\u2082\u2082 (transverse)",
                    "\u03c3\u2083\u2083 (through-thickness)",
                    "\u03c4\u2082\u2083 (interlaminar)",
                    "\u03c4\u2081\u2083 (interlaminar)",
                    "\u03c4\u2081\u2082 (in-plane shear)",
                    "Von Mises",
                ])
                self.stress_component_combo.currentIndexChanged.connect(
                    self._on_stress_component_changed
                )
                stress_ctrl_layout.addWidget(self.stress_component_combo)
                stress_ctrl_layout.addStretch()
                stress_layout.addLayout(stress_ctrl_layout)
                stress_layout.addWidget(self.stress_canvas)

                self.plot_tabs.addTab(stress_container, "Stress")

                # Draw placeholder text on each canvas
                for canvas, msg in [
                    (self.profile_canvas, "Run an analysis to see porosity profile plots."),
                    (self.mesh_canvas, "Run an analysis to see mesh visualization."),
                    (self.results_canvas, "Run an analysis to see knockdown results."),
                    (self.stress_canvas, "Run an analysis to see FE stress contours."),
                ]:
                    ax = canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                            ha="center", va="center", fontsize=12, color="0.5")
                    ax.set_axis_off()
                    canvas.draw()
            else:
                self.profile_canvas = None
                self.mesh_canvas = None
                self.results_canvas = None
                self.stress_canvas = None
                self.stress_component_combo = None
                self.plot_tabs.addTab(
                    QLabel("Run an analysis to see porosity profile plots."), "Profile")
                self.plot_tabs.addTab(
                    QLabel("Run an analysis to see mesh visualization."), "Mesh")
                self.plot_tabs.addTab(
                    QLabel("Run an analysis to see knockdown results."), "Results")
                self.plot_tabs.addTab(
                    QLabel("Run an analysis to see FE stress contours."), "Stress")

            right_splitter.addWidget(self.plot_tabs)

            # Text output
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setMinimumHeight(150)
            self.output_text.setPlaceholderText(
                "Analysis results will appear here.\n"
                "Configure parameters on the left and click 'Run Analysis'."
            )
            right_splitter.addWidget(self.output_text)
            right_splitter.setSizes([500, 200])

            main_layout.addWidget(left_panel)
            main_layout.addWidget(right_splitter, stretch=1)

        # ----------------------------------------------------------
        # Configuration groups
        # ----------------------------------------------------------

        def _create_material_group(self) -> QGroupBox:
            group = QGroupBox("Material && Laminate")
            layout = QFormLayout(group)

            self.material_combo = QComboBox()
            self.material_combo.addItems([
                "T800_epoxy", "T700_epoxy", "glass_epoxy",
            ])
            layout.addRow("Material:", self.material_combo)

            self.layup_edit = QLineEdit("[0/45/-45/90]_3s")
            self.layup_edit.setToolTip(
                "Enter ply angles separated by /. Use _Ns for N repeats, "
                "s for symmetric.\n"
                "Examples: [0/45/-45/90]_3s, [0/90]_6s, 0/0/0/90/90/90"
            )
            layout.addRow("Layup:", self.layup_edit)

            self.ply_thickness_spin = QDoubleSpinBox()
            self.ply_thickness_spin.setRange(0.05, 0.50)
            self.ply_thickness_spin.setValue(0.183)
            self.ply_thickness_spin.setSingleStep(0.01)
            self.ply_thickness_spin.setSuffix(" mm")
            layout.addRow("Ply thickness:", self.ply_thickness_spin)

            return group

        def _create_porosity_group(self) -> QGroupBox:
            group = QGroupBox("Porosity Parameters")
            layout = QFormLayout(group)

            self.vp_spin = QDoubleSpinBox()
            self.vp_spin.setRange(0.1, 15.0)
            self.vp_spin.setValue(3.0)
            self.vp_spin.setSingleStep(0.5)
            self.vp_spin.setDecimals(1)
            self.vp_spin.setSuffix(" %")
            self.vp_spin.setToolTip(
                "Void volume fraction (Vp) as a percentage.\n"
                "Typical range: 0.5-5% for autoclave, 2-10% for OOA."
            )
            layout.addRow("Void volume fraction:", self.vp_spin)

            self.distribution_combo = QComboBox()
            self.distribution_combo.addItems([
                "uniform",
                "clustered (midplane)",
                "clustered (surface)",
                "interface",
            ])
            self.distribution_combo.setToolTip(
                "Through-thickness distribution of porosity.\n"
                "uniform: constant porosity throughout.\n"
                "clustered (midplane): Gaussian peak at midplane.\n"
                "clustered (surface): Gaussian peak at surface.\n"
                "interface: concentrated at ply interfaces."
            )
            layout.addRow("Distribution:", self.distribution_combo)

            self.void_shape_combo = QComboBox()
            self.void_shape_combo.addItems([
                "spherical", "cylindrical", "penny",
            ])
            self.void_shape_combo.setToolTip(
                "Void morphology (aspect ratio).\n"
                "spherical: equiaxed (AR=1)\n"
                "cylindrical: prolate (AR=3)\n"
                "penny: oblate disc (AR=10)"
            )
            layout.addRow("Void shape:", self.void_shape_combo)

            self.loading_combo = QComboBox()
            self.loading_combo.addItems([
                "compression", "tension", "shear", "ilss",
            ])
            self.loading_combo.setToolTip(
                "Primary loading mode for failure prediction.\n"
                "All four modes are computed; this selects the\n"
                "primary mode for the results bar chart."
            )
            layout.addRow("Loading mode:", self.loading_combo)

            return group

        def _create_mesh_group(self) -> QGroupBox:
            group = QGroupBox("Mesh")
            layout = QFormLayout(group)

            self.nx_spin = QSpinBox()
            self.nx_spin.setRange(2, 200)
            self.nx_spin.setValue(30)
            layout.addRow("nx:", self.nx_spin)

            self.ny_spin = QSpinBox()
            self.ny_spin.setRange(2, 100)
            self.ny_spin.setValue(10)
            layout.addRow("ny:", self.ny_spin)

            self.nz_spin = QSpinBox()
            self.nz_spin.setRange(2, 100)
            self.nz_spin.setValue(12)
            layout.addRow("nz:", self.nz_spin)

            return group

        def _create_analysis_group(self) -> QGroupBox:
            group = QGroupBox("Analysis")
            layout = QVBoxLayout(group)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)  # indeterminate
            self.progress_bar.setVisible(False)
            layout.addWidget(self.progress_bar)

            self.timer_label = QLabel("")
            self.timer_label.setStyleSheet("QLabel { color: #555; font-size: 11px; }")
            self.timer_label.setVisible(False)
            layout.addWidget(self.timer_label)

            # Elapsed time tracking
            self._elapsed_timer = QElapsedTimer()
            self._tick_timer = QTimer(self)
            self._tick_timer.setInterval(1000)
            self._tick_timer.timeout.connect(self._tick_elapsed)

            btn_layout = QHBoxLayout()

            self.run_btn = QPushButton("Run Analysis")
            self.run_btn.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; "
                "font-weight: bold; padding: 8px; }"
            )
            self.run_btn.clicked.connect(self._on_run)
            btn_layout.addWidget(self.run_btn)

            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setStyleSheet(
                "QPushButton { background-color: #F44336; color: white; "
                "font-weight: bold; padding: 8px; }"
            )
            self.stop_btn.clicked.connect(self._on_stop)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            btn_layout.addWidget(self.stop_btn)

            layout.addLayout(btn_layout)

            return group

        # ----------------------------------------------------------
        # Status bar
        # ----------------------------------------------------------

        def _setup_status_bar(self) -> None:
            self.statusBar().showMessage("Ready")

        # ----------------------------------------------------------
        # Build config from GUI
        # ----------------------------------------------------------

        def _build_config(self) -> dict:
            """Build a config dict from the current GUI state."""
            angles = self._parse_layup(self.layup_edit.text())
            n_plies = len(angles)
            t_ply = self.ply_thickness_spin.value()

            # Parse distribution combo
            dist_text = self.distribution_combo.currentText()
            if dist_text == "uniform":
                distribution = "uniform"
                cluster_location = "midplane"
            elif dist_text == "clustered (midplane)":
                distribution = "clustered"
                cluster_location = "midplane"
            elif dist_text == "clustered (surface)":
                distribution = "clustered"
                cluster_location = "surface"
            elif dist_text == "interface":
                distribution = "interface"
                cluster_location = "midplane"
            else:
                distribution = "uniform"
                cluster_location = "midplane"

            return {
                "material_name": self.material_combo.currentText(),
                "angles": angles,
                "n_plies": n_plies,
                "t_ply": t_ply,
                "Vp": self.vp_spin.value(),
                "distribution": distribution,
                "cluster_location": cluster_location,
                "void_shape": self.void_shape_combo.currentText(),
                "loading_mode": self.loading_combo.currentText(),
                "nx": self.nx_spin.value(),
                "ny": self.ny_spin.value(),
                "nz": self.nz_spin.value(),
            }

        @staticmethod
        def _parse_layup(text: str) -> list:
            """Parse a layup string like '[0/45/-45/90]_3s' to angle list.

            Raises ValueError on malformed input rather than silently
            falling back to a default — see issue #9.
            """
            return parse_layup(text)

        # ----------------------------------------------------------
        # Actions
        # ----------------------------------------------------------

        def _on_run(self) -> None:
            """Run the porosity analysis in a background thread."""
            # Guard against re-entry: a fast double-click on Run could land a
            # second call while the previous worker is still alive. Without
            # this guard, the new AnalysisWorker would overwrite self._worker
            # and the prior thread would be orphaned (request_stop() can only
            # reach the latest worker).
            if self._worker is not None and self._worker.isRunning():
                return

            # Disable the button *before* parsing widgets so a double-click
            # during _build_config() can't start a second analysis. _build_config
            # iterates many widgets and can take long enough on slower hardware
            # for a fast user to click twice.
            self.run_btn.setEnabled(False)
            try:
                config = self._build_config()
            except Exception as e:
                self.run_btn.setEnabled(True)
                QMessageBox.critical(self, "Configuration Error", str(e))
                return

            self.stop_btn.setEnabled(True)
            self.stop_btn.setVisible(True)
            self.statusBar().showMessage("Running analysis...")
            self.output_text.clear()

            # Show progress bar and timer
            self.progress_bar.setRange(0, 0)  # indeterminate
            self.progress_bar.setVisible(True)
            self.timer_label.setVisible(True)
            self._elapsed_timer.start()
            self._update_timer_display()
            self._tick_timer.start()

            self._worker = AnalysisWorker(config)
            self._worker.finished.connect(self._on_analysis_done)
            self._worker.error.connect(self._on_analysis_error)
            self._worker.progress.connect(self._on_progress)
            self._worker.start()

        def _tick_elapsed(self) -> None:
            """Update elapsed time display every second."""
            self._update_timer_display()

        def _update_timer_display(self) -> None:
            """Refresh the timer label."""
            elapsed_s = self._elapsed_timer.elapsed() / 1000.0
            elapsed_str = self._format_time(elapsed_s)
            self.timer_label.setText(f"Elapsed: {elapsed_str}")

        @staticmethod
        def _format_time(seconds: float) -> str:
            """Format seconds as M:SS or H:MM:SS."""
            seconds = int(seconds)
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                m, s = divmod(seconds, 60)
                return f"{m}:{s:02d}"
            else:
                h, rem = divmod(seconds, 3600)
                m, s = divmod(rem, 60)
                return f"{h}:{m:02d}:{s:02d}"

        def _stop_progress(self) -> None:
            """Stop the progress bar and timer."""
            self._tick_timer.stop()
            self.progress_bar.setVisible(False)
            self.timer_label.setVisible(False)

        def _on_analysis_done(self, result: dict) -> None:
            """Handle completed analysis."""
            actual_s = self._elapsed_timer.elapsed() / 1000.0
            self._stop_progress()

            self._result = result
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)

            # Build summary text
            summary = self._format_results_text(result)
            self.output_text.setPlainText(summary)

            # Update plots
            self._update_plots(result)

            # Status bar
            mode = result["config"]["loading_mode"]
            jw_kd = result["empirical"][mode]["judd_wright"]["knockdown"]
            fe_field = result.get("fe_field")
            fe_str = f"  |  FE knockdown = {fe_field.knockdown:.3f}" if fe_field is not None else ""
            self.statusBar().showMessage(
                f"Analysis complete in {actual_s:.1f}s. "
                f"Judd-Wright {mode} knockdown = {jw_kd:.3f}{fe_str}"
            )

        def _on_analysis_error(self, msg: str) -> None:
            """Handle analysis error."""
            self._stop_progress()
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            self.output_text.setPlainText(f"ERROR:\n{msg}")
            self.statusBar().showMessage("Analysis failed.")
            QMessageBox.critical(self, "Analysis Error", msg[:500])

        def _on_stop(self) -> None:
            """Stop the running analysis."""
            self._stop_progress()

            if self._worker is not None:
                worker = self._worker
                worker.request_stop()
                if worker.isRunning():
                    worker.terminate()
                    worker.wait(2000)
                    if worker.isRunning():
                        # terminate() is advisory — if the worker is wedged in
                        # a C extension (numpy/scipy can sit in BLAS calls), it
                        # may keep running. Surface this rather than pretending
                        # the analysis was cleanly stopped.
                        logger.error(
                            "AnalysisWorker did not terminate within 2s; "
                            "thread state remains running."
                        )
                        QMessageBox.warning(
                            self, "Stop Incomplete",
                            "The analysis worker did not terminate within "
                            "the 2-second grace period — it may still be "
                            "consuming CPU. Please restart the application "
                            "if this keeps happening."
                        )
                self._worker = None

            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setVisible(False)
            self.statusBar().showMessage("Analysis stopped by user.")
            self.output_text.setPlainText("Analysis was stopped by user.")

        def _on_progress(self, msg: str) -> None:
            """Update status bar with progress message."""
            self.statusBar().showMessage(msg)

        def closeEvent(self, event) -> None:
            """Ensure the worker thread exits before the window closes.

            Without this override, closing the window mid-analysis terminates
            the QApplication event loop while the QThread keeps running until
            its solve completes — the Python process keeps consuming CPU
            even though the UI is gone.
            """
            worker = getattr(self, "_worker", None)
            if worker is not None and worker.isRunning():
                worker.request_stop()
                # Give the cooperative checkpoints a chance to fire before
                # falling back to terminate(). 2s matches _on_stop's grace.
                if not worker.wait(2000):
                    worker.terminate()
                    worker.wait()  # no timeout — must finish before close
            event.accept()

        # ----------------------------------------------------------
        # Results formatting
        # ----------------------------------------------------------

        def _format_results_text(self, result: dict) -> str:
            """Format analysis results into a readable text summary."""
            cfg = result["config"]
            material = result["material"]
            emp = result["empirical"]
            mesh = result["mesh"]
            fe_field = result.get("fe_field")
            fe_loading = result.get("fe_loading", "compression")

            f_md = result.get("f_md", 0.5)

            lines = []
            lines.append("=" * 70)
            lines.append("POROSITY ANALYSIS RESULTS")
            lines.append("=" * 70)
            lines.append(f"Material:       {cfg['material_name']}")
            layup_str = self.layup_edit.text().strip()
            lines.append(f"Layup:          {layup_str} ({cfg['n_plies']} plies, t_ply = {cfg['t_ply']:.3f} mm)")
            lines.append(f"Vp:             {cfg['Vp']:.1f}%")
            lines.append(f"Distribution:   {cfg['distribution']}")
            lines.append(f"Void shape:     {cfg['void_shape']}")
            lines.append(f"Mesh:           {cfg['nx']}x{cfg['ny']}x{cfg['nz']} "
                         f"({len(mesh.nodes)} nodes, {len(mesh.elements)} elements)")
            lines.append("")

            modes = ["compression", "tension", "shear", "ilss"]
            models = ["judd_wright", "power_law", "linear"]

            lines.append("-" * 70)
            lines.append("EMPIRICAL MODEL KNOCKDOWN FACTORS")
            lines.append(f"  (evaluated at mean Vp = {cfg['Vp']:.1f}%)")
            if f_md < 0.49:
                lines.append(f"  Layup scaling: f_md = {f_md:.2f} "
                             f"(coefficients reduced for fiber-dominated layup)")
            elif f_md > 0.51:
                lines.append(f"  Layup scaling: f_md = {f_md:.2f} "
                             f"(coefficients increased for matrix-dominated layup)")
            else:
                lines.append(f"  Layup scaling: f_md = {f_md:.2f} (QI reference)")
            lines.append("-" * 70)
            header = f"{'Mode':<15}"
            for m in models:
                header += f"{m.replace('_', ' ').title():>18}"
            lines.append(header)
            lines.append("-" * 70)
            for mode in modes:
                row = f"{mode:<15}"
                for model in models:
                    kd = emp[mode][model]["knockdown"]
                    fs = emp[mode][model]["failure_stress"]
                    row += f"  {kd:.3f} ({fs:.0f} MPa)"
                lines.append(row)

            lines.append("")
            lines.append("-" * 70)
            lines.append("RANKINGS (compression, Judd-Wright)")
            lines.append("-" * 70)
            # Single config, so just show the knockdown
            comp_kd = emp["compression"]["judd_wright"]["knockdown"]
            comp_fs = emp["compression"]["judd_wright"]["failure_stress"]
            lines.append(f"  Knockdown = {comp_kd:.3f}, Failure stress = {comp_fs:.0f} MPa")

            if fe_field is None:
                fe_skipped_reason = result.get("fe_skipped_reason")
                if fe_skipped_reason:
                    lines.append("")
                    lines.append("-" * 70)
                    lines.append(f"FE solve skipped: {fe_skipped_reason}.")
                    lines.append("Empirical results above use this loading mode directly.")
            else:
                lines.append("")
                lines.append("-" * 70)
                lines.append("FINITE ELEMENT ANALYSIS RESULTS")
                lines.append(f"  Loading mode:     {fe_loading}")
                lines.append(f"  FE Knockdown:     {fe_field.knockdown:.4f}")
                lines.append(f"  Max Tsai-Wu FI:   {fe_field.max_failure_index:.4f}")

                disp = fe_field.displacement  # (n_nodes, 3)
                max_disp = float(np.linalg.norm(disp, axis=1).max())
                lines.append(f"  Max displacement: {max_disp:.4e} mm")

                sg = fe_field.stress_global  # (n_elem, n_gp, 6)
                s_avg = sg.mean(axis=1)  # (n_elem, 6)
                comp_names = ["sigma_11", "sigma_22", "sigma_33", "tau_23", "tau_13", "tau_12"]
                lines.append("  Stress range (MPa):")
                for ci, cname in enumerate(comp_names):
                    sv = s_avg[:, ci]
                    lines.append(f"    {cname:<12}  min={sv.min():10.2f}  max={sv.max():10.2f}")

                # Von Mises
                s1, s2, s3 = s_avg[:, 0], s_avg[:, 1], s_avg[:, 2]
                s4, s5, s6 = s_avg[:, 3], s_avg[:, 4], s_avg[:, 5]
                vm = np.sqrt(0.5 * (
                    (s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2
                    + 6.0 * (s4**2 + s5**2 + s6**2)
                ))
                lines.append(f"    {'von_mises':<12}  min={vm.min():10.2f}  max={vm.max():10.2f}")

            return "\n".join(lines)

        # ----------------------------------------------------------
        # Plot updates
        # ----------------------------------------------------------

        def _update_plots(self, result: dict) -> None:
            """Update all plot tabs with analysis results."""
            if not HAS_MPL_QT or self.profile_canvas is None:
                return

            import matplotlib.pyplot as plt

            self._plot_profile(result)
            self._plot_mesh(result)
            self._plot_results(result)
            self._plot_stress(result)

            plt.close("all")

        def _plot_profile(self, result: dict) -> None:
            """Draw through-thickness porosity profile on Profile tab (single panel)."""
            fig = self.profile_canvas.figure
            fig.clear()

            try:
                pf = result["porosity_field"]

                ax = fig.add_subplot(111)
                z, Vp = pf.effective_porosity_profile(nz=200)
                ax.plot(Vp * 100, z, "b-", linewidth=2)
                ax.set_xlabel("Porosity (%)", fontsize=11)
                ax.set_ylabel("z (mm)", fontsize=11)
                ax.set_title("Through-Thickness Porosity Profile",
                             fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(left=0)

                fig.tight_layout()
            except Exception as e:
                logger.exception("Profile plot failed")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Profile plot error:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="red")
                ax.set_axis_off()

            self.profile_canvas.draw()

        def _plot_mesh(self, result: dict) -> None:
            """Draw 2D mesh cross-section colored by stiffness reduction on Mesh tab."""
            fig = self.mesh_canvas.figure
            fig.clear()

            try:
                mesh = result["mesh"]
                nx1 = mesh.nx + 1
                ny1 = mesh.ny + 1
                ny_mid = mesh.ny // 2

                # Gather cross-section nodes at mid-y
                indices = []
                for k in range(mesh.nz + 1):
                    for i in range(mesh.nx + 1):
                        idx = k * ny1 * nx1 + ny_mid * nx1 + i
                        indices.append(idx)
                indices = np.array(indices)

                X = mesh.nodes[indices, 0].reshape(mesh.nz + 1, mesh.nx + 1)
                Z = mesh.nodes[indices, 2].reshape(mesh.nz + 1, mesh.nx + 1)
                # Stiffness retention = 1 - porosity
                Sr = mesh.stiffness_reduction[indices].reshape(mesh.nz + 1, mesh.nx + 1)

                ax = fig.add_subplot(111)
                im = ax.contourf(X, Z, Sr * 100, levels=20, cmap="viridis",
                                 vmin=max(0, Sr.min() * 100 - 1), vmax=100)
                cb = fig.colorbar(im, ax=ax, label="Stiffness Retention (%)")

                # Overlay element boundaries for a FE-mesh look (sampled)
                step_x = max(1, mesh.nx // 20)
                step_z = max(1, mesh.nz // 20)
                for k in range(0, mesh.nz + 1, step_z):
                    row_x = mesh.nodes[
                        [k * ny1 * nx1 + ny_mid * nx1 + i for i in range(mesh.nx + 1)], 0
                    ]
                    row_z = mesh.nodes[
                        [k * ny1 * nx1 + ny_mid * nx1 + i for i in range(mesh.nx + 1)], 2
                    ]
                    ax.plot(row_x, row_z, "k-", linewidth=0.3, alpha=0.4)
                for i in range(0, mesh.nx + 1, step_x):
                    col_x = mesh.nodes[
                        [k * ny1 * nx1 + ny_mid * nx1 + i for k in range(mesh.nz + 1)], 0
                    ]
                    col_z = mesh.nodes[
                        [k * ny1 * nx1 + ny_mid * nx1 + i for k in range(mesh.nz + 1)], 2
                    ]
                    ax.plot(col_x, col_z, "k-", linewidth=0.3, alpha=0.4)

                # Mark void elements as filled polygons (element face outlines)
                void_elems = mesh.void_elements
                if len(void_elems) > 0:
                    from matplotlib.patches import Polygon
                    from matplotlib.collections import PatchCollection

                    void_patches = []
                    for e_idx in void_elems:
                        # Only draw elements that lie in the mid-y slice
                        j_e = (e_idx // mesh.nx) % mesh.ny
                        if j_e != mesh.ny // 2:
                            continue
                        nodes_of_elem = mesh.elements[e_idx]
                        node_coords = mesh.nodes[nodes_of_elem]  # (8, 3)
                        # Extract x-z coordinates for all 8 nodes
                        xz = node_coords[:, [0, 2]]  # (8, 2)
                        unique_xz = np.unique(xz, axis=0)  # typically (4, 2)
                        if len(unique_xz) < 3:
                            continue
                        # Order by angle from centroid to form proper polygon
                        cx_p, cz_p = unique_xz.mean(axis=0)
                        angles = np.arctan2(
                            unique_xz[:, 1] - cz_p,
                            unique_xz[:, 0] - cx_p,
                        )
                        order = np.argsort(angles)
                        poly_coords = unique_xz[order]
                        patch = Polygon(poly_coords, closed=True)
                        void_patches.append(patch)

                    if void_patches:
                        pc = PatchCollection(
                            void_patches,
                            facecolor="white",
                            edgecolor="red",
                            linewidth=1.0,
                            zorder=5,
                            alpha=1.0,
                        )
                        ax.add_collection(pc)
                        # Dummy artist for legend
                        ax.plot([], [], "s", color="white", markeredgecolor="red",
                                markeredgewidth=1.0,
                                label=f"Voids ({len(void_patches)})")
                        ax.legend(fontsize=8, loc="upper right")

                ax.set_xlabel("x (mm)", fontsize=11)
                ax.set_ylabel("z (mm)", fontsize=11)
                ax.set_title(
                    f"FE Mesh — Stiffness Retention  |  "
                    f"{len(mesh.nodes):,} nodes, {len(mesh.elements):,} elements",
                    fontsize=12, fontweight="bold",
                )
                ax.set_aspect("equal")

                fig.tight_layout()
            except Exception as e:
                logger.exception("Mesh plot failed")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Mesh plot error:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="red")
                ax.set_axis_off()

            self.mesh_canvas.draw()

        def _plot_results(self, result: dict) -> None:
            """Draw knockdown bar chart on Results tab."""
            fig = self.results_canvas.figure
            fig.clear()

            try:
                emp = result["empirical"]
                fe_field = result.get("fe_field")
                fe_loading = result.get("fe_loading", "compression")
                cfg = result["config"]
                f_md = result.get("f_md", 0.5)

                modes = ["compression", "tension", "shear", "ilss"]
                # Strength models (solid bars)
                models = ["judd_wright", "power_law", "linear"]
                model_labels = ["Judd-Wright", "Power Law", "Linear"]
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                hatches = [None, None, None]

                # FE stiffness bar (hatched to distinguish from strength)
                has_fe = fe_field is not None
                if has_fe:
                    models.append("fe")
                    model_labels.append(f"FE Stiffness ({fe_loading})")
                    colors.append("#d62728")
                    hatches.append("//")

                n_models = len(models)
                x = np.arange(len(modes))
                width = 0.8 / n_models

                ax = fig.add_subplot(111)

                for i, (model_key, label, color, hatch) in enumerate(
                    zip(models, model_labels, colors, hatches)
                ):
                    vals = []
                    for mode in modes:
                        if model_key == "fe":
                            if mode == fe_loading:
                                vals.append(fe_field.knockdown)
                            else:
                                vals.append(float("nan"))
                        else:
                            vals.append(emp[mode][model_key]["knockdown"])
                    bar_x = x + i * width - (n_models - 1) * width / 2
                    for bx, bv in zip(bar_x, vals):
                        if not np.isnan(bv):
                            ax.bar(bx, bv, width, color=color, hatch=hatch,
                                   edgecolor='white' if hatch is None else '0.3',
                                   label=label if bx == bar_x[0] else "")

                ax.set_xticks(x)
                ax.set_xticklabels([m.upper() for m in modes], fontsize=10)
                ax.set_ylabel("Knockdown Factor", fontsize=11)

                # Build layup string for title
                layup_str = self.layup_edit.text().strip()
                ax.set_title(
                    f"Knockdown Factor by Loading Mode  |  "
                    f"Vp = {cfg['Vp']:.1f}%, {cfg['void_shape']}, "
                    f"{cfg['distribution']}, {layup_str}",
                    fontsize=11, fontweight="bold",
                )
                ax.set_ylim(0, 1.1)
                ax.legend(fontsize=8, loc="lower left")
                ax.grid(True, alpha=0.3, axis="y")

                # Footnote with model basis info
                note = ("Solid bars = strength knockdown (at mean Vp); "
                        "hatched bar = stiffness knockdown (FE)")
                if f_md < 0.49:
                    note += f"\nLayup scaling: f_md = {f_md:.2f} (coefficients reduced for fiber-dominated layup)"
                ax.text(
                    0.01, 0.01, note,
                    transform=ax.transAxes,
                    fontsize=7, color="0.4", va="bottom",
                )

                fig.tight_layout()
            except Exception as e:
                logger.exception("Results plot failed")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Results plot error:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="red")
                ax.set_axis_off()

            self.results_canvas.draw()

        def _on_stress_component_changed(self, _index: int) -> None:
            """Re-draw the stress contour when the component dropdown changes."""
            if self._result is not None and "fe_field" in self._result:
                import matplotlib.pyplot as plt
                self._plot_stress(self._result)
                plt.close("all")

        def _plot_stress(self, result: dict) -> None:
            """Draw FE stress contour at mid-y cross-section on Stress tab."""
            if not HAS_MPL_QT or self.stress_canvas is None:
                return

            fig = self.stress_canvas.figure
            fig.clear()

            fe_field = result.get("fe_field")
            if fe_field is None:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No FE results available.",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, color="0.5")
                ax.set_axis_off()
                self.stress_canvas.draw()
                return

            try:
                mesh = result["mesh"]
                stress_local = fe_field.stress_local  # (n_elem, n_gp, 6) material frame

                # Determine component index from dropdown label
                comp_name = "sigma_11"
                if self.stress_component_combo is not None:
                    comp_name = self.stress_component_combo.currentText()

                # Map display labels to component indices and colourbar labels
                _COMP_INDEX = {
                    "\u03c3\u2081\u2081 (fiber)": (0, r"$\sigma_{11}$ local (MPa)"),
                    "\u03c3\u2082\u2082 (transverse)": (1, r"$\sigma_{22}$ local (MPa)"),
                    "\u03c3\u2083\u2083 (through-thickness)": (2, r"$\sigma_{33}$ local (MPa)"),
                    "\u03c4\u2082\u2083 (interlaminar)": (3, r"$\tau_{23}$ local (MPa)"),
                    "\u03c4\u2081\u2083 (interlaminar)": (4, r"$\tau_{13}$ local (MPa)"),
                    "\u03c4\u2081\u2082 (in-plane shear)": (5, r"$\tau_{12}$ local (MPa)"),
                    "Von Mises": (-1, "Von Mises Stress (MPa)"),
                }
                comp_idx, label = _COMP_INDEX.get(comp_name, (0, comp_name + " (MPa)"))

                # Average GP stresses to element centres using local frame
                if comp_idx == -1:
                    # Von Mises from local stresses
                    s = stress_local.mean(axis=1)  # (n_elem, 6)
                    s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]
                    s4, s5, s6 = s[:, 3], s[:, 4], s[:, 5]
                    elem_stress = np.sqrt(0.5 * (
                        (s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2
                        + 6.0 * (s4**2 + s5**2 + s6**2)
                    ))
                else:
                    elem_stress = stress_local.mean(axis=1)[:, comp_idx]  # (n_elem,)

                # Extract elements at mid-y slice (j = ny // 2)
                ny_mid = mesh.ny // 2
                mid_elem_indices = []
                for k in range(mesh.nz):
                    for i in range(mesh.nx):
                        e_idx = k * mesh.ny * mesh.nx + ny_mid * mesh.nx + i
                        mid_elem_indices.append(e_idx)
                mid_elem_indices = np.array(mid_elem_indices)

                # Element centre coordinates for mid-y elements
                elem_nodes_coords = mesh.nodes[mesh.elements[mid_elem_indices]]  # (n_mid, 8, 3)
                cx = elem_nodes_coords[:, :, 0].mean(axis=1)
                cz = elem_nodes_coords[:, :, 2].mean(axis=1)

                # Exclude boundary elements: trim 10% from each end in x
                x_min_trim = mesh.L_x * 0.10
                x_max_trim = mesh.L_x * 0.90
                interior_mask = (cx > x_min_trim) & (cx < x_max_trim)
                mid_elem_indices = mid_elem_indices[interior_mask]
                cx = cx[interior_mask]
                cz = cz[interior_mask]

                sv = elem_stress[mid_elem_indices]

                ax = fig.add_subplot(111)
                # tricontourf needs at least 3 unique points
                finite_mask = np.isfinite(sv)
                if finite_mask.sum() >= 3:
                    vmin = np.percentile(sv[finite_mask], 5)
                    vmax = np.percentile(sv[finite_mask], 95)
                    tcf = ax.tricontourf(cx[finite_mask], cz[finite_mask],
                                        sv[finite_mask], levels=20, cmap="RdBu_r",
                                        vmin=vmin, vmax=vmax)
                    fig.colorbar(tcf, ax=ax, label=label)
                else:
                    ax.text(0.5, 0.5, "Insufficient interior data for contour plot.",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=10, color="0.5")
                ax.set_xlabel("x (mm)", fontsize=11)
                ax.set_ylabel("z (mm)", fontsize=11)
                ax.set_title(
                    f"FE Stress (local/material frame): {comp_name}  |  "
                    f"interior, mid-y cross-section",
                    fontsize=11, fontweight="bold",
                )
                ax.set_aspect("equal")
                fig.tight_layout()

            except Exception as e:
                logger.exception("Stress plot failed")
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Stress plot error:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="red")
                ax.set_axis_off()

            self.stress_canvas.draw()

        # ----------------------------------------------------------
        # Menu actions
        # ----------------------------------------------------------

        def _on_new(self) -> None:
            """Reset to defaults."""
            self.material_combo.setCurrentIndex(0)
            self.layup_edit.setText("[0/45/-45/90]_3s")
            self.ply_thickness_spin.setValue(0.183)
            self.vp_spin.setValue(3.0)
            self.distribution_combo.setCurrentIndex(0)
            self.void_shape_combo.setCurrentIndex(0)
            self.loading_combo.setCurrentIndex(0)
            self.nx_spin.setValue(30)
            self.ny_spin.setValue(10)
            self.nz_spin.setValue(12)
            self.output_text.clear()
            self._result = None
            self.statusBar().showMessage("Ready")

            # Clear plots
            if HAS_MPL_QT and self.profile_canvas is not None:
                for canvas, msg in [
                    (self.profile_canvas, "Run an analysis to see porosity profile plots."),
                    (self.mesh_canvas, "Run an analysis to see mesh visualization."),
                    (self.results_canvas, "Run an analysis to see knockdown results."),
                    (self.stress_canvas, "Run an analysis to see FE stress contours."),
                ]:
                    canvas.figure.clear()
                    ax = canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                            ha="center", va="center", fontsize=12, color="0.5")
                    ax.set_axis_off()
                    canvas.draw()

        def _on_export(self) -> None:
            """Export results to JSON or CSV."""
            if self._result is None:
                QMessageBox.information(
                    self, "No Results",
                    "Run an analysis first before exporting."
                )
                return

            filepath, selected_filter = QFileDialog.getSaveFileName(
                self, "Export Results", "porosity_results.json",
                "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
            )
            if not filepath:
                return
            try:
                payload = build_export_payload(self._result)
                # Pick the format from the file extension, falling back to the
                # selected filter (so a user who types "results.csv" gets CSV
                # even when the JSON filter is active).
                lower = filepath.lower()
                if lower.endswith(".csv"):
                    write_results_csv(filepath, payload)
                elif lower.endswith(".json"):
                    write_results_json(filepath, payload)
                elif "csv" in (selected_filter or "").lower():
                    write_results_csv(filepath, payload)
                else:
                    write_results_json(filepath, payload)

                self.statusBar().showMessage(f"Results exported to {filepath}")
            except Exception as e:
                logger.exception("Export failed for %s", filepath)
                QMessageBox.critical(self, "Export Error", str(e))

        def _on_about(self) -> None:
            """Show about dialog."""
            QMessageBox.about(
                self, "About PorosityFE",
                "<h3>PorosityFE v1.0.0</h3>"
                "<p>Porosity defect analysis for composite laminates.</p>"
                "<p>Evaluates strength knockdown from distributed porosity "
                "using empirical models (Judd-Wright, Power Law, Linear) "
                "and finite element analysis.</p>"
                "<p><b>References:</b></p>"
                "<ul>"
                "<li>Judd & Wright - Empirical porosity-strength relations</li>"
                "<li>Tsai-Wu - 3D failure criterion</li>"
                "</ul>"
            )


# ======================================================================
# Launch function
# ======================================================================

def launch() -> None:
    """Launch the PorosityFE GUI application.

    Raises
    ------
    ImportError
        If PyQt6 is not installed.
    """
    _check_pyqt6()

    app = QApplication.instance()
    standalone = app is None
    if standalone:
        app = QApplication(sys.argv)

    window = PorosityFEMainWindow()
    window.show()

    if standalone:
        sys.exit(app.exec())


def _console_main() -> int:
    """Console-script wrapper around ``launch()``.

    Catches the friendly ``ImportError`` from ``_check_pyqt6()`` and prints
    it to stderr instead of leaking a Python traceback when a user runs
    the ``porosity-fe`` command without the ``[gui]`` extra installed.
    """
    try:
        launch()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_console_main())
