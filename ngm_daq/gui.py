"""PyQt6 front end for FastLog captures with an R&S NGM201 / NGM202 over LAN."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import psutil
import pyqtgraph as pg
from PyQt6.QtCore import QSettings, QTimer, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QProgressDialog, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSplitter, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from . import __version__, export
from .capture import (
    PYRAMID_BASE, PYRAMID_FACTOR, QUANTITIES, UNITS, Capture, RunConfig, Stats, TriggerMode, merge_intervals,
    window_stats,
)
from .instrument import (
    CHUNK_PERIOD, CURRENT_MIN, CURRENT_RANGES, MAX_TOTAL_RATE, PROTOCOLS, VOLTAGE_MAX, VOLTAGE_RANGES,
    ChannelSetup, ChannelStatus, DeviceInfo, SampleRate, connection_for, format_rate, max_current, range_label,
)
from .plotting import STYLES, Bands, EnvelopeCurve
from .worker import InstrumentWorker

APP_NAME = "R&S NGM FastLog"
HISTORY_LENGTH = 10
CHANNEL_COLORS = {1: "#ff9f1a", 2: "#3fa9f5"}     # the instrument's Ch1 orange and Ch2 blue
QUANTITY_LABELS = {"voltage": "Voltage", "current": "Current", "power": "Power"}
MISSING_COLOR = (255, 70, 70, 90)                  # translucent red: samples lost on the way
CLIPPED_COLOR = (255, 200, 0, 70)                  # translucent amber: readings beyond the range
EVENT_CONTEXT = 0.002                              # s: minimum view width when jumping to an event
MEMORY_FRACTION = 0.5                              # of available RAM that captures may use
BYTES_PER_SAMPLE = 8 + 3 * 2 * 4 / PYRAMID_BASE * PYRAMID_FACTOR / (PYRAMID_FACTOR - 1)   # V, I + pyramids
DEFAULT_DURATION = 2.0


def si(value: float, unit: str, precision: int = 5) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return pg.siFormat(value, precision=precision, suffix=unit)


def format_duration(seconds: float) -> str:
    if seconds < 1000:
        return f"{seconds:.4g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    return f"{seconds / 3600:.3g} h"


def colored(text: str, ch: int) -> str:
    return f"<span style='color:{CHANNEL_COLORS[ch]}; font-weight:bold'>{text}</span>"


class ChannelPanel(QGroupBox):
    """Setpoints, output switch, live readback and measurement ranges of one channel."""

    voltage_requested = Signal(int, float)
    current_requested = Signal(int, float)
    output_requested = Signal(int, bool)
    ranges_requested = Signal(int)
    measure_changed = Signal()

    def __init__(self, ch: int):
        super().__init__(f"Channel {ch}")
        self.ch = ch
        self.setStyleSheet(f"QGroupBox::title {{ color: {CHANNEL_COLORS[ch]}; font-weight: bold; }}")
        self._status: ChannelStatus | None = None
        self._connected = False
        self._running = False
        self._has_ranges = True
        form = QFormLayout(self)

        self.measure = QCheckBox("Measure")
        self.measure.setToolTip("Include this channel in FastLog captures")
        self.measure.toggled.connect(self.measure_changed)
        self.output_btn = QPushButton("Output off")
        self.output_btn.setCheckable(True)
        self.output_btn.setToolTip("Switch this channel's output")
        self.output_btn.clicked.connect(lambda on: self.output_requested.emit(self.ch, on))
        row = QHBoxLayout()
        row.addWidget(self.measure)
        row.addStretch(1)
        row.addWidget(self.output_btn)
        form.addRow(row)

        self.voltage = self._spin(" V", 3, 0.0, VOLTAGE_MAX, 0.01)
        self.voltage.setToolTip("Voltage setpoint; sent when you press Enter or leave the field")
        self.voltage.editingFinished.connect(self._on_voltage)
        form.addRow("Voltage", self.voltage)
        self.current = self._spin(" A", 4, CURRENT_MIN, max_current(0), 0.001)
        self.current.setToolTip("Current limit; sent when you press Enter or leave the field.\n"
                                "Up to 6.01 A at 6 V or less, 3.01 A above.")
        self.current.editingFinished.connect(self._on_current)
        form.addRow("Current limit", self.current)

        self.readout = QLabel("—")
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        font = self.readout.font()
        font.setPointSizeF(font.pointSizeF() * 1.1)
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamily("monospace")
        self.readout.setFont(font)
        # Reserve the widest readout now: the control panel's width is fixed when the window is built.
        self.readout.setMinimumWidth(self.readout.fontMetrics().horizontalAdvance(
            "-20.0050 mV  -6.0100 mA  -120.20 mW  CC") + 4)
        form.addRow(self.readout)

        self.voltage_range = QComboBox()
        self.voltage_range.addItem("Auto", None)
        for r in VOLTAGE_RANGES:
            self.voltage_range.addItem(range_label(r, "V"), r)
        self.current_range = QComboBox()
        self.current_range.addItem("Auto", None)
        for r in CURRENT_RANGES:
            self.current_range.addItem(range_label(r, "A"), r)
        tip = ("Measurement range. A fixed range gives the best accuracy and lets the app flag readings\n"
               "beyond it (they clip at about 1.13 × range). Auto switches ranges as the signal changes;\n"
               "readings during a switch (~20 µs) can be clipped.")
        for combo in (self.voltage_range, self.current_range):
            combo.setToolTip(tip)
            combo.activated.connect(lambda _: self.ranges_requested.emit(self.ch))
        row = QHBoxLayout()
        row.addWidget(self.voltage_range, 1)
        row.addWidget(self.current_range, 1)
        form.addRow("Ranges V / I", row)

    @staticmethod
    def _spin(suffix: str, decimals: int, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        spin.setAccelerated(True)
        return spin

    # --- State ---------------------------------------------------------------------------

    def setup(self) -> ChannelSetup:
        return ChannelSetup(self.ch, self.voltage_range.currentData(), self.current_range.currentData())

    def select_ranges(self, voltage_range: float | None, current_range: float | None) -> None:
        for combo, value in ((self.voltage_range, voltage_range), (self.current_range, current_range)):
            index = next((i for i in range(combo.count())
                          if (combo.itemData(i) is None and value is None)
                          or (combo.itemData(i) is not None and value is not None
                              and math.isclose(combo.itemData(i), value))), 0)
            combo.setCurrentIndex(index)

    def set_connected(self, connected: bool, has_ranges: bool = True) -> None:
        self._connected = connected
        self._has_ranges = has_ranges
        if not connected:
            self._status = None
            self.readout.setText("—")
            self.output_btn.setChecked(False)
            self._style_output(False)
        self._update_enabled()

    def set_running(self, running: bool) -> None:
        self._running = running
        self._update_enabled()

    def _update_enabled(self) -> None:
        live = self._connected
        for w in (self.voltage, self.current, self.output_btn):
            w.setEnabled(live)
        self.measure.setEnabled(not self._running)
        for combo in (self.voltage_range, self.current_range):
            combo.setEnabled(not self._running and self._has_ranges)

    def update_status(self, s: ChannelStatus) -> None:
        self._status = s
        for spin, value in ((self.voltage, s.voltage_set), (self.current, s.current_set)):
            if not spin.hasFocus():
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self.current.setMaximum(max_current(s.voltage_set))
        self.output_btn.setChecked(s.output)
        self._style_output(s.output)
        if s.trips:
            text = "⚠ " + " ".join(s.trips) + " tripped"
        elif not s.output:
            text = "Output off"
        else:
            power = s.voltage * s.current
            text = f"{si(s.voltage, 'V', 6)}  {si(s.current, 'A', 5)}  {si(power, 'W', 4)}  {s.mode}"
        self.readout.setText(text)
        self.readout.setStyleSheet("QLabel { color: #c62828; }" if s.trips else "")

    def _style_output(self, on: bool) -> None:
        self.output_btn.setText("Output on" if on else "Output off")
        self.output_btn.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white; font-weight: bold; }" if on else "")

    def _on_voltage(self) -> None:
        if self._connected and (self._status is None or not math.isclose(self.voltage.value(),
                                                                           self._status.voltage_set)):
            self.voltage_requested.emit(self.ch, self.voltage.value())

    def _on_current(self) -> None:
        if self._connected and (self._status is None or not math.isclose(self.current.value(),
                                                                           self._status.current_set)):
            self.current_requested.emit(self.ch, self.current.value())


class MainWindow(QMainWindow):
    def __init__(self, resource: str | None = None, protocol: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.settings = QSettings("EmbedWise", "NGMDAQ")
        self.info: DeviceInfo | None = None
        self.capture: Capture | None = None       # capture on display
        self.last_complete: Capture | None = None
        self._busy = False                        # scanning, connecting, restarting
        self._connecting = False                  # connecting, or restarting and reconnecting
        self._restarting = False
        self._ranges_synced: set[int] = set()
        self._running = False
        self._stopping = False
        self._armed = False
        self._capturing = False                   # the capture on display is still filling
        self._run_captures = 0
        self._new_run = False
        self._events = np.empty((0, 2))           # highlighted intervals (s), for ◀ ▶
        self._event_nav: tuple[int, tuple[float, float]] | None = None

        self._build_ui()
        self.worker = InstrumentWorker()
        self._connect_worker()
        self._restore_settings(resource, protocol)
        self._update_summary()
        self._update_controls()

    # --- Construction --------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls = self._build_controls()
        scroll.setWidget(controls)
        scroll.setMinimumWidth(controls.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(self._build_plot())
        self.splitter.setCollapsible(0, False)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([scroll.minimumWidth(), 1100])
        self.setCentralWidget(self.splitter)
        self.resize(1440, 900)

        shortcuts = [
            (QKeySequence("F5"), self._on_run),
            (QKeySequence("F6"), self._on_trigger),
            (QKeySequence(QKeySequence.StandardKey.ZoomIn), lambda: self._zoom(0.5)),
            (QKeySequence(QKeySequence.StandardKey.ZoomOut), lambda: self._zoom(2.0)),
            (QKeySequence("Ctrl+0"), self._fit),
            (QKeySequence(QKeySequence.StandardKey.Save), lambda: self._save(view_only=True)),
        ]
        for key, slot in shortcuts:
            QShortcut(key, self).activated.connect(slot)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        box = QGroupBox("Instrument")
        form = QFormLayout(box)
        self.address = QComboBox()
        self.address.setEditable(True)
        self.address.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.address.lineEdit().setPlaceholderText("IP address, hostname, or VISA resource")
        self.address.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.address.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.address.setMinimumContentsLength(15)
        self.address.view().setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.address.editTextChanged.connect(self._update_controls)
        form.addRow("Address", self.address)
        self.protocol = QComboBox()
        self.protocol.addItems(PROTOCOLS)
        self.protocol.setToolTip("Used when the address is an IP or hostname.\n"
                                 "Socket (port 5025) needs no VISA installation and is the fastest.")
        form.addRow("Protocol", self.protocol)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.setToolTip("Discover LAN instruments (VXI-11 broadcast and mDNS)")
        self.scan_btn.clicked.connect(self._on_scan)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.restart_btn = QToolButton()
        self.restart_btn.setText("Restart…")
        self.restart_btn.setToolTip("Reboot the instrument (about 40 s) and reconnect.\n"
                                    "Needed after FastLog reports an internal error.")
        self.restart_btn.clicked.connect(self._on_restart)
        row = QHBoxLayout()
        row.addWidget(self.scan_btn)
        row.addWidget(self.connect_btn, 1)
        row.addWidget(self.restart_btn)
        form.addRow(row)
        self.device_label = QLabel("Not connected")
        self.device_label.setWordWrap(True)
        self.device_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.device_label)
        layout.addWidget(box)

        self.channel_panels: dict[int, ChannelPanel] = {}
        for ch in (1, 2):
            p = ChannelPanel(ch)
            p.voltage_requested.connect(lambda c, v: self.worker.set_voltage(c, v))
            p.current_requested.connect(lambda c, a: self.worker.set_current(c, a))
            p.output_requested.connect(self._on_output_requested)
            p.ranges_requested.connect(self._on_ranges_requested)
            p.measure_changed.connect(self._on_measure_changed)
            self.channel_panels[ch] = p
            layout.addWidget(p)

        box = QGroupBox("Acquisition")
        form = QFormLayout(box)
        self.rate = QComboBox()
        for rate in SampleRate:
            self.rate.addItem(rate.label, rate)
        self.rate.setToolTip(f"FastLog sample rate per channel. Both channels together are limited to "
                             f"{format_rate(MAX_TOTAL_RATE)}.")
        self.rate.currentIndexChanged.connect(self._update_summary)
        form.addRow("Sample rate", self.rate)
        self.duration = QDoubleSpinBox()
        self.duration.setDecimals(2)
        self.duration.setMinimum(CHUNK_PERIOD)
        self.duration.setSuffix(" s")
        self.duration.setAccelerated(True)
        self.duration.valueChanged.connect(self._update_summary)
        form.addRow("Acquisition time", self.duration)
        self.trigger_mode = QComboBox()
        for mode in TriggerMode:
            self.trigger_mode.addItem(mode.value, mode)
        self.trigger_mode.setToolTip(
            "Single: Run takes one capture. Stop early to keep what was recorded.\n"
            "Auto: captures follow each other seamlessly until Stop; the last complete one is kept.\n"
            "Manual: Run arms FastLog; each Trigger press takes one capture, then it re-arms until Stop.")
        self.trigger_mode.currentIndexChanged.connect(self._update_controls)
        self.trigger_mode.currentIndexChanged.connect(self._update_summary)
        form.addRow("Trigger mode", self.trigger_mode)
        self.output_on_start = QCheckBox("Switch outputs on after start")
        self.output_on_start.setToolTip(
            "Once FastLog delivers its first data, switch on the outputs of the measured channels\n"
            "(those that are off). The capture then shows about 0.6 s of baseline, the turn-on and the inrush.")
        form.addRow(self.output_on_start)
        self.output_off_end = QCheckBox("Switch outputs off at the end")
        self.output_off_end.setToolTip("Switch the measured channels' outputs off when the run ends.")
        form.addRow(self.output_off_end)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        self.run_btn = self._big_button("Run", "Configure FastLog and start acquiring (F5)", self._on_run)
        self.trigger_btn = self._big_button("Trigger", "Start a capture now: Manual mode, while armed (F6)",
                                            self._on_trigger)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.trigger_btn)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _big_button(text: str, tip: str, slot) -> QPushButton:
        btn = QPushButton(text)
        btn.setMinimumHeight(44)
        font = btn.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        font.setBold(True)
        btn.setFont(font)
        btn.setToolTip(tip)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _group(*widgets: QWidget) -> QWidget:
        """Keep related toolbar controls together, also when they move to the overflow menu."""
        group = QWidget()
        row = QHBoxLayout(group)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for widget in widgets:
            row.addWidget(widget)
        return group

    @staticmethod
    def _tool_button(text: str, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tip)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        return btn

    def _build_plot(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        bar = QToolBar()
        for text, tip, slot in (
                ("Zoom In", "Zoom in on time; Y fits the visible data (Ctrl++)", lambda: self._zoom(0.5)),
                ("Zoom Out", "Zoom out on time; Y fits the visible data (Ctrl+-)", lambda: self._zoom(2.0)),
                ("Fit", "Show all data (Ctrl+0)", self._fit)):
            btn = self._tool_button(text, tip)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        self.box_zoom_btn = self._tool_button("Box Zoom", "Drag a rectangle to zoom. When off, drag pans; "
                                                          "wheel zooms; right-drag scales.")
        self.box_zoom_btn.setCheckable(True)
        self.box_zoom_btn.toggled.connect(self._on_box_zoom)
        bar.addWidget(self.box_zoom_btn)
        bar.addSeparator()
        self.plot_style = QComboBox()
        self.plot_style.addItems(STYLES)
        self.plot_style.setToolTip(
            "Line + points: markers appear once you zoom in far enough to tell samples apart.\n"
            "Points: markers only. Zoomed out, each marker is the min or max of a group of samples.")
        self.plot_style.currentIndexChanged.connect(self._apply_plot_style)
        bar.addWidget(self._group(QLabel("Style"), self.plot_style))
        bar.addSeparator()
        self.show_quantity: dict[str, QCheckBox] = {}
        for q, label in (("voltage", "V"), ("current", "I"), ("power", "P")):
            cb = QCheckBox(label)
            cb.setToolTip(f"Show the {QUANTITY_LABELS[q].lower()} plot")
            cb.setChecked(True)
            cb.toggled.connect(self._arrange_plots)
            self.show_quantity[q] = cb
        bar.addWidget(self._group(QLabel("Show"), *self.show_quantity.values()))
        bar.addSeparator()
        self.show_missing = QCheckBox("Gaps")
        self.show_missing.setToolTip(
            "Shade samples that never reached the PC in red (the instrument replaced a chunk before it\n"
            "was read). They are stored as NaN. Bands are at least 3 px wide, so a gap stays visible.")
        self.show_clipped = QCheckBox("Clipped")
        self.show_clipped.setToolTip(
            "Shade readings beyond the measurement range in amber (the reading is pinned at ~1.13 × range,\n"
            "so the true value is unknown). Choose a higher range to avoid them.")
        for cb in (self.show_missing, self.show_clipped):
            cb.setChecked(True)
            cb.toggled.connect(self._refresh_bands)
        self.prev_event_btn = QToolButton()
        self.prev_event_btn.setArrowType(Qt.ArrowType.LeftArrow)
        self.prev_event_btn.setToolTip("Previous highlighted gap or clipped run")
        self.prev_event_btn.clicked.connect(lambda: self._goto_event(-1))
        self.next_event_btn = QToolButton()
        self.next_event_btn.setArrowType(Qt.ArrowType.RightArrow)
        self.next_event_btn.setToolTip("Next highlighted gap or clipped run")
        self.next_event_btn.clicked.connect(lambda: self._goto_event(1))
        for btn in (self.prev_event_btn, self.next_event_btn):
            btn.setAutoRaise(True)
        bar.addWidget(self._group(QLabel("Highlight"), self.show_missing, self.show_clipped,
                                  self.prev_event_btn, self.next_event_btn))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        format_row = [QLabel("Format")]
        self.export_format = QButtonGroup(self)
        tips = {
            "CSV": "Text: time, then voltage, current and flags per channel, after a # metadata header.\n"
                   "Opens anywhere; large and slow to write for long captures.",
            "HDF5": "Binary, lossless, compressed: float32 voltage and current per channel, metadata as\n"
                    "attributes, and the gap and clipped runs. Time is t0_s + n × sample_interval_s.",
        }
        for fmt in export.FORMATS:
            button = QRadioButton(fmt)
            button.setToolTip(tips[fmt])
            self.export_format.addButton(button)
            format_row.append(button)
        bar.addWidget(self._group(*format_row))
        bar.addSeparator()
        self.save_view_btn = self._tool_button("Save View…", "Save the samples in the displayed time window (Ctrl+S)")
        self.save_view_btn.clicked.connect(lambda: self._save(view_only=True))
        bar.addWidget(self.save_view_btn)
        self.save_all_btn = self._tool_button("Save All…", "Save the complete capture")
        self.save_all_btn.clicked.connect(lambda: self._save(view_only=False))
        bar.addWidget(self.save_all_btn)
        layout.addWidget(bar)

        self.warning_label = QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("QLabel { background: #fff3cd; color: #664d03; padding: 4px; }")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        pg.setConfigOptions(antialias=False)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plots: dict[str, pg.PlotItem] = {}
        self.curves: dict[tuple[str, int], EnvelopeCurve] = {}
        self.missing_bands: dict[str, Bands] = {}
        self.clipped_bands: dict[str, Bands] = {}
        for q in QUANTITIES:
            p = pg.PlotItem()
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("left", QUANTITY_LABELS[q], units=UNITS[q])
            p.getViewBox().setAutoVisible(y=True)
            p.hideButtons()
            p.setMenuEnabled(True)
            for ch, color in CHANNEL_COLORS.items():
                curve = EnvelopeCurve(color)
                p.addItem(curve)
                self.curves[q, ch] = curve
            self.missing_bands[q] = Bands(MISSING_COLOR, z=-9)
            self.clipped_bands[q] = Bands(CLIPPED_COLOR, z=-10)
            p.addItem(self.missing_bands[q], ignoreBounds=True)
            p.addItem(self.clipped_bands[q], ignoreBounds=True)
            vb = p.getViewBox()
            vb.sigXRangeChanged.connect(lambda *_: self._stats_timer.start())
            vb.sigXRangeChanged.connect(self._update_event_buttons)
            self.plots[q] = p
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        layout.addWidget(self.plot_widget, 1)

        info = QHBoxLayout()
        self.stats_label = QLabel()
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats_label.setWordWrap(True)
        self.cursor_label = QLabel()
        self.cursor_label.setTextFormat(Qt.TextFormat.RichText)
        self.cursor_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        info.addWidget(self.stats_label, 1)
        info.addWidget(self.cursor_label)
        layout.addLayout(info)

        self._stats_timer = QTimer(self, singleShot=True, interval=150)
        self._stats_timer.timeout.connect(self._update_stats)
        self._arrange_plots()
        return panel

    def _connect_worker(self) -> None:
        w = self.worker
        w.scanned.connect(self._on_scanned)
        w.connected.connect(self._on_connected)
        w.disconnected.connect(self._on_disconnected)
        w.status.connect(self._on_status)
        w.armed.connect(self._on_armed)
        w.capture_started.connect(self._on_capture_started)
        w.progress.connect(self._on_progress)
        w.acquired.connect(self._on_acquired)
        w.run_finished.connect(self._on_run_finished)
        w.message.connect(lambda text: self.statusBar().showMessage(text, 10000))
        w.failed.connect(self._on_failed)

    # --- Settings ------------------------------------------------------------------------

    def _restore_settings(self, resource: str | None, protocol: str | None) -> None:
        s = self.settings
        history = s.value("history", [], list)
        self.address.addItems(history)
        self.address.setEditText(resource or (history[0] if history else ""))
        self.protocol.setCurrentText(protocol or s.value("protocol", PROTOCOLS[0], str))
        for ch, panel in self.channel_panels.items():
            panel.measure.setChecked(s.value(f"ch{ch}/measure", ch == 1, bool))
            v = s.value(f"ch{ch}/voltage_range", "auto", str)
            i = s.value(f"ch{ch}/current_range", "auto", str)
            panel.select_ranges(None if v == "auto" else float(v), None if i == "auto" else float(i))
        rate = s.value("rate", SampleRate.R50K.name, str)
        self.rate.setCurrentIndex(self.rate.findData(SampleRate.__members__.get(rate, SampleRate.R50K)))
        mode = TriggerMode.__members__.get(s.value("trigger_mode", "SINGLE", str), TriggerMode.SINGLE)
        self.trigger_mode.setCurrentIndex(self.trigger_mode.findData(mode))
        self.output_on_start.setChecked(s.value("output_on_start", False, bool))
        self.output_off_end.setChecked(s.value("output_off_end", False, bool))
        self.plot_style.setCurrentIndex(s.value("plot_style", 0, int))
        for q, cb in self.show_quantity.items():
            cb.setChecked(s.value(f"show_{q}", True, bool))
        self.show_missing.setChecked(s.value("show_missing", True, bool))
        self.show_clipped.setChecked(s.value("show_clipped", True, bool))
        fmt = s.value("export_format", "CSV", str)
        for button in self.export_format.buttons():
            button.setChecked(button.text() == fmt)
        if self.export_format.checkedButton() is None:
            self.export_format.buttons()[0].setChecked(True)
        self._update_summary()                     # sets the duration limit before the value
        self.duration.setValue(s.value("duration_s", DEFAULT_DURATION, float))
        if geometry := s.value("geometry"):
            self.restoreGeometry(geometry)
        if splitter := s.value("splitter"):
            self.splitter.restoreState(splitter)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("protocol", self.protocol.currentText())
        for ch, panel in self.channel_panels.items():
            setup = panel.setup()
            s.setValue(f"ch{ch}/measure", panel.measure.isChecked())
            s.setValue(f"ch{ch}/voltage_range", "auto" if setup.voltage_range is None else setup.voltage_range)
            s.setValue(f"ch{ch}/current_range", "auto" if setup.current_range is None else setup.current_range)
        s.setValue("rate", self.rate.currentData().name)
        s.setValue("duration_s", self.duration.value())
        s.setValue("trigger_mode", self.trigger_mode.currentData().name)
        s.setValue("output_on_start", self.output_on_start.isChecked())
        s.setValue("output_off_end", self.output_off_end.isChecked())
        s.setValue("plot_style", self.plot_style.currentIndex())
        for q, cb in self.show_quantity.items():
            s.setValue(f"show_{q}", cb.isChecked())
        s.setValue("show_missing", self.show_missing.isChecked())
        s.setValue("show_clipped", self.show_clipped.isChecked())
        s.setValue("export_format", self.export_format.checkedButton().text())
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self.splitter.saveState())

    def _remember_address(self, text: str) -> None:
        history = [text] + [h for h in self.settings.value("history", [], list) if h != text]
        self.settings.setValue("history", history[:HISTORY_LENGTH])

    # --- State ---------------------------------------------------------------------------

    def measured_channels(self) -> list[int]:
        available = self.info.channels if self.info else (1, 2)
        return [ch for ch, p in self.channel_panels.items() if p.measure.isChecked() and ch in available]

    def current_config(self) -> RunConfig:
        return RunConfig(
            channels=tuple(self.channel_panels[ch].setup() for ch in self.measured_channels()),
            rate=self.rate.currentData(),
            duration=self.duration.value(),
            mode=self.trigger_mode.currentData(),
            output_on_at_start=self.output_on_start.isChecked(),
            output_off_at_end=self.output_off_end.isChecked())

    def _update_controls(self) -> None:
        connected = self.info is not None
        idle = not self._busy and not self._running
        self.address.setEnabled(idle and not connected)
        self.scan_btn.setEnabled(idle and not connected)
        self.protocol.setEnabled(idle and not connected and "::" not in self.address.currentText())
        has_address = bool(self.address.currentText().strip())
        if self._connecting:
            self.connect_btn.setText("Cancel")
            self.connect_btn.setEnabled(True)
        else:
            self.connect_btn.setText("Disconnect" if connected else "Connect")
            self.connect_btn.setEnabled(not self._running and not self._busy and (connected or has_address))
        self.restart_btn.setEnabled(connected and idle)
        for ch, panel in self.channel_panels.items():
            present = self.info is None or ch in self.info.channels
            panel.setVisible(present)
            panel.set_running(self._running)
        for w in (self.rate, self.duration, self.trigger_mode, self.output_on_start, self.output_off_end):
            w.setEnabled(not self._running)
        self.run_btn.setText("Stop" if self._running else "Run")
        can_run = connected and not self._busy and bool(self.measured_channels())
        self.run_btn.setEnabled((self._running and not self._stopping) or (not self._running and can_run))
        self.trigger_btn.setEnabled(self._armed)
        has_data = self.capture is not None and self.capture.n > 0 and not self._capturing
        self.save_view_btn.setEnabled(has_data)
        self.save_all_btn.setEnabled(has_data)
        self._update_event_buttons()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_controls()

    def _max_duration(self, rate: SampleRate, channels: int, mode: TriggerMode) -> float:
        """Longest capture that fits the memory budget (two captures may be in memory at once)."""
        budget = psutil.virtual_memory().available * MEMORY_FRACTION
        samples = budget / (BYTES_PER_SAMPLE * max(1, channels) * 2)
        return max(CHUNK_PERIOD, math.floor(samples / rate.hz * 100) / 100)

    def _update_summary(self) -> None:
        channels = max(1, len(self.measured_channels()))
        model = self.rate.model()
        for i in range(self.rate.count()):
            model.item(i).setEnabled(self.rate.itemData(i).hz * channels <= MAX_TOTAL_RATE)
        rate: SampleRate = self.rate.currentData()
        if rate.hz * channels > MAX_TOTAL_RATE:
            rate = SampleRate.R250K
            self.rate.setCurrentIndex(self.rate.findData(rate))
        mode = self.trigger_mode.currentData()
        max_duration = self._max_duration(rate, channels, mode)
        self.duration.setMaximum(max_duration)
        samples = max(1, round(self.duration.value() * rate.hz))
        size = samples * BYTES_PER_SAMPLE * channels
        link = rate.hz * 8 * channels
        self.summary.setText(
            f"{samples:,} samples per channel · {si(size, 'B', 3)} in memory\n"
            f"Link {si(link, 'B/s', 3)} · max {format_duration(max_duration)} (memory)")

    # --- Instrument panel ----------------------------------------------------------------

    def _on_scan(self) -> None:
        self._set_busy(True)
        self.statusBar().showMessage("Scanning for instruments…")
        self.worker.scan()

    @Slot(list)
    def _on_scanned(self, hosts: list) -> None:
        self._set_busy(False)
        current = self.address.currentText()
        existing = {self.address.itemText(i) for i in range(self.address.count())}
        self.address.addItems([h for h in hosts if h not in existing])
        self.address.setEditText(current or (hosts[0] if hosts else ""))
        if hosts:
            self.statusBar().showMessage(f"Found {len(hosts)} LAN instrument(s); pick one from the Address list.", 8000)
            self.address.showPopup()
        else:
            self.statusBar().showMessage("No instruments found. Enter the NGM's IP address (instrument: "
                                         "Settings > Device > Interfaces > Network > LAN).", 10000)

    def _on_connect(self) -> None:
        if self.info is not None or self._connecting:
            self.worker.request_stop()          # also ends waiting for a restart
            self._restarting = False
            self._connecting = False
            self._set_busy(True)
            self.worker.close_session()
            return
        text = self.address.currentText().strip()
        resource, visa = connection_for(text, self.protocol.currentText())
        self._connecting = True
        self._set_busy(True)
        self.statusBar().showMessage(f"Connecting to {resource}…")
        self._remember_address(text)
        self.worker.open_session(resource, visa)

    def _on_restart(self, confirmed: bool = False) -> None:
        if self.info is None or self._running:
            return
        if not confirmed and QMessageBox.question(
                self, "Restart instrument",
                f"Reboot the {self.info.model}? Its outputs switch off, and the app reconnects when it is back "
                "(about 40 s).") != QMessageBox.StandardButton.Yes:
            return
        self.worker.restart_instrument()
        self._restarting = self._connecting = True
        self._set_busy(True)
        self.statusBar().showMessage("Restarting the instrument…")

    @Slot(object)
    def _on_connected(self, info: DeviceInfo) -> None:
        self.info = info
        self._connecting = self._restarting = False
        self.device_label.setText(info.summary())
        self.device_label.setToolTip(
            f"{info.manufacturer} {info.model} ({info.part_number})\nSerial {info.serial}\n"
            f"Firmware {info.firmware}\nOptions {', '.join(sorted(info.options)) or 'none'}\n{info.resource}")
        self._ranges_synced = set()
        for ch, panel in self.channel_panels.items():
            panel.set_connected(ch in info.channels, info.has_ranges)
        msg = f"Connected to {info.model} at {info.resource}"
        if not info.has_ranges:
            msg += " — no K104 option: measurement ranges are automatic"
        self.statusBar().showMessage(msg, 10000)
        self._set_busy(False)
        self._update_summary()

    @Slot()
    def _on_disconnected(self) -> None:
        if self.info is not None:
            self.statusBar().showMessage("Disconnected", 5000)
        self.info = None
        self.device_label.setText("Restarting…" if self._restarting else "Not connected")
        self.device_label.setToolTip("")
        for panel in self.channel_panels.values():
            panel.set_connected(False)
        self._running = self._stopping = self._armed = False
        if not self._restarting:
            self._connecting = False
        self._set_busy(self._restarting)
        self._update_summary()

    @Slot(object)
    def _on_status(self, statuses: dict) -> None:
        for ch, status in statuses.items():
            panel = self.channel_panels[ch]
            panel.update_status(status)
            if ch not in self._ranges_synced and self.info is not None and self.info.has_ranges:
                # Show the instrument's ranges once after connecting; after that the panel drives them.
                self._ranges_synced.add(ch)
                panel.select_ranges(status.voltage_range, status.current_range)

    def _on_output_requested(self, ch: int, on: bool) -> None:
        self.worker.set_output(ch, on)

    def _on_ranges_requested(self, ch: int) -> None:
        if self.info is not None and not self._running:
            setup = self.channel_panels[ch].setup()
            self.worker.set_ranges(ch, setup.voltage_range, setup.current_range)

    def _on_measure_changed(self) -> None:
        self._update_summary()
        self._update_controls()

    # --- Acquisition ---------------------------------------------------------------------

    def _on_run(self) -> None:
        if self._running:
            if not self._stopping:
                self._stopping = True
                self.worker.request_stop()
                self.statusBar().showMessage("Stopping…")
            self._armed = False
            self._update_controls()
            return
        if self.info is None or self._busy or not self.measured_channels():
            return
        cfg = self.current_config()
        self._running = True
        self._run_captures = 0
        self._new_run = True
        self.last_complete = None
        self.warning_label.hide()
        self._set_progress(busy=True)
        self.statusBar().showMessage("Configuring FastLog…")
        self._update_controls()
        self.worker.run(cfg)

    def _on_trigger(self) -> None:
        if not self._armed:
            return
        self._armed = False
        self.worker.request_trigger()
        self.statusBar().showMessage("Triggered")
        self._update_controls()

    @Slot()
    def _on_armed(self) -> None:
        if self._stopping:
            return
        self._armed = True
        self._set_progress(text="Armed — press Trigger (F6)")
        self.statusBar().showMessage(f"Armed, waiting for trigger (captures so far: {self._run_captures})")
        self._update_controls()

    @Slot(object)
    def _on_capture_started(self, capture: Capture) -> None:
        self._armed = False
        self._capturing = True
        self._set_progress(maximum=capture.target)
        first = self._new_run
        self._new_run = False
        self._show(capture, fit=first)
        self.statusBar().showMessage(f"Capture {capture.number}: waiting for data…")
        self._update_controls()

    @Slot(object)
    def _on_progress(self, capture: Capture) -> None:
        if capture is not self.capture:
            return
        n = capture.n
        for curve in self.curves.values():
            curve.data_changed()
        self._refresh_bands()
        self.progress.setValue(n)
        self.statusBar().showMessage(
            f"Capture {capture.number}: {n:,} / {capture.target:,} samples ({format_duration(n * capture.interval)})")

    @Slot(object)
    def _on_acquired(self, capture: Capture) -> None:
        self._capturing = False
        if capture.complete:
            self._run_captures += 1
            self.last_complete = capture
        elif self._stopping and self.last_complete is not None:
            # Stopped part-way through: keep the last complete capture on display.
            self._show(self.last_complete)
            self.statusBar().showMessage(f"Stopped; showing capture {self.last_complete.number} "
                                         "(the interrupted capture was discarded)", 10000)
            return
        self._show(capture)
        n = capture.n
        self.statusBar().showMessage(f"Capture {capture.number}: {n:,} samples in {format_duration(n * capture.interval)}"
                                     f"{'' if capture.complete else ' (partial)'}", 10000)

    @Slot()
    def _on_run_finished(self) -> None:
        stopped = self._stopping
        self._running = self._stopping = self._armed = self._capturing = False
        if self.capture is not None:
            self._set_progress(maximum=max(1, self.capture.target), value=self.capture.n)
            self._show(self.capture)
        else:
            self._set_progress()
        if stopped and self.statusBar().currentMessage() == "Stopping…":
            self.statusBar().showMessage(f"Stopped after {self._run_captures} capture(s)", 10000)
        self._update_controls()

    @Slot(str, str, str)
    def _on_failed(self, title: str, message: str, kind: str) -> None:
        self.statusBar().showMessage(title, 10000)
        if self.info is None:
            self._connecting = self._restarting = False
            self._set_busy(False)
        if kind == "fastlog_stopped" and self.info is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, title, message, parent=self)
            restart = box.addButton("Restart Instrument", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is restart:
                QTimer.singleShot(0, lambda: self._on_restart(confirmed=True))
            return
        if self._running:
            message += "\n\nIf the instrument stays unresponsive, disconnect and reconnect."
        QMessageBox.critical(self, title, message)

    def _set_progress(self, maximum: int = 1, value: int = 0, busy: bool = False, text: str = "%p%") -> None:
        self.progress.setRange(0, 0 if busy else maximum)
        self.progress.setValue(value)
        self.progress.setFormat(text)

    # --- Display -------------------------------------------------------------------------

    def _show(self, capture: Capture, fit: bool = False) -> None:
        self.capture = capture
        dt = capture.interval
        for (q, ch), curve in self.curves.items():
            data = capture.channels.get(ch)
            curve.set_source(data, q, dt)
            curve.setVisible(data is not None)
        end = max(capture.target - 1, 1) * dt
        for p in self.plots.values():
            p.getViewBox().setLimits(xMin=0, xMax=end, minXRange=5 * dt)
        if fit:
            vb = self._visible_plots()[0].getViewBox()
            vb.setXRange(0, end, padding=0)
            for p in self.plots.values():
                p.getViewBox().enableAutoRange(y=True)
        self._update_title()
        self._refresh_bands()
        self._update_warnings()
        self._update_controls()
        self._update_stats()

    def _update_title(self) -> None:
        cap = self.capture
        if cap is None:
            return
        parts = []
        for ch, data in cap.channels.items():
            s = data.settings
            parts.append(f"{colored(f'CH{ch}', ch)} {s.voltage_set:g} V, {si(s.current_set, 'A', 3)} limit, "
                         f"ranges {range_label(s.voltage_range, 'V')} / {range_label(s.current_range, 'A')}")
        title = (f"{cap.device.model} · {format_rate(cap.sample_rate)} · " + " · ".join(parts)
                 + f" · #{cap.number} {cap.started:%Y-%m-%d %H:%M:%S}")
        for p in self.plots.values():
            p.setTitle(None)
        self._visible_plots()[0].setTitle(title, size="9pt")

    def _visible_plots(self) -> list[pg.PlotItem]:
        visible = [self.plots[q] for q in QUANTITIES if self.show_quantity[q].isChecked()]
        return visible or [self.plots["voltage"]]

    def _arrange_plots(self) -> None:
        """Lay out the selected plots top to bottom, linked in time, with the time axis on the last."""
        layout = self.plot_widget.ci
        x_range = self._visible_plots()[0].getViewBox().viewRange()[0] if layout.items else None
        for p in self.plots.values():
            if p in layout.items:
                layout.removeItem(p)
        visible = self._visible_plots()
        for row, p in enumerate(visible):
            layout.addItem(p, row=row, col=0)
            p.setXLink(visible[0] if row else None)
            last = row == len(visible) - 1
            p.setLabel("bottom", "Time" if last else None, units="s" if last else None)
            p.getAxis("bottom").setStyle(showValues=last)
        if x_range is not None:
            visible[0].getViewBox().setXRange(*x_range, padding=0)
        self._update_title()

    def _refresh_bands(self) -> None:
        cap = self.capture
        events = []
        for q in QUANTITIES:
            missing, clipped = [], []
            if cap is not None:
                for ch in cap.channels.values():
                    missing.append(ch.missing.array())
                    if q in ("voltage", "power"):
                        clipped.append(ch.voltage_clipped.array())
                    if q in ("current", "power"):
                        clipped.append(ch.current_clipped.array())
            m = merge_intervals(*missing) if self.show_missing.isChecked() else np.empty((0, 2))
            c = merge_intervals(*clipped) if self.show_clipped.isChecked() else np.empty((0, 2))
            dt = cap.interval if cap is not None else 1.0
            for bands, runs in ((self.missing_bands[q], m), (self.clipped_bands[q], c)):
                bands.set_intervals((runs[:, 0] - 0.5) * dt, (runs[:, 1] - 0.5) * dt)
            if self.show_quantity[q].isChecked():
                events += [m, c]
        merged = merge_intervals(*events) if cap is not None else np.empty((0, 2))
        self._events = merged * (cap.interval if cap is not None else 1.0)
        self._event_nav = None
        self._update_event_buttons()

    def _update_warnings(self) -> None:
        cap = self.capture
        warnings = list(cap.warnings)
        for ch, data in cap.channels.items():
            lost, runs = data.missing.count_in(0, data.n)
            if lost:
                warnings.append(
                    f"CH{ch}: {lost:,} samples ({format_duration(lost * cap.interval)}) never reached the PC, "
                    f"in {runs:,} gap{'s' if runs != 1 else ''}. The instrument replaces a chunk after 250 ms; "
                    "a busy PC or network delayed the read.")
            for name, runs_list, unit in (("voltage", data.voltage_clipped, "V"),
                                          ("current", data.current_clipped, "A")):
                count, runs = runs_list.count_in(0, data.n)
                if count:
                    rng = data.settings.voltage_range if unit == "V" else data.settings.current_range
                    hint = (f"choose a range above {range_label(rng, unit)}" if rng is not None
                            else "the range switched while the signal changed")
                    warnings.append(f"CH{ch}: {count:,} {name} readings clipped in {runs:,} run"
                                    f"{'s' if runs != 1 else ''} ({hint}).")
        if warnings:
            self.warning_label.setText("⚠ " + "\n⚠ ".join(warnings))
        self.warning_label.setVisible(bool(warnings))

    # --- Plot interaction ----------------------------------------------------------------

    def _view_range(self) -> tuple[float, float]:
        return tuple(self._visible_plots()[0].getViewBox().viewRange()[0])

    def _zoom(self, factor: float) -> None:
        vb = self._visible_plots()[0].getViewBox()
        x0, x1 = vb.viewRange()[0]
        center, half = (x0 + x1) / 2, (x1 - x0) * factor / 2
        vb.setXRange(center - half, center + half, padding=0)
        for p in self.plots.values():
            p.getViewBox().enableAutoRange(y=True)

    def _fit(self) -> None:
        if self.capture is None:
            return
        end = max(self.capture.n - 1, 1) * self.capture.interval
        self._visible_plots()[0].getViewBox().setXRange(0, end, padding=0.01)
        for p in self.plots.values():
            p.getViewBox().enableAutoRange(y=True)

    def _apply_plot_style(self) -> None:
        for curve in self.curves.values():
            curve.set_style(self.plot_style.currentText())

    def _on_box_zoom(self, enabled: bool) -> None:
        for p in self.plots.values():
            p.getViewBox().setMouseMode(pg.ViewBox.RectMode if enabled else pg.ViewBox.PanMode)

    def _visible_samples(self) -> tuple[int, int]:
        cap = self.capture
        x0, x1 = self._view_range()
        start = max(0, math.ceil(x0 / cap.interval - 1e-6))
        stop = min(cap.n, math.floor(x1 / cap.interval + 1e-6) + 1)
        return start, max(start, stop)

    def _target_event(self, step: int) -> int | None:
        """Index of the highlighted interval Previous (-1) or Next (+1) would jump to."""
        events = self._events
        if self.capture is None or self._capturing or len(events) == 0:
            return None
        dt = self.capture.interval
        x0, x1 = self._view_range()
        if self._event_nav is not None and np.allclose(self._event_nav[1], (x0, x1), rtol=0, atol=dt / 10):
            i = self._event_nav[0] + step
        elif x0 <= dt and x1 >= (self.capture.n - 2) * dt:
            i = 0 if step > 0 else len(events) - 1
        else:
            centers = events.mean(axis=1)
            mid = (x0 + x1) / 2
            if step > 0:
                i = int(np.searchsorted(centers, mid + dt / 2, side="right"))
            else:
                i = int(np.searchsorted(centers, mid - dt / 2, side="left")) - 1
        return i if 0 <= i < len(events) else None

    def _goto_event(self, step: int) -> None:
        i = self._target_event(step)
        if i is None:
            return
        dt = self.capture.interval
        a, b = self._events[i]
        vb = self._visible_plots()[0].getViewBox()
        x0, x1 = vb.viewRange()[0]
        target = max(3 * (b - a), EVENT_CONTEXT, 200 * dt)
        width = x1 - x0
        if not target <= width <= max(target, 0.1 * self.capture.n * dt):
            width = target
        center = (a + b) / 2 - dt / 2
        vb.setXRange(center - width / 2, center + width / 2, padding=0)
        for p in self.plots.values():
            p.getViewBox().enableAutoRange(y=True)
        self._event_nav = (i, tuple(vb.viewRange()[0]))
        self._update_event_buttons()
        n = round((b - a) / dt)
        self.statusBar().showMessage(
            f"Highlight {i + 1} of {len(self._events)}: {n:,} sample{'s' if n != 1 else ''} "
            f"({si(b - a, 's', 4)}) at {si(a, 's', 7)}", 10000)

    def _update_event_buttons(self, *_) -> None:
        self.prev_event_btn.setEnabled(self._target_event(-1) is not None)
        self.next_event_btn.setEnabled(self._target_event(1) is not None)

    def _update_stats(self) -> None:
        cap = self.capture
        if cap is None or self._capturing:
            return
        start, stop = self._visible_samples()
        if stop <= start:
            self.stats_label.setText("")
            return
        lines = [f"View {si((stop - start) * cap.interval, 's', 4)} · {stop - start:,} samples"]
        for ch, data in cap.channels.items():
            lines.append(self._stats_line(ch, window_stats(data, start, stop, cap.interval)))
        self.stats_label.setText("<br>".join(lines))

    @staticmethod
    def _stats_line(ch: int, s: Stats) -> str:
        if s.valid == 0:
            return f"{colored(f'CH{ch}', ch)} no valid samples"
        def q(name: str, unit: str) -> str:
            return (f"{si(s.mean[name], unit, 6)} (σ {si(s.std[name], unit, 3)}, "
                    f"{si(s.min[name], unit, 4)} … {si(s.max[name], unit, 4)})")
        text = (f"{colored(f'CH{ch}', ch)}&nbsp; V {q('voltage', 'V')} &nbsp; I {q('current', 'A')} &nbsp; "
                f"P {si(s.mean['power'], 'W', 5)} (max {si(s.max['power'], 'W', 4)}) &nbsp; "
                f"E {si(s.energy_j, 'J', 4)} = {si(s.energy_j / 3600, 'Wh', 4)} &nbsp; "
                f"Q {si(s.charge_c, 'C', 4)} = {si(s.charge_c / 3600, 'Ah', 4)}")
        flags = []
        if s.missing[0]:
            flags.append(f"{s.missing[0]:,} missing in {s.missing[1]:,} gaps")
        clipped = s.voltage_clipped[0] + s.current_clipped[0]
        if clipped:
            flags.append(f"{clipped:,} clipped")
        if flags:
            text += " &nbsp; ⚠ " + ", ".join(flags)
        return text

    def _on_mouse_moved(self, pos) -> None:
        cap = self.capture
        if cap is None:
            return
        for p in self._visible_plots():
            vb = p.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                t = vb.mapSceneToView(pos).x()
                break
        else:
            return
        i = round(t / cap.interval)
        if not 0 <= i < cap.n:
            return
        parts = [f"#{i:,} &nbsp; t = {si(i * cap.interval, 's', 7)}"]
        for ch, data in cap.channels.items():
            v, a = float(data.voltage[i]), float(data.current[i])
            if math.isnan(v):
                parts.append(f"{colored(f'CH{ch}', ch)} missing")
            else:
                parts.append(f"{colored(f'CH{ch}', ch)} {si(v, 'V', 6)} {si(a, 'A', 5)} {si(v * a, 'W', 4)}")
        self.cursor_label.setText("<br>".join(parts))

    # --- Saving --------------------------------------------------------------------------

    def _save(self, view_only: bool) -> None:
        cap = self.capture
        if cap is None or self._capturing or cap.n == 0:
            return
        start, stop = self._visible_samples() if view_only else (0, cap.n)
        if stop <= start:
            QMessageBox.information(self, "Nothing to save", "No samples are inside the displayed time window.")
            return
        fmt = self.export_format.checkedButton().text()
        suffixes = export.FORMATS[fmt]
        chans = "".join(f"CH{c}" for c in cap.channels)
        name = f"{cap.device.model}_{chans}_{cap.started:%Y%m%d-%H%M%S}" + ("_view" if view_only else "") + suffixes[0]
        directory = self.settings.value("save_dir", str(Path.home()), str)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save displayed data" if view_only else "Save all data",
            str(Path(directory) / name), f"{fmt} ({' '.join('*' + s for s in suffixes)})")
        if not path:
            return
        path = Path(path)
        if export.format_for(path) != fmt:
            path = path.with_suffix(suffixes[0]) if export.format_for(path) else path.with_name(path.name + suffixes[0])
        dialog = QProgressDialog(f"Saving {stop - start:,} samples to {path.name}…", "Cancel", 0, 1000, self)
        dialog.setWindowTitle("Saving")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done: int, total: int) -> bool:
            dialog.setValue(int(1000 * done / max(total, 1)))
            QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            count = export.save(path, cap, start, stop, progress)
        except export.Cancelled:
            self.statusBar().showMessage("Save cancelled", 5000)
            return
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        finally:
            dialog.close()
        self.settings.setValue("save_dir", str(path.parent))
        self.statusBar().showMessage(f"Saved {count:,} samples per channel to {path}", 10000)

    # --- Shutdown ------------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_settings()
        self.worker.shutdown()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="FastLog captures with an R&S NGM201/NGM202 over LAN.")
    parser.add_argument("--resource", help="IP address, hostname, or VISA resource string to prefill")
    parser.add_argument("--protocol", choices=PROTOCOLS, help="LAN protocol used with an IP/hostname")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    window = MainWindow(args.resource, args.protocol)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
