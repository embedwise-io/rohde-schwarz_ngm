"""PyQt6 front end for FastLog captures with an R&S NGM201 / NGM202 over LAN."""

from __future__ import annotations

import argparse
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import psutil
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QSettings, QTimer, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtGui import QAction, QKeySequence, QShortcut, QTextDocument
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox,
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
from .measurements import MeasurementsPanel
from .plotting import (
    CHANNEL_COLORS, QUANTITY_LABELS, ROLL_DIVISIONS, STYLES, Bands, EnvelopeCurve, PlotViewBox, TimeAxis, ValueAxis,
    colored, format_time, si,
)
from .widgets import PopOutWindow, SetpointSpinBox, ToggleSwitch
from .worker import ROLL_SLACK, InstrumentWorker, roll_capacity

APP_NAME = "R&S NGM FastLog"
HISTORY_LENGTH = 10
MISSING_COLOR = (255, 70, 70, 90)                  # translucent red: samples lost on the way
CLIPPED_COLOR = (255, 200, 0, 70)                  # translucent amber: readings beyond the range
EVENT_CONTEXT = 0.002                              # s: minimum view width when jumping to an event
MEMORY_FRACTION = 0.5                              # of available RAM that captures may use
BYTES_PER_SAMPLE = 8 + 3 * 2 * 4 / PYRAMID_BASE * PYRAMID_FACTOR / (PYRAMID_FACTOR - 1)   # V, I + pyramids
DEFAULT_DURATION = 2.0
PENDING_TIMEOUT = 2.0                              # s a control shows a request before the instrument confirms it
# Roll mode: time per division (s), in the oscilloscope's 1-2-5 steps. The span is ROLL_DIVISIONS of them.
TIMEBASES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1800, 3600)
DEFAULT_TIMEBASE = 1.0
ROLL_FRAME_INTERVAL = 33                           # ms between scroll steps
# Data arrives in chunks of CHUNK_PERIOD. The display runs this far behind the newest sample, so it
# can reveal each chunk at a steady pace instead of jumping a chunk at a time.
ROLL_LATENCY = CHUNK_PERIOD + 0.05
ROLL_STATS_PERIOD = 1.0                            # s between statistics updates while rolling (or 10 × their cost)


def rich_text_size(html: str, font, width: float = -1) -> tuple[float, float]:
    """Size of rich text as a QLabel lays it out, wrapped at `width` (or unwrapped)."""
    doc = QTextDocument()
    doc.setDefaultFont(font)
    doc.setDocumentMargin(0)
    doc.setHtml(html)
    doc.setTextWidth(width)
    size = doc.size()
    return size.width(), size.height()


def worst_stats() -> Stats:
    """Statistics whose line is as long as one can get, to reserve room for it under the plots."""
    value = dict.fromkeys(QUANTITIES, -888.888e-3)
    return Stats(samples=0, valid=1, mean=value, std=dict.fromkeys(QUANTITIES, 888e-3), min=value, max=value,
                 energy_j=-888.8, charge_c=-888.8, missing=(88_888_888, 88_888),
                 voltage_clipped=(88_888_888, 88_888), current_clipped=(0, 0))


def format_span(seconds: float) -> str:
    """A round time such as a time base: 100 ms, 2 s, 5 min, 1 h."""
    if seconds < 1:
        return f"{seconds * 1e3:g} ms"
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:g} min"
    return f"{seconds / 3600:g} h"


def format_duration(seconds: float) -> str:
    if seconds < 1000:
        return f"{seconds:.4g} s"
    if seconds < 3600:
        return f"{seconds / 60:.3g} min"
    return f"{seconds / 3600:.3g} h"


class ChannelPanel(QGroupBox):
    """Setpoints, output switch, live readback and measurement ranges of one channel.

    Everything shown comes from the instrument: nothing is sent until the user changes a control,
    and the controls follow changes made on the front panel. A change the user just made is shown
    until the instrument reports it (or PENDING_TIMEOUT passes), so a status read that was already
    under way can't flip the control back.
    """

    voltage_requested = Signal(int, float)
    current_requested = Signal(int, float)
    output_requested = Signal(int, bool)
    ranges_requested = Signal(int, object, object)     # channel, voltage range, current range (None = auto)
    measure_changed = Signal()

    def __init__(self, ch: int):
        super().__init__(f"Channel {ch}")
        self.ch = ch
        self.setStyleSheet(f"QGroupBox::title {{ color: {CHANNEL_COLORS[ch]}; font-weight: bold; }}")
        self._status: ChannelStatus | None = None
        self._connected = False
        self._running = False
        self._has_ranges = True
        self._pending: dict[str, tuple[object, float]] = {}     # control -> (requested value, deadline)
        form = QFormLayout(self)

        self.measure = QCheckBox("Measure")
        self.measure.setToolTip("Include this channel in FastLog captures")
        self.measure.toggled.connect(self.measure_changed)
        self.output = ToggleSwitch()
        self.output.setToolTip(f"Channel {ch} output: click (or press Space) to switch it on or off")
        self.output.setAccessibleName(f"Channel {ch} output")
        self.output.clicked.connect(self._on_output_clicked)
        output_label = QLabel("Output")
        output_label.setBuddy(self.output)
        row = QHBoxLayout()
        row.addWidget(self.measure)
        row.addStretch(1)
        row.addWidget(output_label)
        row.addWidget(self.output)
        form.addRow(row)

        self.voltage = SetpointSpinBox(" V", 3, 0.0, VOLTAGE_MAX, 0.01)
        self.voltage.setToolTip("Voltage setpoint; sent when you press Enter or leave the field")
        self.voltage.committed.connect(lambda v: self.voltage_requested.emit(self.ch, v))
        form.addRow("Voltage", self.voltage)
        self.current = SetpointSpinBox(" A", 4, CURRENT_MIN, max_current(0), 0.001)
        self.current.setToolTip("Current limit; sent when you press Enter or leave the field.\n"
                                "Up to 6.01 A at 6 V or less, 3.01 A above.")
        self.current.committed.connect(lambda a: self.current_requested.emit(self.ch, a))
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
               "readings during a switch (~20 µs) can be clipped. Applied when you choose it.")
        for combo in (self.voltage_range, self.current_range):
            combo.setToolTip(tip)
            combo.setPlaceholderText("—")
            combo.activated.connect(self._on_ranges_chosen)
        row = QHBoxLayout()
        row.addWidget(self.voltage_range, 1)
        row.addWidget(self.current_range, 1)
        form.addRow("Ranges V / I", row)
        self.set_connected(False)

    # --- State ---------------------------------------------------------------------------

    def setup(self) -> ChannelSetup:
        return ChannelSetup(self.ch, *self._chosen_ranges())

    def _chosen_ranges(self) -> tuple[float | None, float | None]:
        values = []
        for combo, name in ((self.voltage_range, "voltage_range"), (self.current_range, "current_range")):
            if combo.currentIndex() >= 0:
                values.append(combo.currentData())
            else:
                values.append(getattr(self._status, name) if self._status is not None else None)
        return values[0], values[1]

    def _show_ranges(self, voltage_range: float | None, current_range: float | None) -> None:
        for combo, value in ((self.voltage_range, voltage_range), (self.current_range, current_range)):
            index = next((i for i in range(combo.count()) if _same(combo.itemData(i), value)), -1)
            combo.setCurrentIndex(index)

    def set_connected(self, connected: bool, has_ranges: bool = True) -> None:
        self._connected = connected
        self._has_ranges = has_ranges
        if not connected:
            self._status = None
            self._pending.clear()
            self.voltage.set_unknown()
            self.current.set_unknown()
            self.output.setChecked(False)
            self._show_ranges(-1.0, -1.0)            # no match: shows the '—' placeholder
            self.readout.setText("—")
            self.readout.setStyleSheet("")
        elif not has_ranges:
            self._show_ranges(None, None)            # without K104 the instrument autoranges
        self._update_enabled()

    def set_running(self, running: bool) -> None:
        self._running = running
        self._update_enabled()

    def _update_enabled(self) -> None:
        live = self._connected
        for w in (self.voltage, self.current, self.output):
            w.setEnabled(live)
        self.measure.setEnabled(not self._running)
        for combo in (self.voltage_range, self.current_range):
            combo.setEnabled(live and not self._running and self._has_ranges)

    def _expected(self, control: str, reported):
        """What to show for a control: a change the user just made, until the instrument reports it."""
        pending = self._pending.get(control)
        if pending is None:
            return reported
        wanted, deadline = pending
        if _same(wanted, reported) or time.monotonic() > deadline:
            del self._pending[control]
            return reported
        return wanted

    def update_status(self, s: ChannelStatus) -> None:
        self._status = s
        self.current.setMaximum(max_current(s.voltage_set))
        self.voltage.show_value(s.voltage_set)
        self.current.show_value(s.current_set)
        self.output.setChecked(bool(self._expected("output", s.output)))
        combos_open = any(c.view().isVisible() for c in (self.voltage_range, self.current_range))
        if self._has_ranges and not combos_open:
            self._show_ranges(*self._expected("ranges", (s.voltage_range, s.current_range)))
        if s.trips:
            text = "⚠ " + " ".join(s.trips) + " tripped"
        elif not s.output:
            text = "Output off"
        else:
            power = s.voltage * s.current
            text = f"{si(s.voltage, 'V', 6)}  {si(s.current, 'A', 5)}  {si(power, 'W', 4)}  {s.mode}"
        self.readout.setText(text)
        self.readout.setStyleSheet("QLabel { color: #c62828; }" if s.trips else "")

    def _on_output_clicked(self, on: bool) -> None:
        self._pending["output"] = (on, time.monotonic() + PENDING_TIMEOUT)
        self.output_requested.emit(self.ch, on)

    def _on_ranges_chosen(self, *_) -> None:
        ranges = (self.voltage_range.currentData(), self.current_range.currentData())
        self._pending["ranges"] = (ranges, time.monotonic() + PENDING_TIMEOUT)
        self.ranges_requested.emit(self.ch, *ranges)


def _same(a, b) -> bool:
    """Equal, treating floats as equal when close and None (autorange) as a value."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b)
    return a == b


class MainWindow(QMainWindow):
    def __init__(self, resource: str | None = None, protocol: str | None = None,
                 settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        # Only app preferences are remembered. Channel settings (setpoints, outputs, ranges) always
        # come from the instrument, so connecting never changes it.
        self.settings = settings or QSettings("EmbedWise", "NGMDAQ")
        self.info: DeviceInfo | None = None
        self.capture: Capture | None = None       # capture on display
        self.last_complete: Capture | None = None
        self._busy = False                        # scanning, connecting, restarting
        self._connecting = False                  # connecting, or restarting and reconnecting
        self._restarting = False
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

        # The Measurements panel sits in a dock beside the plots, or in a window of its own (Pop Out).
        # The dock doesn't float or move: Qt moves floating docks itself, which Wayland doesn't allow.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.measurements)
        scroll.setMinimumWidth(self.measurements.minimumWidth() + scroll.verticalScrollBar().sizeHint().width())
        self.pop_out_btn = QPushButton()
        self.pop_out_btn.clicked.connect(lambda: self._set_popped_out(not self._popped_out))
        self.measure_view = QWidget()               # what moves between the dock and the window
        view = QVBoxLayout(self.measure_view)
        view.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(9, 6, 9, 0)
        row.addStretch(1)
        row.addWidget(self.pop_out_btn)
        view.addLayout(row)
        view.addWidget(scroll)
        self.measure_dock = QDockWidget("Measurements", self)
        self.measure_dock.setObjectName("measurements")
        self.measure_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.measure_dock)
        self.measure_dock.hide()
        self.measure_window = PopOutWindow("Measurements", self)
        self.measure_window.resize(scroll.minimumWidth() + 24, 800)
        self.measure_window.closed.connect(self._on_measure_window_closed)
        self._popped_out = False
        self._set_popped_out(False)

        shortcuts = [
            (QKeySequence("F5"), self._on_run),
            (QKeySequence("F6"), self._on_trigger),
            (QKeySequence(QKeySequence.StandardKey.ZoomIn), lambda: self._zoom(0.5)),
            (QKeySequence(QKeySequence.StandardKey.ZoomOut), lambda: self._zoom(2.0)),
            (QKeySequence("Ctrl+0"), self._fit),
            (QKeySequence(QKeySequence.StandardKey.Save), lambda: self._save(view_only=True)),
            (QKeySequence("Esc"), self.measurements.clear_box),
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
        self.timebase = QComboBox()
        for div in TIMEBASES:
            self.timebase.addItem(f"{format_span(div)}/div", div)
        self.timebase.setCurrentIndex(self.timebase.findData(DEFAULT_TIMEBASE))
        self.timebase.setToolTip(
            f"Roll mode: time per division. The plots show {ROLL_DIVISIONS} divisions, and only that span\n"
            "is kept in memory: older samples are discarded as new ones arrive.")
        self.timebase.currentIndexChanged.connect(self._update_summary)
        self.span_label = QLabel()
        self.timebase_row = self._group(self.timebase, self.span_label)
        form.addRow("Time base", self.timebase_row)
        self._acquisition_form = form
        self.trigger_mode = QComboBox()
        for mode in TriggerMode:
            self.trigger_mode.addItem(mode.value, mode)
        self.trigger_mode.setToolTip(
            "Single: Run takes one capture. Stop early to keep what was recorded.\n"
            "Auto: captures follow each other seamlessly until Stop; the last complete one is kept.\n"
            "Manual: Run arms FastLog; each Trigger press takes one capture, then it re-arms until Stop.\n"
            "Roll: acquires until Stop and scrolls like a strip chart, keeping only the span on display.")
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
        self.select_btn = self._tool_button(
            "Select", "Drag a rectangle over the data for statistics of the samples inside it.\n"
                      "Drag a corner to resize it, or the inside to move it. Esc clears it. When off, drag pans.")
        self.select_btn.setCheckable(True)
        self.select_btn.toggled.connect(self._on_select)
        bar.addWidget(self.select_btn)
        self.markers_btn = self._tool_button(
            "Markers", "X1, X2 (time) and Y1, Y2 (level) markers, read out in the Measurements panel.\n"
                       "Drag a marker to move it; right-click a plot to place one there. Click again to clear them.")
        self.markers_btn.setCheckable(True)
        self.markers_btn.clicked.connect(self._on_markers_clicked)
        bar.addWidget(self.markers_btn)
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
            p = pg.PlotItem(viewBox=PlotViewBox(), axisItems={"left": ValueAxis(), "bottom": TimeAxis()})
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
            vb.sigSelecting.connect(partial(self._on_selecting, q))
            self.plots[q] = p
        self.measurements = MeasurementsPanel(self.plots)
        self.measurements.changed.connect(self._on_measurements_changed)
        for q, p in self.plots.items():
            self._add_measure_actions(q, p.getViewBox())
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        layout.addWidget(self.plot_widget, 1)

        info = QHBoxLayout()
        self.stats_label = QLabel()
        self.stats_label.setTextFormat(Qt.TextFormat.RichText)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.cursor_label = QLabel()
        self.cursor_label.setTextFormat(Qt.TextFormat.RichText)
        self.cursor_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        # The two labels share the row, and the statistics wrap. Neither may change size with the
        # values shown, or the plots above would resize as the mouse moves: the cursor readout keeps
        # the width of its longest text, and the statistics the height of theirs (_reserve_stats_height).
        widest = self._cursor_text(88_888_888_888, 88888.8888,
                                   [(ch, -888.888e-3, -888.88e-3, -888.8e-3) for ch in CHANNEL_COLORS])
        self.cursor_label.setMinimumWidth(math.ceil(rich_text_size(widest, self.cursor_label.font())[0]) + 2)
        self.stats_label.installEventFilter(self)
        info.addWidget(self.stats_label, 1)
        info.addWidget(self.cursor_label)
        layout.addLayout(info)

        self._stats_timer = QTimer(self, singleShot=True, interval=150)
        self._stats_timer.timeout.connect(self._update_stats)
        self._roll_timer = QTimer(self, interval=ROLL_FRAME_INTERVAL)
        self._roll_timer.timeout.connect(self._roll_frame)
        self._roll_end = 0.0                        # time of the newest sample received (s)
        self._roll_arrival = 0.0                    # when it arrived (monotonic)
        self._roll_shown: float | None = None       # time at the right edge of the plots
        self._roll_stats_due = 0.0
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
        self.timebase.setCurrentIndex(max(0, self.timebase.findData(s.value("timebase_s", DEFAULT_TIMEBASE, float))))
        self._update_summary()
        if geometry := s.value("geometry"):
            self.restoreGeometry(geometry)
        if splitter := s.value("splitter"):
            self.splitter.restoreState(splitter)
        if geometry := s.value("measurements_geometry"):
            self.measure_window.restoreGeometry(geometry)
        self._set_popped_out(s.value("measurements_popped_out", False, bool))
        y_plot = s.value("y_markers_in", "current", str)
        if y_plot in QUANTITIES:
            self.measurements.move_y_markers(y_plot)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("protocol", self.protocol.currentText())
        for ch, panel in self.channel_panels.items():
            s.setValue(f"ch{ch}/measure", panel.measure.isChecked())
            for key in ("voltage_range", "current_range"):     # remembered by versions before 0.2.0
                s.remove(f"ch{ch}/{key}")
        s.setValue("rate", self.rate.currentData().name)
        s.setValue("duration_s", self.duration.value())
        s.setValue("timebase_s", self.timebase.currentData())
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
        s.remove("window_state")                   # a floating dock, saved by an earlier build
        s.setValue("measurements_popped_out", self._popped_out)
        s.setValue("measurements_geometry", self.measure_window.saveGeometry())
        s.setValue("y_markers_in", self.measurements.y_quantity)

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
            duration=self._roll_span() if self.trigger_mode.currentData() is TriggerMode.ROLL
            else self.duration.value(),
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
        for w in (self.rate, self.duration, self.timebase, self.trigger_mode, self.output_on_start,
                  self.output_off_end):
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
        """Longest capture (or roll span) that fits the memory budget.

        Two captures may be in memory at once. In roll mode those are two buffers, each the span
        plus its slack, and a copy in flight when one replaces the other.
        """
        budget = psutil.virtual_memory().available * MEMORY_FRACTION
        samples = budget / (BYTES_PER_SAMPLE * max(1, channels) * 2)
        if mode is TriggerMode.ROLL:
            samples /= (1 + ROLL_SLACK) * 1.2
        return max(CHUNK_PERIOD, math.floor(samples / rate.hz * 100) / 100)

    def _roll_span(self) -> float:
        return self.timebase.currentData() * ROLL_DIVISIONS

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
        rolling = mode is TriggerMode.ROLL
        self._acquisition_form.setRowVisible(self.duration, not rolling)
        self._acquisition_form.setRowVisible(self.timebase_row, rolling)
        link = rate.hz * 8 * channels
        if rolling:
            # Time bases whose span doesn't fit in memory can't be chosen.
            model = self.timebase.model()
            fits = [self.timebase.itemData(i) * ROLL_DIVISIONS <= max_duration for i in range(self.timebase.count())]
            for i, ok in enumerate(fits):
                model.item(i).setEnabled(ok)
            if not fits[self.timebase.currentIndex()]:
                self.timebase.setCurrentIndex(max((i for i, ok in enumerate(fits) if ok), default=0))
            span = self._roll_span()
            samples = max(1, round(span * rate.hz))
            size = roll_capacity(samples, rate.chunk_samples) * BYTES_PER_SAMPLE * channels
            self.span_label.setText(f"span {format_span(span)}")
            self.summary.setText(
                f"{samples:,} samples per channel · {si(size, 'B', 3)} in memory\n"
                f"Link {si(link, 'B/s', 3)} · runs until Stop")
            return
        self.duration.setMaximum(max_duration)
        samples = max(1, round(self.duration.value() * rate.hz))
        size = samples * BYTES_PER_SAMPLE * channels
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
            self.channel_panels[ch].update_status(status)

    def _on_output_requested(self, ch: int, on: bool) -> None:
        self.worker.set_output(ch, on)

    def _on_ranges_requested(self, ch: int, voltage_range: float | None, current_range: float | None) -> None:
        if self.info is not None and not self._running:
            self.worker.set_ranges(ch, voltage_range, current_range)

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
        if capture.rolling:
            self._set_progress(busy=True)
        else:
            self._set_progress(maximum=capture.target)
        first = self._new_run
        self._new_run = False
        self._show(capture, fit=first)
        if capture.rolling:
            self._start_rolling(capture)
        self.statusBar().showMessage(f"Capture {capture.number}: waiting for data…")
        self._update_controls()

    @Slot(object)
    def _on_progress(self, capture: Capture) -> None:
        rolling = capture.rolling and self._capturing
        if capture is not self.capture:
            if not rolling:
                return
            # The roll buffer filled and the worker continued in a new one: same data, shifted.
            self.capture = capture
            for (q, ch), curve in self.curves.items():
                curve.set_source(capture.channels.get(ch), q, capture.interval, capture.offset)
        n = capture.n
        for curve in self.curves.values():
            curve.data_changed()
        self._refresh_bands()
        self.measurements.set_capture(capture, capturing=True)
        if rolling:
            end = capture.time_of(n)
            if end > self._roll_end:
                self._roll_end, self._roll_arrival = end, time.monotonic()
            self.statusBar().showMessage(
                f"Rolling: {format_duration(end)} elapsed, showing the last {format_span(capture.config.duration)}")
            return
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
        if capture.rolling:
            self._stop_rolling()
            self.statusBar().showMessage(
                f"Stopped after {format_duration(capture.time_of(n))}; kept the last "
                f"{format_duration(n * capture.interval)} ({n:,} samples)", 10000)
            return
        self.statusBar().showMessage(f"Capture {capture.number}: {n:,} samples in {format_duration(n * capture.interval)}"
                                     f"{'' if capture.complete else ' (partial)'}", 10000)

    @Slot()
    def _on_run_finished(self) -> None:
        stopped = self._stopping
        self._running = self._stopping = self._armed = self._capturing = False
        self._roll_timer.stop()
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
            curve.set_source(data, q, dt, capture.offset)
            curve.setVisible(data is not None)
        start, end = capture.t0, capture.time_of(max(capture.target - 1, 1))
        scrolling = capture.rolling and self._capturing     # the view follows the data; see _roll_frame
        division = capture.config.duration / ROLL_DIVISIONS if capture.rolling else None
        for p in self.plots.values():
            vb = p.getViewBox()
            if scrolling:
                vb.setLimits(xMin=None, xMax=None, minXRange=5 * dt)
            else:
                vb.setLimits(xMin=start, xMax=end, minXRange=5 * dt)
            vb.setMouseEnabled(x=not scrolling, y=True)
            p.getAxis("bottom").set_division(division)
        if fit:
            vb = self._visible_plots()[0].getViewBox()
            if scrolling:
                vb.setXRange(-capture.config.duration, 0, padding=0)
            else:
                vb.setXRange(start, end, padding=0)
            for p in self.plots.values():
                p.getViewBox().enableAutoRange(y=True)
        self._update_title()
        self._refresh_bands()
        self._update_warnings()
        self._update_controls()
        self._reserve_stats_height()
        self._update_stats()
        self.measurements.set_capture(capture, self._capturing)

    # Roll mode: the plots always show the last span, and scroll as data arrives.

    def _start_rolling(self, capture: Capture) -> None:
        self._roll_end, self._roll_arrival = capture.t0, time.monotonic()
        self._roll_shown = None
        self._roll_stats_due = 0.0
        self._roll_timer.start()

    def _roll_frame(self) -> None:
        """Scroll the plots so their right edge follows the data, ROLL_LATENCY behind the newest sample."""
        cap = self.capture
        if cap is None or not (cap.rolling and self._capturing):
            return
        span = cap.config.duration
        now = time.monotonic()
        edge = min(self._roll_end, self._roll_end + (now - self._roll_arrival) - ROLL_LATENCY)
        vb = self._visible_plots()[0].getViewBox()
        pixel = span / max(1.0, vb.width())
        if self._roll_shown is None or edge - self._roll_shown >= pixel / 2:      # never backwards
            self._roll_shown = edge
            vb.setXRange(edge - span, edge, padding=0)
        if now >= self._roll_stats_due and cap.n:
            self._update_stats(live=True)
            self._roll_stats_due = now + max(ROLL_STATS_PERIOD, 10 * (time.monotonic() - now))

    def _stop_rolling(self) -> None:
        """Leave the plots on the span that was kept."""
        self._roll_timer.stop()
        cap = self.capture
        end = cap.time_of(max(cap.n - 1, 1))
        self._visible_plots()[0].getViewBox().setXRange(max(cap.t0, end - cap.config.duration), end, padding=0)

    def _update_title(self) -> None:
        cap = self.capture
        if cap is None:
            return
        parts = []
        for ch, data in cap.channels.items():
            s = data.settings
            parts.append(f"{colored(f'CH{ch}', ch)} {s.voltage_set:g} V, {si(s.current_set, 'A', 3)} limit, "
                         f"ranges {range_label(s.voltage_range, 'V')} / {range_label(s.current_range, 'A')}")
        what = f"roll {format_span(cap.config.duration / ROLL_DIVISIONS)}/div" if cap.rolling else f"#{cap.number}"
        title = (f"{cap.device.model} · {format_rate(cap.sample_rate)} · " + " · ".join(parts)
                 + f" · {what} {cap.started:%Y-%m-%d %H:%M:%S}")
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
            dt, off = (cap.interval, cap.offset) if cap is not None else (1.0, 0)
            for bands, runs in ((self.missing_bands[q], m), (self.clipped_bands[q], c)):
                bands.set_intervals((off + runs[:, 0] - 0.5) * dt, (off + runs[:, 1] - 0.5) * dt)
            if self.show_quantity[q].isChecked():
                events += [m, c]
        merged = merge_intervals(*events) if cap is not None else np.empty((0, 2))
        self._events = (merged + cap.offset) * cap.interval if cap is not None else merged
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
        cap = self.capture
        self._visible_plots()[0].getViewBox().setXRange(cap.t0, cap.time_of(max(cap.n - 1, 1)), padding=0.01)
        for p in self.plots.values():
            p.getViewBox().enableAutoRange(y=True)

    def _apply_plot_style(self) -> None:
        for curve in self.curves.values():
            curve.set_style(self.plot_style.currentText())

    def _on_box_zoom(self, enabled: bool) -> None:
        if enabled:
            self.select_btn.setChecked(False)
        for p in self.plots.values():
            p.getViewBox().setMouseMode(pg.ViewBox.RectMode if enabled else pg.ViewBox.PanMode)

    def _on_select(self, enabled: bool) -> None:
        if enabled:
            self.box_zoom_btn.setChecked(False)
        for p in self.plots.values():
            p.getViewBox().selecting = enabled
        self.measurements.set_box_movable(enabled)
        if enabled:
            self.plot_widget.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.plot_widget.unsetCursor()

    def _on_selecting(self, quantity: str, start, end, finished: bool) -> None:
        self.measurements.drag_box(quantity, start, end, finished)

    def _on_markers_clicked(self, checked: bool) -> None:
        if checked:
            self.measurements.place_markers(self._view_range())
        else:
            self.measurements.clear_markers()

    def _on_measurements_changed(self) -> None:
        """Show the Measurements panel (docked or popped out) while there are markers or a selection."""
        m = self.measurements
        self.markers_btn.setChecked(m.has_markers)
        active = m.has_markers or m.has_box
        self.measure_dock.setVisible(active and not self._popped_out)
        if not (active and self._popped_out):
            self.measure_window.hide()
        elif not self.measure_window.isVisible():
            self.measure_window.show()
            self.measure_window.raise_()

    def _set_popped_out(self, popped: bool) -> None:
        """Move the Measurements panel into its own window, or back into the dock beside the plots."""
        self._popped_out = popped
        if popped:
            self.measure_window.layout().addWidget(self.measure_view)
        else:
            self.measure_dock.setWidget(self.measure_view)
        self.measure_view.show()
        self.pop_out_btn.setText("Dock" if popped else "Pop Out")
        self.pop_out_btn.setToolTip(
            "Put the measurements back beside the plots" if popped else
            "Show the measurements in a window of their own, which you can move anywhere.\n"
            "Closing that window clears the markers and the selection.")
        self._on_measurements_changed()

    def _on_measure_window_closed(self) -> None:
        self.measurements.clear_markers()
        self.measurements.clear_box()

    def _add_measure_actions(self, quantity: str, vb: PlotViewBox) -> None:
        """Right-click menu entries: place a marker where the menu was opened, and clear."""
        menu = vb.menu
        first = menu.actions()[0]
        for name in ("X1", "X2", "Y1", "Y2"):
            action = QAction(f"Place {name} Here", menu)
            action.triggered.connect(partial(self._place_marker, name, quantity))
            menu.insertAction(first, action)
        clear_markers = QAction("Clear Markers", menu)
        clear_markers.triggered.connect(self.measurements.clear_markers)
        clear_selection = QAction("Clear Selection", menu)
        clear_selection.triggered.connect(self.measurements.clear_box)
        menu.insertActions(first, [clear_markers, clear_selection])
        menu.insertSeparator(first)
        menu.aboutToShow.connect(lambda: (clear_markers.setEnabled(self.measurements.has_markers),
                                          clear_selection.setEnabled(self.measurements.has_box)))

    def _place_marker(self, name: str, quantity: str) -> None:
        point = self.plots[quantity].getViewBox().menu_point
        if point is not None:
            self.measurements.place_marker(name, quantity, point)

    def _visible_samples(self) -> tuple[int, int]:
        return self.capture.samples_between(*self._view_range())

    def _target_event(self, step: int) -> int | None:
        """Index of the highlighted interval Previous (-1) or Next (+1) would jump to."""
        events = self._events
        if self.capture is None or self._capturing or len(events) == 0:
            return None
        dt, t0 = self.capture.interval, self.capture.t0
        x0, x1 = self._view_range()
        if self._event_nav is not None and np.allclose(self._event_nav[1], (x0, x1), rtol=0, atol=dt / 10):
            i = self._event_nav[0] + step
        elif x0 <= t0 + dt and x1 >= t0 + (self.capture.n - 2) * dt:
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

    def _update_stats(self, live: bool = False) -> None:
        """Statistics of the displayed window. They wait for a capture to finish, except in roll
        mode, which asks for them now and then (live)."""
        cap = self.capture
        if cap is None or (self._capturing and not live):
            return
        start, stop = self._visible_samples()
        if stop <= start:
            self.stats_label.setText("")
            return
        lines = [self._view_line((stop - start) * cap.interval, stop - start)]
        for ch, data in cap.channels.items():
            lines.append(self._stats_line(ch, window_stats(data, start, stop, cap.interval)))
        self.stats_label.setText("<br>".join(lines))

    @staticmethod
    def _view_line(seconds: float, samples: int) -> str:
        return f"View {si(seconds, 's', 4)} · {samples:,} samples"

    def _reserve_stats_height(self) -> None:
        """Keep the statistics as tall as their longest text needs at the current width.

        So their height changes only with the window width and the channels shown, not with the
        values: otherwise a line that wraps or unwraps as you zoom would resize the plots.
        """
        label = self.stats_label
        channels = self.capture.channels if self.capture is not None else {}
        html = "<br>".join([self._view_line(888.8e-3, 88_888_888)]
                           + [self._stats_line(ch, worst_stats()) for ch in channels])
        height = rich_text_size(html, label.font(), max(1, label.contentsRect().width()))[1]
        label.setMinimumHeight(math.ceil(height))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.stats_label and event.type() == QEvent.Type.Resize:
            self._reserve_stats_height()
        return super().eventFilter(obj, event)

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
        i = cap.index_at(t)
        if not 0 <= i < cap.n:
            return
        values = []
        for ch, data in cap.channels.items():
            v, a = float(data.voltage[i]), float(data.current[i])
            values.append((ch, v, a, v * a))
        self.cursor_label.setText(self._cursor_text(cap.offset + i, cap.time_of(i), values))

    @staticmethod
    def _cursor_text(i: int, t: float, values: list[tuple[int, float, float, float]]) -> str:
        """The sample under the mouse: index, time, and per channel (channel, V, I, P)."""
        parts = [f"#{i:,} &nbsp; t = {format_time(t)}"]
        for ch, v, a, p in values:
            if math.isnan(v):
                parts.append(f"{colored(f'CH{ch}', ch)} missing")
            else:
                parts.append(f"{colored(f'CH{ch}', ch)} {si(v, 'V', 6)} {si(a, 'A', 5)} {si(p, 'W', 4)}")
        return "<br>".join(parts)

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
