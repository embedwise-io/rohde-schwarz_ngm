"""X1/X2 and Y1/Y2 markers and a selection box on the plots, and the panel that reads them out.

X markers are times: one vertical line in every plot, moving together, since the plots share the
time axis. Y markers are levels of one quantity, so Y1 and Y2 always sit in the same plot and ΔY
has one unit. The selection box is a time span and a level range in one plot; its statistics
cover each channel's samples that fall inside both.
"""

from __future__ import annotations

import math
from functools import partial

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .capture import QUANTITIES, UNITS, Capture, box_stats, samples_between
from .plotting import (
    QUANTITY_LABELS, X_MARKER_COLOR, Y_MARKER_COLOR, MarkerLine, SelectionBox, colored, si,
)

X_MARKERS = ("X1", "X2")
Y_MARKERS = ("Y1", "Y2")
DEFAULT_POSITION = {"X1": 0.25, "X2": 0.75, "Y1": 0.25, "Y2": 0.75}    # fraction of the view
LABEL_POSITION = {"X1": 0.94, "X2": 0.86, "Y1": 0.03, "Y2": 0.10}      # along the line; staggered
PRECISION = {"s": 7, "V": 6, "A": 5, "W": 4}                           # significant digits shown
UPDATE_INTERVAL = 80                # ms: readouts refresh at most this often while dragging
STATS_DELAY = 200                   # ms after a drag pauses before large selections are computed
LIVE_SAMPLES = 2_000_000            # selections up to this many samples update live (~15 ms)
MIN_BOX_PX = 3                      # a smaller drag makes no selection
# Columns are sized for the longest text they can hold, so they don't shift as values change.
WIDEST_VALUE = "-888.8888 mV"       # times have 7 digits
WIDEST_LABEL = "Voltage"            # first column of the marker and selection-edge tables
WIDEST_STAT = "888.8 ms (100.0%)"   # statistics table: one or two channel columns
WIDEST_STAT_LABEL = "Points in time range"


def _value(value: float, unit: str) -> str:
    return si(value, unit, PRECISION[unit])


def _reciprocal(delta: float, unit: str) -> str:
    """1/Δ: in Hz for time; for a level, without an SI prefix (12 mA⁻¹ would be ambiguous)."""
    if not delta or not math.isfinite(delta):
        return "—"
    return si(1 / delta, "Hz", 6) if unit == "s" else f"{1 / delta:#.6g} {unit}⁻¹"


def _table(header: list[str], rows: list[list[str]], column: int) -> str:
    cells = "".join(f"<th align='right' width='{column}'>{h}</th>" for h in header[1:])
    html = [f"<table cellspacing='0' cellpadding='2'><tr><th></th>{cells}</tr>"]
    for label, *values in rows:
        html.append(f"<tr><td>{label}&nbsp;</td>"
                    + "".join(f"<td align='right'>&nbsp;{v}</td>" for v in values) + "</tr>")
    return "".join(html) + "</table>"


def _tinted(text: str, color: str) -> str:
    return f"<span style='color:{color}; font-weight:bold'>{text}</span>"


class MeasurementsPanel(QWidget):
    """Owns the markers and the selection box on `plots`, and shows what they measure."""

    changed = Signal()              # markers or the selection box appeared or disappeared

    def __init__(self, plots: dict[str, pg.PlotItem]):
        super().__init__()
        self._plots = plots
        self._capture: Capture | None = None
        self._capturing = False
        self._placed: set[str] = set()
        self._syncing = False
        self._timer = QTimer(self, singleShot=True, interval=UPDATE_INTERVAL)
        self._timer.timeout.connect(lambda: self.update_readouts(final=False))
        self._stats_timer = QTimer(self, singleShot=True, interval=STATS_DELAY)
        self._stats_timer.timeout.connect(self.update_readouts)
        self._stats_cache: tuple[tuple, str] | None = None     # (what was computed, its table)

        self._x_lines: dict[str, dict[str, MarkerLine]] = {}
        for name in X_MARKERS:
            self._x_lines[name] = {}
            for q, plot in plots.items():
                line = MarkerLine(name, True, X_MARKER_COLOR, name == "X2", LABEL_POSITION[name])
                line.hide()
                plot.addItem(line, ignoreBounds=True)
                line.sigPositionChanged.connect(partial(self._x_moved, name))
                self._x_lines[name][q] = line
        self.y_quantity = "current"
        self._y_lines: dict[str, MarkerLine] = {}
        for name in Y_MARKERS:
            line = MarkerLine(name, False, Y_MARKER_COLOR, name == "Y2", LABEL_POSITION[name])
            line.hide()
            plots[self.y_quantity].addItem(line, ignoreBounds=True)
            line.sigPositionChanged.connect(self._schedule)
            self._y_lines[name] = line
        self.box = SelectionBox()
        self.box_quantity: str | None = None        # plot the box is in; None: no selection
        self.box.sigRegionChanged.connect(self._schedule)

        layout = QVBoxLayout(self)
        group = QGroupBox("Markers")
        group.setToolTip("Drag a marker to move it. Right-click a plot to place a marker at that point.")
        box = QVBoxLayout(group)
        row = QHBoxLayout()
        self.y_plot = QComboBox()
        for q in QUANTITIES:
            self.y_plot.addItem(QUANTITY_LABELS[q], q)
        self.y_plot.setCurrentIndex(self.y_plot.findData(self.y_quantity))
        self.y_plot.setToolTip("Plot that Y1 and Y2 measure")
        self.y_plot.activated.connect(lambda: self.move_y_markers(self.y_plot.currentData()))
        y_label = QLabel("Y1, Y2 in")
        y_label.setBuddy(self.y_plot)
        clear = QPushButton("Clear")
        clear.setToolTip("Remove all markers")
        clear.clicked.connect(self.clear_markers)
        for w in (y_label, self.y_plot):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(clear)
        box.addLayout(row)
        self.markers_label = self._readout()
        box.addWidget(self.markers_label)
        layout.addWidget(group)

        group = QGroupBox("Selection")
        group.setToolTip("Points in time range: samples between the box's left and right edges.\n"
                         "Points in box: of those, the ones between its bottom and top edges.\n"
                         "Time in box: points in box × sample interval, and its share of the time range.\n"
                         "Min, mean, max and σ are of the points in box. Missing samples are left out.")
        box = QVBoxLayout(group)
        row = QHBoxLayout()
        row.addStretch(1)
        clear = QPushButton("Clear")
        clear.setToolTip("Remove the selection box (Esc)")
        clear.clicked.connect(self.clear_box)
        row.addWidget(clear)
        box.addLayout(row)
        self.selection_label = self._readout()
        box.addWidget(self.selection_label)
        layout.addWidget(group)
        layout.addStretch(1)
        fm = self.markers_label.fontMetrics()
        def cell(text: str) -> int:
            return fm.horizontalAdvance("\u00a0" + text) + 8                 # + the cell's &nbsp; and padding
        self._column = cell(WIDEST_VALUE)
        self._stat_column = cell(WIDEST_STAT)
        margins = 2 * (layout.contentsMargins().left() + box.contentsMargins().left() + 8)
        self.setMinimumWidth(margins + max(cell(WIDEST_LABEL) + 3 * self._column,
                                           cell(WIDEST_STAT_LABEL) + 2 * self._stat_column))
        self.update_readouts()

    @staticmethod
    def _readout() -> QLabel:
        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return label

    # --- Markers -------------------------------------------------------------------------

    @property
    def has_markers(self) -> bool:
        return bool(self._placed)

    def marker(self, name: str) -> float | None:
        """A marker's position (time for X, level for Y), or None if it isn't placed."""
        if name not in self._placed:
            return None
        line = next(iter(self._x_lines[name].values())) if name in X_MARKERS else self._y_lines[name]
        return line.value()

    def place_markers(self, x_range: tuple[float, float]) -> None:
        """Place every marker not placed yet: X1, X2 across x_range, Y1, Y2 within the Y plot's view."""
        y_range = self._plots[self.y_quantity].getViewBox().viewRange()[1]
        for name in X_MARKERS + Y_MARKERS:
            if name not in self._placed:
                lo, hi = x_range if name in X_MARKERS else y_range
                self._set_marker(name, lo + DEFAULT_POSITION[name] * (hi - lo))
        self.changed.emit()

    def place_marker(self, name: str, quantity: str, point: QPointF) -> None:
        """Place one marker at a point in a plot. A Y marker brings its partner to that plot."""
        if name in Y_MARKERS:
            self.move_y_markers(quantity)
        self._set_marker(name, point.x() if name in X_MARKERS else point.y())
        self.changed.emit()

    def _set_marker(self, name: str, value: float) -> None:
        lines = self._x_lines[name].values() if name in X_MARKERS else (self._y_lines[name],)
        for line in lines:
            line.setValue(value)
            line.show()
        self._placed.add(name)
        self._schedule()

    def move_y_markers(self, quantity: str) -> None:
        """Move Y1 and Y2 to another plot, at the same height relative to its view."""
        if quantity == self.y_quantity:
            return
        old = self._plots[self.y_quantity].getViewBox().viewRange()[1]
        new = self._plots[quantity].getViewBox().viewRange()[1]
        for line in self._y_lines.values():
            fraction = (line.value() - old[0]) / (old[1] - old[0]) if old[1] != old[0] else 0.5
            self._plots[self.y_quantity].removeItem(line)
            self._plots[quantity].addItem(line, ignoreBounds=True)
            line.setValue(new[0] + min(max(fraction, 0.0), 1.0) * (new[1] - new[0]))
        self.y_quantity = quantity
        self.y_plot.setCurrentIndex(self.y_plot.findData(quantity))
        self._schedule()

    def clear_markers(self) -> None:
        for lines in self._x_lines.values():
            for line in lines.values():
                line.hide()
        for line in self._y_lines.values():
            line.hide()
        self._placed.clear()
        self._schedule()
        self.changed.emit()

    def _x_moved(self, name: str, line: MarkerLine) -> None:
        """Keep an X marker's lines in all plots together."""
        if self._syncing:
            return
        self._syncing = True
        try:
            for other in self._x_lines[name].values():
                if other is not line:
                    other.setValue(line.value())
        finally:
            self._syncing = False
        self._schedule()

    # --- Selection box -------------------------------------------------------------------

    @property
    def has_box(self) -> bool:
        return self.box_quantity is not None

    def drag_box(self, quantity: str, start: QPointF, end: QPointF, finished: bool) -> None:
        """Draw the selection box from a drag in a plot (data coordinates), replacing any other.

        `changed` waits for the end of the drag: the panel appearing mid-drag would resize the plot
        under the mouse.
        """
        if quantity != self.box_quantity:
            if self.box_quantity is not None:
                self._plots[self.box_quantity].removeItem(self.box)
            self._plots[quantity].addItem(self.box, ignoreBounds=True)
            self.box_quantity = quantity
        self.box.set_corners(start.x(), start.y(), end.x(), end.y(), finish=finished)
        if finished:
            px, py = self._plots[quantity].getViewBox().viewPixelSize()
            x0, x1, y0, y1 = self.box.bounds()
            if (x1 - x0) < MIN_BOX_PX * px or (y1 - y0) < MIN_BOX_PX * py:
                self.clear_box()
            else:
                self.changed.emit()

    def clear_box(self) -> None:
        if self.box_quantity is None:
            return
        self._plots[self.box_quantity].removeItem(self.box)
        self.box_quantity = None
        self._schedule()
        self.changed.emit()

    def set_box_movable(self, movable: bool) -> None:
        """Let a drag inside the box move it (Select mode); otherwise it pans the plot."""
        self.box.translatable = movable

    # --- Readouts ------------------------------------------------------------------------

    def set_capture(self, capture: Capture | None, capturing: bool = False) -> None:
        """Data to read at the markers. Selection statistics wait until the capture is complete."""
        self._capture = capture
        self._capturing = capturing
        self._schedule()

    def _schedule(self, *_) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self._stats_timer.start()                   # restarted: fires once changes pause

    def update_readouts(self, final: bool = True) -> None:
        """Refresh the readouts. Unless final, statistics of a large selection wait (see LIVE_SAMPLES)."""
        self.markers_label.setText(self._markers_html())
        self.selection_label.setText(self._selection_html(final))

    def _markers_html(self) -> str:
        if not self._placed:
            return "<i>None. Turn on Markers in the toolbar, or right-click a plot to place one.</i>"
        parts = []
        x1, x2 = (self.marker(n) for n in X_MARKERS)
        if x1 is not None or x2 is not None:
            dx = x2 - x1 if x1 is not None and x2 is not None else math.nan
            rows = [["t", _value(x1, "s"), _value(x2, "s"), si(dx, "s", 6)],
                    ["1/ΔX", "", "", _reciprocal(dx, "s")]]
            for ch, at1, at2 in self._samples_at(x1, x2):
                for q, label in (("voltage", "V"), ("current", "I"), ("power", "P")):
                    a, b = at1[q], at2[q]
                    rows.append([colored(f"CH{ch} {label}", ch), _value(a, UNITS[q]), _value(b, UNITS[q]),
                                 _value(b - a, UNITS[q])])
            parts.append(_table(["", *(_tinted(h, X_MARKER_COLOR) for h in ("X1", "X2", "ΔX"))], rows,
                                self._column))
        y1, y2 = (self.marker(n) for n in Y_MARKERS)
        if y1 is not None or y2 is not None:
            unit = UNITS[self.y_quantity]
            dy = y2 - y1 if y1 is not None and y2 is not None else math.nan
            rows = [[QUANTITY_LABELS[self.y_quantity], _value(y1, unit), _value(y2, unit), _value(dy, unit)],
                    ["1/ΔY", "", "", _reciprocal(dy, unit)]]
            parts.append(_table(["", *(_tinted(h, Y_MARKER_COLOR) for h in ("Y1", "Y2", "ΔY"))], rows,
                                self._column))
        return "<br>".join(parts)

    @staticmethod
    def _time_in_box(s, interval: float) -> str:
        """Points in box × sample interval, and its share of the time range."""
        if s.span == 0:
            return "—"
        return f"{si(s.count * interval, 's', 4)} ({100 * s.count / s.span:.1f}%)"

    def _samples_at(self, *times: float | None) -> list[tuple]:
        """Per channel: (channel, then for each time {quantity: value of the nearest sample})."""
        cap = self._capture
        if cap is None:
            return []
        n = cap.n
        result = []
        for ch, data in cap.channels.items():
            values = []
            for t in times:
                i = round(t / cap.interval) if t is not None else -1
                v, a = (float(data.voltage[i]), float(data.current[i])) if 0 <= i < n else (math.nan, math.nan)
                values.append({"voltage": v, "current": a, "power": v * a})
            result.append((ch, *values))
        return result

    def _selection_html(self, final: bool) -> str:
        q = self.box_quantity
        if q is None:
            return "<i>None. Turn on Select in the toolbar and drag a rectangle in a plot.</i>"
        unit = UNITS[q]
        x0, x1, y0, y1 = self.box.bounds()
        html = _table(["", "From", "To", "Δ"], [
            ["t", _value(x0, "s"), _value(x1, "s"), si(x1 - x0, "s", 6)],
            [QUANTITY_LABELS[q], _value(y0, unit), _value(y1, unit), _value(y1 - y0, unit)]], self._column)
        cap = self._capture
        if cap is None:
            return html
        if self._capturing:
            return html + "<br><i>Statistics follow when the capture is complete.</i>"
        key = (q, x0, x1, y0, y1, id(cap), cap.number, cap.started, cap.n)
        if self._stats_cache is not None and self._stats_cache[0] == key:
            return html + self._stats_cache[1]
        start, stop = samples_between(x0, x1, cap.interval, cap.n)
        if not final and (stop - start) * len(cap.channels) > LIVE_SAMPLES:
            return html + "<br><i>Updating…</i>"
        stats = {ch: box_stats(data, q, start, stop, y0, y1) for ch, data in cap.channels.items()}
        rows = [["Points in time range", *(f"{s.span:,}" for s in stats.values())],
                ["Points in box", *(f"{s.count:,}" for s in stats.values())],
                ["Time in box", *(self._time_in_box(s, cap.interval) for s in stats.values())]]
        for label, field in (("Min", "min"), ("Mean", "mean"), ("Max", "max"), ("σ", "std")):
            rows.append([label, *(_value(getattr(s, field), unit) for s in stats.values())])
        table = "<br>" + _table(["", *(colored(f"CH{ch}", ch) for ch in stats)], rows, self._stat_column)
        self._stats_cache = (key, table)
        return html + table
