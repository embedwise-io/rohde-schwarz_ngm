"""pyqtgraph items for long, evenly sampled FastLog traces.

A 500 kSa/s capture holds 30 million samples per minute, too many to hand to PlotDataItem
(which also needs a float64 time array of the same length). EnvelopeCurve draws straight from a
ChannelData buffer: zoomed out, one min/max pair per pixel column from the capture's pyramids;
zoomed in, the raw samples. Either way a repaint touches a few thousand points.

Also here: the X/Y marker lines and the selection box drawn over the plots, and the colours and
number formatting shared by the plots and their readouts.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal as Signal
from PyQt6.QtGui import QFont, QFontMetricsF, QPainter, QPolygonF

from .capture import ChannelData, binned_raw

RAW_MAX_SAMPLES_PER_PX = 2.0      # draw raw samples up to this density, else a min/max envelope
MARKER_MIN_SPACING = 3.0          # px between samples before "Line + points" draws markers
MARKER_SIZE = 5
BAND_MIN_WIDTH_PX = 3             # a single flagged sample stays visible when zoomed out
BOUNDS_RAW_LIMIT = 1_000_000      # y auto-range uses raw samples up to this many, else the pyramid
STYLES = ("Line", "Line + points", "Points")
TICK_DIGITS = 8                   # significant digits in y tick labels: one more than float32 resolves

CHANNEL_COLORS = {1: "#ff9f1a", 2: "#3fa9f5"}     # the instrument's Ch1 orange and Ch2 blue
QUANTITY_LABELS = {"voltage": "Voltage", "current": "Current", "power": "Power"}
X_MARKER_COLOR = "#f06bff"                         # magenta: X1, X2 (time)
Y_MARKER_COLOR = "#5df07d"                         # green: Y1, Y2 (level)
SELECTION_FILL = (255, 255, 255, 28)               # light enough to see the samples underneath
SELECTION_HANDLE_SIZE = 8                          # px


def si(value: float, unit: str, precision: int = 5) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return pg.siFormat(value, precision=precision, suffix=unit)


def colored(text: str, ch: int) -> str:
    return f"<span style='color:{CHANNEL_COLORS[ch]}; font-weight:bold'>{text}</span>"


class ValueAxis(pg.AxisItem):
    """Y axis with a constant width, so stacked plots keep identical plot areas.

    pyqtgraph sizes an axis to its widest tick label, so zooming (more or fewer decimals) would
    move the plot area sideways, and plots with different labels would no longer line up with the
    shared time axis. This one keeps room for TICK_DIGITS digits, a sign and a point, and never
    labels with more.
    """

    def __init__(self, orientation: str = "left", **kwargs):
        super().__init__(orientation, **kwargs)
        font = self.style["tickFont"] or QFont()
        widest = QFontMetricsF(font).horizontalAdvance("-." + "8" * TICK_DIGITS)
        self.setStyle(autoExpandTextSpace=False, tickTextWidth=math.ceil(widest) + 2)

    def tickStrings(self, values, scale, spacing):
        if self.logMode:
            return super().tickStrings(values, scale, spacing)
        scaled = [v * scale for v in values]
        step = spacing * scale
        whole = len(str(int(max(map(abs, scaled), default=0.0))))      # digits before the point
        places = max(0, min(math.ceil(-math.log10(step)), TICK_DIGITS - whole))
        # Rounding can leave a zero tick at -1e-17, which would print as -0.00.
        return [f"{0.0 if abs(v) < step * 1e-3 else v:.{places}f}" for v in scaled]


class EnvelopeCurve(pg.GraphicsObject):
    """One quantity ('voltage', 'current' or 'power') of one channel, drawn from a ChannelData."""

    def __init__(self, color: str):
        super().__init__()
        self._pen = pg.mkPen(color, width=1)
        self._marker_pen = pg.mkPen(color, width=MARKER_SIZE)
        self._marker_pen.setCapStyle(pg.QtCore.Qt.PenCapStyle.RoundCap)
        self._data: ChannelData | None = None
        self._quantity = "voltage"
        self._dt = 1.0
        self._style = STYLES[0]
        self._cache_key = None
        self._path = None
        self._points: tuple[np.ndarray, np.ndarray] | None = None   # marker positions, data coordinates

    # --- Data ----------------------------------------------------------------------------

    def set_source(self, data: ChannelData | None, quantity: str, dt: float) -> None:
        self._data, self._quantity, self._dt = data, quantity, dt
        self.data_changed()

    def data_changed(self) -> None:
        self._cache_key = None
        self.informViewBoundsChanged()
        self.update()

    def set_style(self, style: str) -> None:
        self._style = style
        self._cache_key = None
        self.update()

    def value_at(self, t: float) -> float | None:
        if self._data is None:
            return None
        i = round(t / self._dt)
        if 0 <= i < self._data.n:
            return float(self._data.series(self._quantity, i, i + 1)[0])
        return None

    # --- Geometry ------------------------------------------------------------------------

    def viewRangeChanged(self) -> None:
        self.prepareGeometryChange()
        self._cache_key = None
        self.update()

    def boundingRect(self) -> QRectF:
        view = self.viewRect()
        return QRectF() if view is None else view

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        data = self._data
        if data is None or data.n == 0:
            return None, None
        n, dt = data.n, self._dt
        if ax == 0:
            return 0.0, (n - 1) * dt
        i0, i1 = 0, n
        if orthoRange is not None:
            i0 = max(0, math.floor(orthoRange[0] / dt))
            i1 = min(n, math.ceil(orthoRange[1] / dt) + 1)
        if i1 <= i0:
            return None, None
        if i1 - i0 <= BOUNDS_RAW_LIMIT:
            values = data.series(self._quantity, i0, i1)
            with np.errstate(invalid="ignore"):
                lo, hi = np.fmin.reduce(values), np.fmax.reduce(values)
        else:
            pyr = data.pyramids[self._quantity]
            level = pyr.level_for((i1 - i0) / 4)
            mins, maxs = pyr.binned(level, i0, i1, np.array([i0]))
            lo, hi = mins[0], maxs[0]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None, None
        return float(lo), float(hi)

    # --- Drawing -------------------------------------------------------------------------

    def paint(self, painter: QPainter, *_) -> None:
        data = self._data
        view = self.viewRect()
        px = self.pixelWidth()
        if data is None or view is None or not px or data.n == 0:
            return
        n, dt = data.n, self._dt
        i0 = max(0, math.floor(view.left() / dt))
        i1 = min(n, math.ceil(view.right() / dt) + 1)
        if i1 <= i0:
            return
        width_px = max(1.0, view.width() / px)
        spp = (i1 - i0) / width_px                           # samples per pixel
        key = (i0, i1, round(width_px), n, self._style)
        if key != self._cache_key:
            self._build(data, i0, i1, spp, px)
            self._cache_key = key
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        if self._path is not None and self._style != "Points":
            painter.setPen(self._pen)
            painter.drawPath(self._path)
        if self._points is not None:
            # Wide-pen points drawn through the view's large scale factors smear in Qt's raster
            # engine, so map them to device pixels and draw them unscaled.
            t = painter.transform()
            x, y = self._points
            xd = x * t.m11() + y * t.m21() + t.dx()
            yd = x * t.m12() + y * t.m22() + t.dy()
            painter.save()
            painter.resetTransform()
            painter.setPen(self._marker_pen)
            painter.drawPoints(QPolygonF([QPointF(a, b) for a, b in zip(xd.tolist(), yd.tolist())]))
            painter.restore()

    def _build(self, data: ChannelData, i0: int, i1: int, spp: float, px: float) -> None:
        dt = self._dt
        self._points = None
        if spp <= RAW_MAX_SAMPLES_PER_PX:
            x = np.arange(i0, i1) * dt
            y = data.series(self._quantity, i0, i1).astype(np.float64)
            spacing_px = dt / px
            if self._style == "Points" or (self._style == "Line + points" and spacing_px >= MARKER_MIN_SPACING):
                self._points = _polygon(x, y)
        else:
            # Bins on a fixed grid of whole samples, so they don't shimmer while panning.
            width = max(1, math.ceil(spp))
            edges = np.arange((i0 // width) * width, i1, width, dtype=np.int64)
            edges[0] = max(edges[0], i0)
            pyr = data.pyramids[self._quantity]
            level = pyr.level_for(width)
            if level is None:
                lo, hi = binned_raw(data.series(self._quantity, i0, i1), edges - i0)
            else:
                lo, hi = pyr.binned(level, i0, i1, edges)
            centers = (edges + width / 2) * dt
            x = np.repeat(centers, 2)
            y = np.column_stack((lo, hi)).ravel().astype(np.float64)
            if self._style == "Points":
                self._points = _polygon(x, y)
        self._path = pg.arrayToQPath(x, y, connect="finite") if len(x) > 1 else None


def _polygon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y)
    return x[ok], y[ok]


class Bands(pg.GraphicsObject):
    """Translucent full-height bands over time intervals, at least BAND_MIN_WIDTH_PX wide.

    Bands that would touch on screen are merged, so millions of flagged runs still draw quickly.
    """

    def __init__(self, color: tuple[int, int, int, int], z: float = -10):
        super().__init__()
        self._starts = self._stops = np.empty(0)
        self._brush = pg.mkBrush(color)
        self.setZValue(z)

    def set_intervals(self, starts: np.ndarray, stops: np.ndarray) -> None:
        order = np.argsort(starts, kind="stable")
        self._starts = np.asarray(starts, float)[order]
        self._stops = np.maximum.accumulate(np.asarray(stops, float)[order]) if len(stops) else np.empty(0)
        self.update()

    def viewRangeChanged(self) -> None:
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self) -> QRectF:
        view = self.viewRect()
        return QRectF() if view is None else view

    def paint(self, painter: QPainter, *_) -> None:
        view = self.viewRect()
        px = self.pixelWidth()
        if view is None or not px or len(self._starts) == 0:
            return
        lo = np.searchsorted(self._stops, view.left())
        hi = np.searchsorted(self._starts, view.right(), side="right")
        if lo >= hi:
            return
        starts, stops = self._starts[lo:hi].copy(), self._stops[lo:hi].copy()
        pad = np.maximum(0.0, (BAND_MIN_WIDTH_PX * px - (stops - starts)) / 2)
        starts -= pad
        stops += pad
        first = np.r_[0, np.flatnonzero(starts[1:] > stops[:-1] + px) + 1]
        starts, stops = starts[first], np.maximum.reduceat(stops, first)
        painter.setPen(pg.mkPen(None))
        painter.setBrush(self._brush)
        for a, b in zip(starts, stops):
            painter.drawRect(QRectF(a, view.top(), b - a, view.height()))


class PlotViewBox(pg.ViewBox):
    """ViewBox that can draw a selection box instead of panning, and remembers where its menu opened.

    While `selecting` is set, a left-drag emits sigSelecting(start, end, finished) in data
    coordinates instead of panning. Other buttons, the wheel and axis drags work as usual.
    """

    sigSelecting = Signal(object, object, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selecting = False
        self.menu_point: QPointF | None = None      # where the context menu was opened (data coordinates)
        self._origin = QPointF()

    def mouseDragEvent(self, ev, axis=None):
        if self.selecting and axis is None and ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            if ev.isStart():        # in data coordinates, in case the layout changes during the drag
                self._origin = self.mapSceneToView(ev.buttonDownScenePos())
            self.sigSelecting.emit(self._origin, self.mapSceneToView(ev.scenePos()), ev.isFinish())
            return
        super().mouseDragEvent(ev, axis)

    def raiseContextMenu(self, ev):
        self.menu_point = self.mapSceneToView(ev.scenePos())
        super().raiseContextMenu(ev)


class MarkerLine(pg.InfiniteLine):
    """A draggable marker: a vertical (time) or horizontal (level) line labelled with its name."""

    def __init__(self, name: str, vertical: bool, color: str, dashed: bool, label_position: float):
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        super().__init__(angle=90 if vertical else 0, movable=True, pen=pg.mkPen(color, width=1, style=style),
                         hoverPen=pg.mkPen(color, width=3), label=name,
                         labelOpts={"position": label_position, "color": color, "fill": (0, 0, 0, 220)})
        self.name = name
        self.setCursor(Qt.CursorShape.SizeHorCursor if vertical else Qt.CursorShape.SizeVerCursor)
        self.setZValue(20)


class SelectionBox(pg.ROI):
    """Translucent selection rectangle with a handle on each corner to resize it.

    Dragging the inside moves it only while `translatable` is set (Select mode); otherwise a drag
    there pans the plot as usual.
    """

    def __init__(self):
        super().__init__((0, 0), (1, 1), movable=False, rotatable=False, resizable=False,
                         pen=pg.mkPen("w", width=1), hoverPen=pg.mkPen("y", width=1),
                         handlePen=pg.mkPen("w", width=1), handleHoverPen=pg.mkPen("y", width=2))
        self.handleSize = SELECTION_HANDLE_SIZE
        for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
            self.addScaleHandle((x, y), (1 - x, 1 - y))        # the opposite corner stays put
        self._brush = pg.mkBrush(SELECTION_FILL)
        self.setZValue(15)

    def set_corners(self, x0: float, y0: float, x1: float, y1: float, finish: bool = True) -> None:
        self.setPos((min(x0, x1), min(y0, y1)), update=False)
        self.setSize((abs(x1 - x0), abs(y1 - y0)), finish=finish)

    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1), lowest first."""
        (x, y), (w, h) = self.pos(), self.size()
        return min(x, x + w), max(x, x + w), min(y, y + h), max(y, y + h)

    def paint(self, p, opt, widget):
        p.fillRect(QRectF(0, 0, self.state["size"][0], self.state["size"][1]).normalized(), self._brush)
        super().paint(p, opt, widget)
