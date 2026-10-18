"""pyqtgraph items for long, evenly sampled FastLog traces.

A 500 kSa/s capture holds 30 million samples per minute, too many to hand to PlotDataItem
(which also needs a float64 time array of the same length). EnvelopeCurve draws straight from a
ChannelData buffer: zoomed out, one min/max pair per pixel column from the capture's pyramids;
zoomed in, the raw samples. Either way a repaint touches a few thousand points.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QPainter, QPolygonF

from .capture import ChannelData, binned_raw

RAW_MAX_SAMPLES_PER_PX = 2.0      # draw raw samples up to this density, else a min/max envelope
MARKER_MIN_SPACING = 3.0          # px between samples before "Line + points" draws markers
MARKER_SIZE = 5
BAND_MIN_WIDTH_PX = 3             # a single flagged sample stays visible when zoomed out
BOUNDS_RAW_LIMIT = 1_000_000      # y auto-range uses raw samples up to this many, else the pyramid
STYLES = ("Line", "Line + points", "Points")


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
