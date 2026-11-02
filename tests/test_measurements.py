"""Markers, the selection box and the Measurements panel, driven with the mouse where it matters."""

import time
from datetime import datetime

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget

from ngm_daq.capture import Capture, ChannelData, ChannelSettings, RunConfig, box_stats, samples_between
from ngm_daq.gui import MainWindow
from ngm_daq.instrument import ChannelSetup, DeviceInfo, SampleRate

DEVICE = DeviceInfo("TCPIP::127.0.0.1::5025::SOCKET", "Rohde&Schwarz", "NGM202", "3638.4472k03", "103062",
                    "04.013 0092C027A93", frozenset({"K103", "K104"}))
N = 10_000                          # samples at 1 kSa/s: 10 s
MOVE_INTERVAL = 0.015               # s between mouse moves: pyqtgraph drops moves faster than ~100 per second


def make_capture() -> Capture:
    k = np.arange(N)
    cfg = RunConfig((ChannelSetup(1, 5.0, 0.01), ChannelSetup(2)), SampleRate.R1K, duration=N / 1e3)
    channels = {}
    for c in (1, 2):
        ch = ChannelData(ChannelSettings(c, 4.2, 0.1, True, 5.0, 0.01), N)
        v = np.full(N, 4.2 if c == 1 else 3.0)
        i = (k % 100) * 1e-4 if c == 1 else np.full(N, 0.002)      # CH1: 0 ... 9.9 mA sawtooth
        ch.append(np.column_stack((v, i)).astype(np.float32))
        channels[c] = ch
    return Capture(DEVICE, cfg, 1e-3, datetime(2026, 11, 1, 12, 0).astimezone(), channels)


@pytest.fixture
def win(app, tmp_path):
    w = MainWindow(settings=QSettings(str(tmp_path / "ngm.conf"), QSettings.Format.IniFormat))
    w.resize(1500, 950)
    w.show()
    w._show(make_capture(), fit=True)
    settle(app)
    yield w
    w.close()


def settle(app):
    for _ in range(3):
        app.processEvents()


def readouts(app, w) -> tuple[str, str]:
    settle(app)
    m = w.measurements
    m.update_readouts()
    return m.markers_label.text(), m.selection_label.text()


def point(w, quantity: str, x: float, y: float) -> QPoint:
    """Where a data point of a plot is, in the plot widget's viewport."""
    scene = w.plots[quantity].getViewBox().mapViewToScene(QPointF(x, y))
    return w.plot_widget.mapFromScene(scene)


def send(widget, kind, pos: QPoint, button, buttons):
    global_pos = QPointF(widget.mapToGlobal(pos))
    QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(pos), global_pos, button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


def drag(app, w, a: QPoint, b: QPoint, button=Qt.MouseButton.LeftButton):
    vp = w.plot_widget.viewport()
    send(vp, QEvent.Type.MouseMove, a, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)    # hover first
    send(vp, QEvent.Type.MouseButtonPress, a, button, button)
    for k in range(1, 11):
        time.sleep(MOVE_INTERVAL)
        p = a + (b - a) * (k / 10)
        send(vp, QEvent.Type.MouseMove, p, Qt.MouseButton.NoButton, button)
        app.processEvents()
    send(vp, QEvent.Type.MouseButtonRelease, b, button, Qt.MouseButton.NoButton)
    settle(app)


def x_range(w) -> list[float]:
    return w.plots["voltage"].getViewBox().viewRange()[0]


# --- Markers -----------------------------------------------------------------------------

def test_markers_button_places_and_clears_them(app, win):
    m = win.measurements
    assert not win.measure_dock.isVisible()
    x0, x1 = x_range(win)
    win.markers_btn.click()
    settle(app)
    assert win.markers_btn.isChecked() and win.measure_dock.isVisible()
    assert m.marker("X1") == pytest.approx(x0 + 0.25 * (x1 - x0))
    assert m.marker("X2") == pytest.approx(x0 + 0.75 * (x1 - x0))
    y0, y1 = win.plots["current"].getViewBox().viewRange()[1]       # Y markers start in the current plot
    assert y0 < m.marker("Y1") < m.marker("Y2") < y1
    assert m._y_lines["Y1"].getViewBox() is win.plots["current"].getViewBox()
    assert all(line.isVisible() for lines in m._x_lines.values() for line in lines.values())
    win.markers_btn.click()
    settle(app)
    assert not m.has_markers and not win.measure_dock.isVisible()
    assert m.marker("X1") is None
    assert not any(line.isVisible() for lines in m._x_lines.values() for line in lines.values())


def test_readout_values_deltas_and_reciprocals(app, win):
    m = win.measurements
    m.place_marker("X1", "current", QPointF(2.0123, 0))     # sample 2012: 1.2 mA
    m.place_marker("X2", "current", QPointF(2.0573, 0))     # sample 2057: 5.7 mA
    m.place_marker("Y1", "current", QPointF(0, 0.001))
    m.place_marker("Y2", "current", QPointF(0, 0.005))
    markers, _ = readouts(app, win)
    for text in ("2.0123 s", "2.0573 s", "45 ms",           # X1, X2, ΔX
                 "22.2222 Hz",                              # 1/ΔX
                 "1.2 mA", "5.7 mA", "4.5 mA",              # CH1 I at X1, X2 and the difference
                 "4 mA", "250.000 A⁻¹"):                    # ΔY and 1/ΔY
        assert text in markers, text


def test_x_marker_lines_move_together(app, win):
    m = win.measurements
    win.markers_btn.click()
    m._x_lines["X1"]["power"].setValue(3.5)                  # as when dragged in the power plot
    assert all(line.value() == pytest.approx(3.5) for line in m._x_lines["X1"].values())
    assert m.marker("X1") == pytest.approx(3.5)


def test_dragging_a_marker_moves_it(app, win):
    m = win.measurements
    m.place_marker("X1", "current", QPointF(3.0, 0))
    settle(app)
    y = 0.005
    drag(app, win, point(win, "current", 3.0, y), point(win, "current", 6.0, y))
    assert m.marker("X1") == pytest.approx(6.0, abs=0.05)
    assert x_range(win) == pytest.approx([0, (N - 1) * 1e-3], abs=1e-3)     # the drag didn't pan


def test_right_click_places_a_marker_there(app, win):
    m = win.measurements
    vp = win.plot_widget.viewport()
    where = point(win, "power", 4.0, 0.02)
    send(vp, QEvent.Type.MouseButtonPress, where, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    send(vp, QEvent.Type.MouseButtonRelease, where, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    settle(app)
    menu = win.plots["power"].getViewBox().menu
    assert menu.isVisible()
    action = next(a for a in menu.actions() if a.text() == "Place Y2 Here")
    action.trigger()
    menu.close()
    assert m.y_quantity == "power" and m.marker("Y2") == pytest.approx(0.02, rel=0.02)
    assert m._y_lines["Y1"].getViewBox() is win.plots["power"].getViewBox()     # Y1 moved with it
    assert m.y_plot.currentData() == "power" and win.measure_dock.isVisible()


def test_y_markers_follow_the_plot_choice(app, win):
    m = win.measurements
    win.markers_btn.click()
    m.y_plot.setCurrentIndex(m.y_plot.findData("voltage"))
    m.y_plot.activated.emit(m.y_plot.currentIndex())
    assert m.y_quantity == "voltage"
    vb = win.plots["voltage"].getViewBox()
    y0, y1 = vb.viewRange()[1]
    assert all(line.getViewBox() is vb and y0 <= line.value() <= y1 for line in m._y_lines.values())
    markers, _ = readouts(app, win)
    assert "Voltage" in markers and "V⁻¹" in markers


def test_y_marker_plot_is_remembered(app, win, tmp_path):
    win.measurements.move_y_markers("power")
    win.close()
    settings = QSettings(str(tmp_path / "ngm.conf"), QSettings.Format.IniFormat)
    again = MainWindow(settings=settings)
    assert again.measurements.y_quantity == "power"
    again.close()


# --- Selection box -----------------------------------------------------------------------

def test_select_drag_draws_a_box_and_does_not_pan(app, win):
    m = win.measurements
    before = x_range(win)
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.007), point(win, "current", 4.0, 0.003))
    assert x_range(win) == pytest.approx(before)
    assert m.has_box and m.box_quantity == "current" and win.measure_dock.isVisible()
    x0, x1, y0, y1 = m.box.bounds()
    assert (x0, x1) == pytest.approx((2.0, 4.0), abs=0.03)
    assert (y0, y1) == pytest.approx((0.003, 0.007), abs=1e-4)
    _, selection = readouts(app, win)
    s = box_stats(win.capture.channels[1], "current", *samples_between(x0, x1, 1e-3, N), y0, y1)
    assert s.count > 0 and f"{s.count:,}" in selection
    assert "Points in box" in selection and "Mean" in selection


def test_box_zoom_and_select_are_exclusive(app, win):
    win.box_zoom_btn.click()
    win.select_btn.click()
    assert not win.box_zoom_btn.isChecked()
    win.box_zoom_btn.click()
    assert not win.select_btn.isChecked()
    assert not any(p.getViewBox().selecting for p in win.plots.values())


def test_corner_handle_resizes_the_box(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.003), point(win, "current", 4.0, 0.007))
    win.select_btn.click()                                   # resizing works without Select too
    drag(app, win, point(win, "current", 4.0, 0.007), point(win, "current", 6.0, 0.008))
    x0, x1, y0, y1 = m.box.bounds()
    assert (x0, y0) == pytest.approx((2.0, 0.003), abs=0.03)          # the opposite corner stayed
    assert (x1, y1) == pytest.approx((6.0, 0.008), abs=0.05)


def test_inside_of_the_box_moves_it_only_in_select_mode(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.003), point(win, "current", 4.0, 0.007))
    drag(app, win, point(win, "current", 3.0, 0.005), point(win, "current", 5.0, 0.005))   # move it
    x0, x1, _, _ = m.box.bounds()
    assert (x0, x1) == pytest.approx((4.0, 6.0), abs=0.05)
    win.plots["voltage"].getViewBox().setXRange(1.0, 9.0, padding=0)    # room to pan
    settle(app)
    win.select_btn.click()                                   # Select off: dragging inside pans
    drag(app, win, point(win, "current", 5.0, 0.005), point(win, "current", 4.0, 0.005))
    assert x_range(win) == pytest.approx([2.0, 10.0], abs=0.05)
    assert m.box.bounds()[:2] == pytest.approx((x0, x1))


def test_escape_and_clear_remove_the_box(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "voltage", 1.0, 4.1), point(win, "voltage", 2.0, 4.3))
    assert m.box_quantity == "voltage"
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert not m.has_box and not win.measure_dock.isVisible()


def test_a_new_box_replaces_the_old_one(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "voltage", 1.0, 4.1), point(win, "voltage", 2.0, 4.3))
    drag(app, win, point(win, "power", 5.0, 0.01), point(win, "power", 7.0, 0.03))
    assert m.box_quantity == "power" and m.box.getViewBox() is win.plots["power"].getViewBox()
    assert m.box not in win.plots["voltage"].getViewBox().addedItems


def test_tiny_drag_makes_no_box(app, win):
    m = win.measurements
    vb = win.plots["current"].getViewBox()
    px, py = vb.viewPixelSize()
    vb.sigSelecting.emit(QPointF(2.0, 0.005), QPointF(2.0 + px, 0.005 + 20 * py), True)
    assert not m.has_box


def test_selection_statistics_wait_for_a_complete_capture(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.003), point(win, "current", 4.0, 0.007))
    m.set_capture(win.capture, capturing=True)
    _, selection = readouts(app, win)
    assert "when the capture is complete" in selection and "Points in box" not in selection


def test_large_selection_waits_for_the_drag_to_pause(app, win, monkeypatch):
    from ngm_daq import measurements
    monkeypatch.setattr(measurements, "LIVE_SAMPLES", 1000)
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.003), point(win, "current", 4.0, 0.007))
    m.box.set_corners(1.0, 0.003, 5.0, 0.007)               # as while dragging: 4,000 samples per channel
    m.update_readouts(final=False)
    assert "Updating…" in m.selection_label.text()
    m.update_readouts()                                      # the drag paused
    assert "Points in box" in m.selection_label.text()
    m.update_readouts(final=False)                           # unchanged: no recomputation needed
    assert "Points in box" in m.selection_label.text()


# --- Readouts under the plots ------------------------------------------------------------

def hover(app, w, t: float):
    w._on_mouse_moved(w.plots["current"].getViewBox().mapViewToScene(QPointF(t, 0.005)))
    settle(app)


def sizes(w) -> tuple[int, int, int]:
    return w.cursor_label.width(), w.stats_label.height(), w.plot_widget.height()


@pytest.mark.parametrize("width", [1500, 1000])          # 1000: the statistics wrap
def test_readouts_under_the_plots_keep_their_size(app, win, width):
    """The cursor readout and the statistics share a row. If either changed size with the values
    shown, the plots would resize as the mouse moves, worst with the Measurements panel open."""
    win.resize(width, 950)
    win.markers_btn.click()                                  # the panel narrows the plots
    settle(app)
    hover(app, win, 2.0)                                     # 0 A
    before = sizes(win)
    for t in (2.057, 2.0123, 5.0991, 9.999):                 # 5.7 mA, 1.2 mA, 9.1 mA, ...
        hover(app, win, t)
        assert sizes(win) == before, t
    win.plots["voltage"].getViewBox().setXRange(2.0, 2.1, padding=0)   # other statistics
    win._update_stats()
    settle(app)
    assert sizes(win)[1:] == before[1:]
    label = win.stats_label
    assert label.height() >= label.heightForWidth(label.width())       # the text fits



# --- Docked or popped out ----------------------------------------------------------------

def test_pop_out_moves_the_panel_into_its_own_window(app, win):
    win.markers_btn.click()
    settle(app)
    assert win.measure_dock.isVisible() and not win.measure_window.isVisible()
    win.pop_out_btn.click()
    settle(app)
    assert win.measure_window.isVisible() and not win.measure_dock.isVisible()
    assert win.measurements.window() is win.measure_window and win.pop_out_btn.text() == "Dock"
    win.pop_out_btn.click()
    settle(app)
    assert win.measure_dock.isVisible() and not win.measure_window.isVisible()
    assert win.measurements.window() is win and win.pop_out_btn.text() == "Pop Out"


def test_closing_the_window_clears_markers_and_selection(app, win):
    m = win.measurements
    win.select_btn.click()
    drag(app, win, point(win, "current", 2.0, 0.003), point(win, "current", 4.0, 0.007))
    win.markers_btn.click()
    win.pop_out_btn.click()
    settle(app)
    win.measure_window.close()
    settle(app)
    assert not m.has_markers and not m.has_box and not win.markers_btn.isChecked()
    assert not win.measure_window.isVisible() and not win.measure_dock.isVisible()
    win.markers_btn.click()                                  # still popped out: the window opens again
    settle(app)
    assert win.measure_window.isVisible() and not win.measure_dock.isVisible()


def test_the_dock_does_not_float(app, win):
    """Qt moves a floating dock itself, which Wayland doesn't allow: it would be stuck in place."""
    assert win.measure_dock.features() == QDockWidget.DockWidgetFeature.NoDockWidgetFeatures


def test_popped_out_is_remembered(app, win, tmp_path):
    win.markers_btn.click()
    win.pop_out_btn.click()
    win.settings.setValue("window_state", b"floating dock")  # saved by an earlier build
    win.close()
    settings = QSettings(str(tmp_path / "ngm.conf"), QSettings.Format.IniFormat)
    assert not settings.contains("window_state")
    again = MainWindow(settings=settings)
    again.show()
    settle(app)
    assert again._popped_out and again.pop_out_btn.text() == "Dock"
    assert not again.measure_window.isVisible()              # nothing to show until there are markers
    again.close()


def test_value_columns_fit_the_longest_values(app, win):
    m = win.measurements
    fm = m.markers_label.fontMetrics()
    for text in ("672.2466 ms", "-888.888 mV", "-16.2877 ms", "-888.88 mA", "-118.9 nW"):
        assert fm.horizontalAdvance("\u00a0" + text) + 4 <= m._column, text     # 4: cell padding



def test_selection_counts_and_time_in_box(app, win):
    m = win.measurements
    # 2.000 ... 3.999 s: 2,000 samples. CH1's sawtooth is between 3 and 7 mA for 41 of every 100;
    # CH2 is a steady 2 mA, below the box.
    m.drag_box("current", QPointF(2.0, 0.00295), QPointF(3.9995, 0.00705), True)
    _, selection = readouts(app, win)
    assert selection.index("Points in time range") < selection.index("Points in box")
    for text in ("2,000", "820", "820 ms (41.0%)", "0 s (0.0%)"):
        assert text in selection, text
