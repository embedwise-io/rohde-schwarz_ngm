"""Tick labels of the fixed-width y axis."""

from PyQt6.QtGui import QFont, QFontMetricsF

from ngm_daq.plotting import TICK_DIGITS, ValueAxis


def test_tick_labels(app):
    axis = ValueAxis()
    assert axis.tickStrings([-0.2, 5.551115123125783e-17, 0.2], 1.0, 0.2) == ["-0.2", "0.0", "0.2"]
    assert axis.tickStrings([4.19, 4.2], 1.0, 0.005) == ["4.190", "4.200"]
    assert axis.tickStrings([0.02, 0.05], 1e3, 0.01) == ["20", "50"]          # mA


def test_tick_labels_never_outgrow_the_axis(app):
    axis = ValueAxis()
    room = axis.style["tickTextWidth"]
    fm = QFontMetricsF(QFont())
    for value in (-999.9, -123.456789, 4.2, 0.5, -11.2):
        for step in (1e-12, 1e-8, 1e-6, 1e-3, 0.1, 10):
            labels = axis.tickStrings([value, value + step], 1.0, step)
            assert all(len(s.lstrip("-").replace(".", "")) <= TICK_DIGITS for s in labels), labels
            assert all(fm.horizontalAdvance(s) <= room for s in labels), labels


def test_time_axis_keeps_ten_divisions_in_roll_mode(app):
    from ngm_daq.plotting import TimeAxis
    axis = TimeAxis()
    axis.setLabel("Time", units="s")
    axis.set_division(0.1)
    assert axis.tickSpacing(4.2, 5.2, 1000) == [(0.1, 0), (0.1 / 5, 0)]         # 10 divisions in view
    assert axis.tickSpacing(4.2, 4.5, 1000)[0][0] != 0.1                         # zoomed in: automatic
    axis.set_division(None)
    assert axis.tickSpacing(4.2, 5.2, 1000)[0][0] != 0.1


def test_time_axis_stays_in_seconds_on_long_runs(app):
    from ngm_daq.plotting import TimeAxis, format_time
    axis = TimeAxis()
    axis.setLabel("Time", units="s")
    axis.setRange(36000.0, 36001.0)
    assert axis.labelUnitPrefix == "" and axis.autoSIPrefixScale == 1.0          # not ks
    assert axis.tickStrings([36000.1, 36000.2], 1.0, 0.1) == ["36000.1", "36000.2"]
    axis.setRange(0.0, 0.5)
    assert axis.labelUnitPrefix == "m"
    assert format_time(3602.0123) == "3602.0123 s" and format_time(0.0123) == "12.3 ms"
