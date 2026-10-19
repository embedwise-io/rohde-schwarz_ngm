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
