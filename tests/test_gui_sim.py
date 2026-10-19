"""The main window against the simulator: connecting only reads, and the channel controls mirror the instrument."""

import time

import pytest
from conftest import wait_for
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest

from ngm_daq.gui import MainWindow

# Commands that change a channel; connecting and monitoring must never send any of these.
CHANGING = ("VOLT", "CURR", "OUTP", "SENS", "FLOG", "APPL", "*RST", "*TRG")


def changes(sim) -> list[str]:
    return [c for c in sim.command_log if c.upper().startswith(CHANGING)]


@pytest.fixture
def settings(tmp_path):
    s = QSettings(str(tmp_path / "ngm.conf"), QSettings.Format.IniFormat)
    # What version 0.1.0 remembered: ranges that differ from the instrument's.
    s.setValue("ch1/voltage_range", "5")
    s.setValue("ch1/current_range", "0.01")
    s.setValue("ch1/measure", True)
    return s


@pytest.fixture
def window(app, sim, settings):
    simulator, port = sim
    win = MainWindow(f"TCPIP::127.0.0.1::{port}::SOCKET", settings=settings)
    win.show()
    yield win, simulator
    win.close()


def connect(app, win):
    win._on_connect()
    wait_for(app, lambda: all(p._status is not None for p in win.channel_panels.values()))


def configure_instrument(sim):
    ch1, ch2 = sim.channels[1], sim.channels[2]
    ch1.voltage, ch1.current, ch1.output, ch1.t_on = 5.0, 0.2, True, time.monotonic()
    ch1.v_range, ch1.v_auto, ch1.i_range, ch1.i_auto = 20.0, False, 1.0, False
    ch2.voltage, ch2.current, ch2.output = 3.3, 0.05, False
    ch2.v_auto = ch2.i_auto = True


def test_controls_are_blank_until_connected(window):
    win, _ = window
    p = win.channel_panels[1]
    assert p.voltage.text() == "—" and p.current.text() == "—"
    assert p.voltage_range.currentIndex() == -1 and p.current_range.currentIndex() == -1
    assert not p.output.isEnabled() and not p.output.isChecked()


def test_connect_reads_the_instrument_and_changes_nothing(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    p1, p2 = win.channel_panels[1], win.channel_panels[2]
    assert p1.voltage.value() == pytest.approx(5.0) and p1.current.value() == pytest.approx(0.2)
    assert p1.output.isChecked() and not p2.output.isChecked()
    assert (p1.voltage_range.currentData(), p1.current_range.currentData()) == (20.0, 1.0)
    assert (p2.voltage_range.currentText(), p2.current_range.currentText()) == ("Auto", "Auto")
    assert p2.voltage.value() == pytest.approx(3.3)
    # A Run would use the instrument's ranges, not the ones remembered from the last session.
    assert p1.setup().voltage_range == 20.0 and p1.setup().current_range == 1.0
    time.sleep(1.2)                    # a few monitor cycles
    wait_for(app, lambda: True)
    assert changes(sim) == []


def test_leaving_a_field_without_editing_sends_nothing(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    p1 = win.channel_panels[1]
    sim.channels[1].voltage = 4.0      # changed on the front panel
    wait_for(app, lambda: p1.voltage.value() == pytest.approx(4.0))
    p1.voltage.editingFinished.emit()  # as when focus leaves the field
    p1.current.editingFinished.emit()
    wait_for(app, lambda: True)
    assert changes(sim) == []


def test_typed_setpoint_is_sent(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    spin = win.channel_panels[1].voltage
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), "4.5")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    wait_for(app, lambda: sim.channels[1].voltage == pytest.approx(4.5))
    assert [c for c in changes(sim) if c.startswith("VOLT")] == ["VOLT 4.5000"]
    wait_for(app, lambda: win.channel_panels[1]._status.voltage_set == pytest.approx(4.5))


def test_output_switch(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    switch = win.channel_panels[2].output
    assert not switch.isChecked()
    switch.click()
    assert switch.isChecked()                  # shows the request straight away
    wait_for(app, lambda: sim.channels[2].output)
    wait_for(app, lambda: win.channel_panels[2]._status.output)
    assert switch.isChecked()
    switch.click()
    wait_for(app, lambda: not sim.channels[2].output)
    wait_for(app, lambda: not win.channel_panels[2]._status.output and not switch.isChecked())


def test_front_panel_changes_are_followed(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    p1 = win.channel_panels[1]
    sim.channels[1].output = False
    sim.channels[1].i_range, sim.channels[1].i_auto = 0.01, False
    wait_for(app, lambda: not p1.output.isChecked() and p1.current_range.currentData() == 0.01)
    assert changes(sim) == []


def test_range_choice_is_applied(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    combo = win.channel_panels[1].current_range
    combo.setCurrentIndex(combo.findText("100 mA"))
    combo.activated.emit(combo.currentIndex())
    wait_for(app, lambda: sim.channels[1].i_range == pytest.approx(0.1) and not sim.channels[1].i_auto)
    time.sleep(0.8)
    wait_for(app, lambda: True)
    assert combo.currentText() == "100 mA"


def test_disconnect_blanks_the_controls(app, window):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    win._on_connect()                          # Disconnect
    wait_for(app, lambda: win.info is None)
    p1 = win.channel_panels[1]
    assert p1.voltage.text() == "—" and p1.voltage_range.currentIndex() == -1 and not p1.output.isChecked()


def plot_areas(app, win) -> set[tuple[float, float]]:
    for _ in range(2):                         # paint (the axes measure their labels), then relayout
        win.plot_widget.grab()
        app.processEvents()
    return {(round(r.left(), 1), round(r.right(), 1))
            for r in (p.getViewBox().sceneBoundingRect() for p in win.plots.values())}


def test_plot_areas_stay_aligned_whatever_the_tick_labels(app, window):
    """The stacked plots share the time axis under the bottom one, so their plot areas must line up
    however wide the y tick labels are, and must not move when zooming changes them."""
    win, _ = window
    win.resize(1400, 900)
    for q, (lo, hi) in {"voltage": (4.19, 4.215), "current": (-0.02, 0.05), "power": (0, 0.15)}.items():
        win.plots[q].setYRange(lo, hi, padding=0)
    areas = plot_areas(app, win)
    assert len(areas) == 1
    win.plots["voltage"].setYRange(4.2, 4.2000005, padding=0)     # as far as float32 samples resolve
    win.plots["current"].setYRange(-0.1234565, -0.1234560, padding=0)
    win.plots["power"].setYRange(4.2, 4.2 + 1e-12, padding=0)      # and far beyond
    assert plot_areas(app, win) == areas
    for p in win.plots.values():
        axis = p.getAxis("left")
        assert axis.textWidth <= axis.style["tickTextWidth"]       # the labels fit the space kept for them


def test_channel_settings_are_not_remembered(app, window, settings):
    win, _ = window
    win.close()
    settings.sync()
    assert not settings.contains("ch1/current_range") and not settings.contains("ch1/voltage_range")
    assert settings.value("ch1/measure", type=bool) is True
