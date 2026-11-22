"""The main window against the simulator: connecting only reads, and the channel controls mirror the instrument."""

import time

import pytest
from conftest import wait_for
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest

from ngm_daq import export
from ngm_daq.capture import TriggerMode
from ngm_daq.gui import MainWindow
from ngm_daq.instrument import SampleRate

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


def test_roll_mode_scrolls_and_keeps_only_the_span(app, window, tmp_path):
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    win.trigger_mode.setCurrentIndex(win.trigger_mode.findData(TriggerMode.ROLL))
    win.rate.setCurrentIndex(win.rate.findData(SampleRate.R10K))
    win.timebase.setCurrentIndex(win.timebase.findData(0.1))            # 100 ms/div: a 1 s span
    assert win.timebase_row.isVisible() and not win.duration.isVisible()
    assert win.span_label.text() == "span 1 s" and win.current_config().duration == pytest.approx(1.0)
    vb = win.plots["voltage"].getViewBox()
    win._on_run()
    wait_for(app, lambda: win.capture is not None and win.capture.offset > 0, 30)     # the buffer has rolled
    wait_for(app, lambda: vb.viewRange()[0][1] > 2.0)
    cap = win.capture
    x0, x1 = vb.viewRange()[0]
    assert x1 - x0 == pytest.approx(1.0)                                # always the span ...
    assert cap.time_of(cap.n) - 0.7 < x1 <= cap.time_of(cap.n)          # ... ending just behind the newest sample
    assert not vb.state["mouseEnabled"][0] and vb.state["mouseEnabled"][1]
    assert cap.target == 40_000                                         # bounded, however long it runs
    assert win.plots["power"].getAxis("bottom").tickSpacing(x0, x1, 1000)[0][0] == pytest.approx(0.1)
    assert "CH1" in win.stats_label.text()                              # statistics follow while rolling
    assert not win.save_all_btn.isEnabled()
    win._on_run()                                                       # Stop
    wait_for(app, lambda: not win._running)
    cap = win.capture
    assert cap.n == 10_000 and cap.offset >= 10_000                     # exactly the span remains
    x0, x1 = vb.viewRange()[0]
    assert (x0, x1) == pytest.approx((cap.t0, cap.time_of(cap.n - 1)))
    assert vb.state["mouseEnabled"][0] and win.save_all_btn.isEnabled()
    path = tmp_path / "roll.csv"
    assert export.save(path, cap) == 10_000
    rows = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    assert float(rows[1].split(",")[0]) == pytest.approx(cap.t0)        # times count from the start of the run
    assert f"# first_sample_index: {cap.offset}" in path.read_text()


def test_roll_mode_left_edge_never_blank_after_a_rollover(app, window):
    """The display runs behind the newest sample, so each rollover must keep more than the span:
    otherwise the plots' left edge has no data behind it for a moment, which looks like a dropout."""
    from ngm_daq.gui import ROLL_LATENCY
    from ngm_daq.worker import ROLL_DISPLAY_MARGIN
    assert ROLL_DISPLAY_MARGIN > ROLL_LATENCY
    win, sim = window
    configure_instrument(sim)
    connect(app, win)
    win.trigger_mode.setCurrentIndex(win.trigger_mode.findData(TriggerMode.ROLL))
    win.rate.setCurrentIndex(win.rate.findData(SampleRate.R10K))
    win.timebase.setCurrentIndex(win.timebase.findData(0.02))           # 0.2 s span: rolls every ~1.5 s
    vb = win.plots["voltage"].getViewBox()
    offsets, blank, frames = set(), [], 0
    win._on_run()
    deadline = time.monotonic() + 30
    while len(offsets) < 4 and time.monotonic() < deadline:           # 3 rollovers
        app.processEvents()
        cap = win.capture
        if cap is not None and win._capturing and cap.n:
            x0 = vb.viewRange()[0][0]
            if x0 >= 0:
                frames += 1
                offsets.add(cap.offset)
                if cap.t0 > x0 + 1e-9:
                    blank.append(cap.t0 - x0)
        time.sleep(0.01)
    win._on_run()
    wait_for(app, lambda: not win._running)
    assert len(offsets) >= 4 and frames > 100
    assert blank == []
    assert win.capture.n == 2000                                        # Stop still keeps exactly the span


# --- Address list ------------------------------------------------------------------------

def scanned_window(app, tmp_path, monkeypatch, found, history, identities=(), resource=None):
    from ngm_daq import discovery
    monkeypatch.setattr(discovery, "discover", lambda rm=None, browse_time=None: list(found))
    settings = QSettings(str(tmp_path / "addr.conf"), QSettings.Format.IniFormat)
    settings.setValue("history", list(history))
    settings.setValue("identities", [f"{h}|{i}" for h, i in identities])
    w = MainWindow(resource, settings=settings)
    w.show()
    wait_for(app, lambda: not w._scanning)
    return w


def items(w):
    return [(w.address.itemText(i), w.address.itemData(i),
             w.address.itemData(i, Qt.ItemDataRole.ForegroundRole) is not None) for i in range(w.address.count())]


def test_startup_scan_lists_instruments_with_their_identity(app, tmp_path, monkeypatch):
    from ngm_daq.discovery import FoundInstrument
    found = [FoundInstrument("10.101.0.120", "Rohde&Schwarz", "NGM202", "103062")]
    w = scanned_window(app, tmp_path, monkeypatch, found,
                       history=["10.101.0.121", "TCPIP::10.101.0.120::5025::SOCKET"],
                       identities=[("10.101.0.121", "NGM201 · S/N 100001")])
    try:
        assert items(w) == [
            ("10.101.0.120 — NGM202 · S/N 103062", "10.101.0.120", False),
            ("10.101.0.121 — NGM201 · S/N 100001 · not found", "10.101.0.121", True),   # greyed
        ]
        # The last address used (10.101.0.121) didn't answer, so the box shows one that did.
        assert w.address.currentText() == "10.101.0.120" and w.connect_btn.isEnabled()
        w.address.setCurrentIndex(1)
        w.address.activated.emit(1)                          # picking an entry puts just its address in
        assert w.address.currentText() == "10.101.0.121" and w._address() == "10.101.0.121"
    finally:
        w.close()


def test_startup_scan_prefers_the_last_instrument_used(app, tmp_path, monkeypatch):
    from ngm_daq.discovery import FoundInstrument
    found = [FoundInstrument("10.101.0.120", "Rohde&Schwarz", "NGM202", "103062"),
             FoundInstrument("10.101.0.121", "Rohde&Schwarz", "NGM201", "100001")]
    w = scanned_window(app, tmp_path, monkeypatch, found, history=["10.101.0.121", "10.101.0.120"])
    try:
        assert w.address.currentText() == "10.101.0.121"
    finally:
        w.close()


def test_nothing_found_leaves_the_box_empty(app, tmp_path, monkeypatch):
    w = scanned_window(app, tmp_path, monkeypatch, [], history=["10.101.0.120"])
    try:
        assert w.address.currentText() == "" and not w.connect_btn.isEnabled()
        assert items(w) == [("10.101.0.120 — not found", "10.101.0.120", True)]
    finally:
        w.close()


def test_command_line_address_wins(app, tmp_path, monkeypatch):
    from ngm_daq.discovery import FoundInstrument
    w = scanned_window(app, tmp_path, monkeypatch, [FoundInstrument("10.101.0.120", "Rohde&Schwarz", "NGM202", "X")],
                       history=[], resource="10.101.0.99")
    try:
        assert w.address.currentText() == "10.101.0.99"
    finally:
        w.close()


def test_connecting_records_the_identity(app, window):
    win, _ = window
    connect(app, win)
    assert win._identities()["127.0.0.1"] == "NGM202 · S/N 000000"


def test_a_scan_ending_during_a_connection_leaves_it_alone(app, window):
    win, _ = window
    address = win._address()
    win._on_scan()                                           # scan and connect queued together
    win._on_connect()
    wait_for(app, lambda: win.info is not None and not win._scanning)
    assert win.address.currentText() == address and win.connect_btn.text() == "Disconnect"
    assert not win.address.isEnabled()
