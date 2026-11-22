"""The worker thread against the simulator (tools/ngm_simulator.py), in-process."""

import time

import numpy as np
import pytest
from conftest import wait_for

from ngm_daq.capture import RunConfig, TriggerMode
from ngm_daq.instrument import ChannelSetup, SampleRate
from ngm_daq.worker import InstrumentWorker


class Recorder:
    def __init__(self, worker: InstrumentWorker):
        self.events = []
        self.captures = []
        self.status = {}
        worker.connected.connect(lambda info: self.events.append("connected"))
        worker.failed.connect(lambda title, msg, kind: self.events.append(("failed", title, msg, kind)))
        worker.armed.connect(lambda: self.events.append("armed"))
        worker.acquired.connect(self.captures.append)
        worker.run_finished.connect(lambda: self.events.append("finished"))
        worker.status.connect(self.status.update)


@pytest.fixture
def connected(app, sim):
    simulator, port = sim
    worker = InstrumentWorker()
    rec = Recorder(worker)
    worker.open_session(f"TCPIP::127.0.0.1::{port}::SOCKET", "socket")
    wait_for(app, lambda: "connected" in rec.events and rec.status)
    yield worker, rec, simulator
    worker.shutdown()


def run(app, worker, rec, cfg, timeout=15.0):
    worker.run(cfg)
    wait_for(app, lambda: "finished" in rec.events, timeout)


def test_single_capture_500k(app, connected):
    worker, rec, _ = connected
    cfg = RunConfig((ChannelSetup(1, 5.0, 0.1),), SampleRate.R500K, duration=1.0)
    run(app, worker, rec, cfg)
    assert not [e for e in rec.events if isinstance(e, tuple)]
    cap = rec.captures[-1]
    assert cap.complete and cap.n == 500_000
    assert cap.interval == pytest.approx(2e-6)
    assert len(cap.channels[1].missing) == 0


def test_both_channels_aligned_captures(app, connected):
    worker, rec, _ = connected
    cfg = RunConfig((ChannelSetup(1), ChannelSetup(2)), SampleRate.R250K, duration=0.6)
    run(app, worker, rec, cfg)
    cap = rec.captures[-1]
    assert set(cap.channels) == {1, 2} and cap.complete and cap.n == 150_000


def test_auto_mode_is_seamless(app, connected):
    worker, rec, _ = connected
    cfg = RunConfig((ChannelSetup(1),), SampleRate.R50K, duration=0.3, mode=TriggerMode.AUTO)
    worker.run(cfg)
    wait_for(app, lambda: len(rec.captures) >= 4)
    worker.request_stop()
    wait_for(app, lambda: "finished" in rec.events)
    complete = [c for c in rec.captures if c.complete]
    assert [c.number for c in complete[:4]] == [1, 2, 3, 4]
    assert [c.first_sample for c in complete[:4]] == [0, 15_000, 30_000, 45_000]
    assert all(len(c.channels[1].missing) == 0 for c in complete)


def test_manual_trigger(app, connected):
    worker, rec, _ = connected
    cfg = RunConfig((ChannelSetup(1),), SampleRate.R10K, duration=0.3, mode=TriggerMode.MANUAL)
    worker.run(cfg)
    wait_for(app, lambda: "armed" in rec.events)
    time.sleep(0.3)
    assert not rec.captures
    worker.request_trigger()
    wait_for(app, lambda: rec.captures)
    wait_for(app, lambda: rec.events.count("armed") >= 2)       # re-armed for the next trigger
    worker.request_stop()
    wait_for(app, lambda: "finished" in rec.events)
    assert rec.captures[0].complete


def test_stall_becomes_a_gap(app, connected, monkeypatch):
    worker, rec, _ = connected
    psu = worker._psu
    original = psu.fetch
    calls = {"n": 0}

    def slow_fetch(ch):
        calls["n"] += 1
        if calls["n"] == 2:
            time.sleep(0.62)          # longer than two chunk periods: the instrument replaces 2 chunks
        return original(ch)

    monkeypatch.setattr(psu, "fetch", slow_fetch)
    cfg = RunConfig((ChannelSetup(1),), SampleRate.R10K, duration=2.0)
    run(app, worker, rec, cfg)
    ch = rec.captures[-1].channels[1]
    missing = ch.missing.array()
    assert len(missing) == 1
    start, stop = missing[0]
    assert stop - start == 2 * 2500 and start % 2500 == 0
    assert np.isnan(ch.voltage[start:stop]).all()


def test_output_on_at_start_captures_turn_on(app, connected):
    worker, rec, sim = connected
    worker.set_voltage(1, 4.2)
    worker.set_current(1, 0.1)
    cfg = RunConfig((ChannelSetup(1, 5.0, 0.1),), SampleRate.R50K, duration=1.0,
                    output_on_at_start=True, output_off_at_end=True)
    run(app, worker, rec, cfg)
    ch = rec.captures[-1].channels[1]
    assert not ch.settings.output
    v = ch.voltage[:ch.n]
    rise = np.flatnonzero(v > 2.1)
    assert 0.2 * 50e3 < rise[0] < 0.9 * 50e3        # baseline first, then the turn-on
    assert ch.current_clipped.array().size or ch.current[:ch.n].max() >= 0.099   # inrush hits the limit
    assert not sim.channels[1].output                # switched off at the end


def test_fastlog_left_running_is_stopped_first(app, connected):
    worker, rec, sim = connected
    sim.channels[2].flog = True          # CH2 left logging at 500k (e.g. to USB from the front panel):
    sim.channels[2].rate = 500e3         # starting CH1 on top would exceed the limit and crash FastLog
    cfg = RunConfig((ChannelSetup(1),), SampleRate.R500K, duration=1.0)
    run(app, worker, rec, cfg)
    # The worker stops CH2 first, so the instrument never sees an illegal combination.
    assert not sim.broken
    assert rec.captures[-1].complete


def test_broken_fastlog_is_detected(app, connected):
    worker, rec, sim = connected
    sim.broken = True
    cfg = RunConfig((ChannelSetup(1),), SampleRate.R1K, duration=1.0)
    run(app, worker, rec, cfg)
    failures = [e for e in rec.events if isinstance(e, tuple)]
    assert failures and failures[0][3] == "fastlog_stopped"


def test_repeated_dual_channel_runs(app, connected):
    """The instrument ignores every other *TRG; consecutive dual-channel runs must all start."""
    worker, rec, sim = connected
    cfg = RunConfig((ChannelSetup(1), ChannelSetup(2)), SampleRate.R50K, duration=0.3)
    for _ in range(3):
        rec.events.clear()
        run(app, worker, rec, cfg)
        assert not [e for e in rec.events if isinstance(e, tuple)], rec.events
        assert rec.captures[-1].complete
    assert len(rec.captures) == 3


def test_setpoint_change_during_capture(app, connected):
    worker, rec, sim = connected
    worker.set_voltage(1, 4.2)
    worker.set_output(1, True)
    cfg = RunConfig((ChannelSetup(1, 5.0, 0.1),), SampleRate.R50K, duration=1.5)
    worker.run(cfg)
    wait_for(app, lambda: rec.status.get(1) is not None and rec.status[1].output)
    time.sleep(0.6)
    worker.set_voltage(1, 3.3)                # runs between polls while FastLog streams
    wait_for(app, lambda: "finished" in rec.events)
    ch = rec.captures[-1].channels[1]
    assert rec.captures[-1].complete and len(ch.missing) == 0
    v = ch.voltage[:ch.n]
    assert v[:int(0.3 * 50e3)].mean() == pytest.approx(4.2, abs=0.02)
    assert v[-int(0.3 * 50e3):].mean() == pytest.approx(3.3, abs=0.02)
    assert rec.status[1].voltage_set == pytest.approx(3.3)
    worker.set_output(1, False)


def test_roll_mode_keeps_only_the_span(app, connected):
    worker, rec, _ = connected
    seen = []
    worker.progress.connect(seen.append)
    cfg = RunConfig((ChannelSetup(1), ChannelSetup(2)), SampleRate.R10K, duration=0.5, mode=TriggerMode.ROLL)
    worker.run(cfg)
    wait_for(app, lambda: seen and seen[-1].offset + seen[-1].n >= 40_000)        # 4 s: several times the buffer
    worker.request_stop()
    wait_for(app, lambda: "finished" in rec.events)
    assert not [e for e in rec.events if isinstance(e, tuple)], rec.events
    # While running: bounded buffers that slide along the stream, never losing the span.
    capacity = 5000 + 10_000 + 8 * 2500                 # span + 1 s display margin + slack
    assert all(c.target == capacity for c in seen)
    offsets = [c.offset for c in seen]
    assert offsets == sorted(offsets) and len(set(offsets)) >= 3 and offsets[0] == 0
    assert all(c.n >= 5000 + 10_000 for c in seen if c.offset)      # a rolled buffer keeps span + margin
    # At the end: exactly the span, on both channels, ending where the stream ended.
    final = rec.captures[-1]
    assert final.n == 5000 and final.complete and final.offset >= 35_000
    assert final.offset + final.n >= seen[-1].offset + seen[-1].n
    assert final.t0 == pytest.approx(final.offset * 1e-4)
    for ch in final.channels.values():
        assert ch.n == 5000 and len(ch.missing) == 0 and np.isfinite(ch.voltage).all()


def test_roll_keeps_samples_and_flags_in_place():
    from ngm_daq.capture import ChannelData, ChannelSettings
    ch = ChannelData(ChannelSettings(1, 4.2, 0.1, True, 5.0, 0.01), 1000)
    k = np.arange(600)
    i = np.full(600, 0.002)
    i[400:410] = 0.0113391                                  # clipped on the 10 mA range
    ch.append(np.column_stack((k * 1e-3, i)).astype(np.float32))
    ch.append_missing(100)
    rolled = ch.rolled(drop=384)
    assert rolled.n == 316 and rolled.capacity == 1000
    assert rolled.voltage[:216] == pytest.approx(np.arange(384, 600) * 1e-3, rel=1e-6)
    assert rolled.current_clipped.array().tolist() == [[16, 26]]
    assert rolled.missing.array().tolist() == [[216, 316]] and np.isnan(rolled.voltage[216:316]).all()
    lo, hi = rolled.pyramids["voltage"].binned(0, 0, 216, np.array([0]))
    assert (lo[0], hi[0]) == pytest.approx((0.384, 0.599), rel=1e-6)
    trimmed = ch.rolled(drop=650, capacity=50)
    assert trimmed.n == 50 and trimmed.full and trimmed.missing.array().tolist() == [[0, 50]]


def test_copying_in_steps_matches_rolling_at_once():
    from ngm_daq.capture import ChannelData, ChannelSettings
    rng = np.random.default_rng(0)
    source = ChannelData(ChannelSettings(1, 4.2, 0.1, True, 5.0, 0.01), 5000)
    source.append(rng.normal(1.0, 0.1, (1500, 2)).astype(np.float32))
    source.append_missing(300)
    new = ChannelData(source.settings, 5000)
    assert not new.copy_from(source, 700, 400)               # 400 of 1,100 so far
    source.append(rng.normal(1.0, 0.1, (900, 2)).astype(np.float32))     # the old buffer keeps filling
    while not new.copy_from(source, 700, 400):
        pass
    once = source.rolled(700)
    assert new.n == once.n == 2000
    assert np.array_equal(new.voltage[:2000], once.voltage[:2000], equal_nan=True)
    assert new.missing.array().tolist() == once.missing.array().tolist() == [[800, 1100]]
    for q in ("voltage", "current", "power"):
        a, b = new.pyramids[q], once.pyramids[q]
        assert a.counts == b.counts
        assert all(np.array_equal(x[:c], y[:c], equal_nan=True) for x, y, c in zip(a.mins, b.mins, a.counts))
