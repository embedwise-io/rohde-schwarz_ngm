"""The instrument thread: owns the session, streams FastLog, and runs commands from the GUI.

Everything that talks to the instrument runs on one Python thread, fed by a command queue. While a
capture streams, the thread polls for chunks every few milliseconds and runs queued commands
(setpoint changes, output switching) between polls, so they never hold up the data. Results go
back to the GUI as Qt signals, which Qt delivers on the GUI thread.

A chunk must be read within 250 ms of becoming available or the instrument replaces it. The
thread only blocks on the instrument, so the GUI being busy doesn't cost data; anything lost
anyway is detected (GapTracker) and stored as a gap.
"""

from __future__ import annotations

import dataclasses
import math
import queue
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pyvisa
from PyQt6.QtCore import QObject, pyqtSignal as Signal
from RsInstrument import RsInstrException, TimeoutException

from .capture import Capture, ChannelData, ChannelSettings, GapTracker, RunConfig, TriggerMode
from .instrument import (
    FastLogConfig, InstrumentError, NgmPowerSupply,
)

POLL_INTERVAL = 0.004          # s between status polls while streaming and nothing is ready
MONITOR_PERIOD = 0.5           # s between readback updates while idle
LIVE_STATUS_PERIOD = 1.0       # s between output/mode updates while streaming
PROGRESS_PERIOD = 0.25         # s between progress signals (plot refreshes)
FIRST_CHUNK_GRACE = 1.5        # s after the trigger before a missing first chunk counts as a stall
STALL_TIMEOUT = 1.5            # s without a chunk before checking whether FastLog still runs
STALL_GIVE_UP = 5.0            # s without a chunk before giving up although FastLog reports running
RESTART_WAIT = 120.0           # s to wait for the instrument to come back after SYST:RESTart


class FastLogStopped(InstrumentError):
    """FastLog stopped on the instrument, typically after 'Internal critical error -23'."""


@dataclasses.dataclass
class _Stream:
    """State of one FastLog stream (one trigger), which may fill several Auto-mode captures."""
    channels: tuple[int, ...]
    chunk_samples: int
    t_trigger: float
    started: datetime
    trackers: dict[int, GapTracker]
    last_data: dict[int, float | None]
    last_means: dict[int, tuple[float, float]]
    pending: dict[int, list]                  # per channel: overflow for the next capture
    samples_streamed: int = 0
    last_stall_check: float = 0.0
    outputs_switched: bool = False
    size_warned: bool = False


class InstrumentWorker(QObject):
    scanned = Signal(list)
    connected = Signal(object)            # DeviceInfo
    disconnected = Signal()
    status = Signal(object)               # dict[int, ChannelStatus]
    armed = Signal()                      # waiting for a manual trigger
    capture_started = Signal(object)      # Capture
    progress = Signal(object)             # Capture
    acquired = Signal(object)             # Capture (complete, or partial when stopped)
    run_finished = Signal()
    message = Signal(str)                 # for the status bar
    failed = Signal(str, str, str)        # title, message, kind ("" or "fastlog_stopped")

    def __init__(self):
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._trigger = threading.Event()
        self._psu: NgmPowerSupply | None = None
        self._thread = threading.Thread(target=self._main, name="ngm-instrument", daemon=True)
        self._thread.start()

    # --- Requests from the GUI thread ----------------------------------------------------

    def _post(self, name: str, *args) -> None:
        self._queue.put((name, args))

    def scan(self) -> None:
        self._post("scan")

    def open_session(self, resource: str, visa_select: str) -> None:
        self._post("open", resource, visa_select)

    def close_session(self) -> None:
        self._post("close")

    def set_voltage(self, ch: int, volts: float) -> None:
        self._post("set_voltage", ch, volts)

    def set_current(self, ch: int, amps: float) -> None:
        self._post("set_current", ch, amps)

    def set_output(self, ch: int, on: bool) -> None:
        self._post("set_output", ch, on)

    def set_ranges(self, ch: int, voltage_range: float | None, current_range: float | None) -> None:
        self._post("set_ranges", ch, voltage_range, current_range)

    def run(self, cfg: RunConfig) -> None:
        self._stop.clear()
        self._trigger.clear()
        self._post("run", cfg)

    def request_stop(self) -> None:
        self._stop.set()

    def request_trigger(self) -> None:
        self._trigger.set()

    def restart_instrument(self) -> None:
        self._stop.clear()
        self._post("restart")

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop any run, close the session and end the thread."""
        self._stop.set()
        self._post("quit")
        self._thread.join(timeout)

    # --- Thread --------------------------------------------------------------------------

    def _main(self) -> None:
        next_monitor = 0.0
        while True:
            timeout = None
            if self._psu is not None:
                timeout = max(0.0, next_monitor - time.monotonic())
            try:
                name, args = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._guarded(self._monitor)
                next_monitor = time.monotonic() + MONITOR_PERIOD
                continue
            if name == "quit":
                self._close()
                return
            self._guarded(getattr(self, f"_do_{name}"), *args)

    def _guarded(self, fn, *args) -> None:
        try:
            fn(*args)
        except FastLogStopped as e:
            self.failed.emit("FastLog stopped", str(e), "fastlog_stopped")
        except (InstrumentError, RsInstrException, pyvisa.Error, OSError, ValueError) as e:
            # A timeout or socket error mid-conversation leaves the session unusable.
            lost = isinstance(e, (OSError, pyvisa.VisaIOError, TimeoutException))
            self.failed.emit("Instrument error", str(e) or type(e).__name__, "")
            if lost and self._psu is not None:
                self._close()
        except Exception as e:     # keep the thread alive whatever happens
            self.failed.emit("Unexpected error", f"{type(e).__name__}: {e}", "")

    def _close(self) -> None:
        if self._psu is not None:
            psu, self._psu = self._psu, None
            try:
                psu.close()
            finally:
                self.disconnected.emit()

    def _monitor(self) -> None:
        if self._psu is not None:
            self.status.emit({ch: self._psu.status(ch) for ch in self._psu.info.channels})

    # --- Commands ------------------------------------------------------------------------

    def _do_scan(self) -> None:
        try:
            found = pyvisa.ResourceManager("@py").list_resources("?*::INSTR")
        except Exception as e:
            self.failed.emit("Scan failed", str(e), "")
            found = ()
        hosts = []
        for resource in found:
            parts = resource.split("::")
            if parts[0].upper().startswith("TCPIP") and len(parts) > 1 and parts[1] not in hosts:
                hosts.append(parts[1])
        self.scanned.emit(hosts)

    def _do_open(self, resource: str, visa_select: str) -> None:
        self._close()
        psu = NgmPowerSupply(resource, visa_select)
        try:
            info = psu.open()
        except Exception as e:
            self.failed.emit("Connection failed", f"{resource}\n\n{e}", "")
            self.disconnected.emit()
            return
        self._psu = psu
        self.connected.emit(info)
        self._monitor()

    def _do_close(self) -> None:
        if self._psu is None:
            self.disconnected.emit()
        self._close()

    def _do_set_voltage(self, ch: int, volts: float) -> None:
        self._psu.set_voltage(ch, volts)
        self._emit_status(ch)

    def _do_set_current(self, ch: int, amps: float) -> None:
        self._psu.set_current(ch, amps)
        self._emit_status(ch)

    def _do_set_output(self, ch: int, on: bool) -> None:
        self._psu.set_output(ch, on)
        self._emit_status(ch)

    def _do_set_ranges(self, ch: int, voltage_range: float | None, current_range: float | None) -> None:
        if self._psu.info.has_ranges:
            self._psu.set_ranges(ch, voltage_range, current_range)
        self._emit_status(ch)

    def _emit_status(self, ch: int, stream: _Stream | None = None) -> None:
        """Status of one channel without blocking on READ? while streaming (measured values then come from the data)."""
        status = self._psu.status(ch, measure=stream is None)
        if stream is not None and ch in stream.last_means:
            v, i = stream.last_means[ch]
            status = dataclasses.replace(status, voltage=v, current=i)
        self.status.emit({ch: status})

    def _do_restart(self) -> None:
        psu = self._psu
        if psu is None:
            return
        resource, visa_select = psu.resource, psu.visa_select
        psu.restart()
        self._psu = None
        self.disconnected.emit()
        self.message.emit("Instrument restarting…")
        time.sleep(5)       # it keeps answering briefly before going down
        deadline = time.monotonic() + RESTART_WAIT
        while time.monotonic() < deadline and not self._stop.is_set():
            time.sleep(3)
            candidate = NgmPowerSupply(resource, visa_select, timeout_ms=2000)
            try:
                candidate.open()
                candidate.close()
            except Exception:
                elapsed = RESTART_WAIT - (deadline - time.monotonic())
                self.message.emit(f"Waiting for the instrument to restart… {elapsed:.0f} s")
                continue
            self.message.emit("Instrument restarted; reconnecting")
            self._do_open(resource, visa_select)
            return
        self.failed.emit("Restart", "The instrument didn't come back within "
                         f"{RESTART_WAIT:.0f} s. Check it, then connect again.", "")

    def _run_queued_commands(self, stream: _Stream | None) -> None:
        """Run setpoint/output commands queued while a run is busy; defer anything else."""
        deferred = []
        while True:
            try:
                name, args = self._queue.get_nowait()
            except queue.Empty:
                break
            if name in ("set_voltage", "set_current", "set_output"):
                ch = args[0]
                getattr(self._psu, name)(*args)
                self._emit_status(ch, stream)
            elif name == "quit":
                self._stop.set()
                deferred.append((name, args))
            else:
                deferred.append((name, args))
        for item in deferred:
            self._queue.put(item)

    # --- Acquisition ---------------------------------------------------------------------

    def _do_run(self, cfg: RunConfig) -> None:
        psu = self._psu
        if psu is None:
            self.run_finished.emit()
            return
        channels = tuple(s.channel for s in cfg.channels)
        try:
            interval = psu.prepare_fastlog(FastLogConfig(cfg.channels, cfg.rate))
            if not math.isclose(interval, 1 / cfg.rate.hz, rel_tol=1e-6):
                self.message.emit(f"Instrument sample interval is {interval:g} s")
            number = 0
            stream: _Stream | None = None
            while not self._stop.is_set():
                if stream is None:
                    psu.arm(channels)
                    if cfg.mode is TriggerMode.MANUAL:
                        self.armed.emit()
                        if not self._wait_for_trigger():
                            break
                    stream = self._start_stream(channels, cfg)
                number += 1
                capture = self._new_capture(cfg, interval, stream, number)
                self.capture_started.emit(capture)
                self._stream_into(capture, stream, cfg)
                self.acquired.emit(capture)
                if cfg.mode is TriggerMode.SINGLE or self._stop.is_set():
                    break
                if cfg.mode is TriggerMode.MANUAL:
                    psu.stop_fastlog(channels)
                    stream = None
        finally:
            try:
                psu.stop_fastlog(channels)
                if cfg.output_off_at_end:
                    for ch in channels:
                        psu.set_output(ch, False)
                for ch in channels:
                    self._emit_status(ch)
            except Exception:
                pass
            self.run_finished.emit()

    def _wait_for_trigger(self) -> bool:
        next_monitor = 0.0
        while not self._stop.is_set():
            if self._trigger.wait(0.05):
                self._trigger.clear()
                return True
            self._run_queued_commands(None)
            if time.monotonic() >= next_monitor:
                self._monitor()
                next_monitor = time.monotonic() + MONITOR_PERIOD
        return False

    def _start_stream(self, channels: tuple[int, ...], cfg: RunConfig) -> _Stream:
        started = datetime.now().astimezone()
        t = self._psu.start(channels)
        return _Stream(
            channels=channels, chunk_samples=cfg.rate.chunk_samples, t_trigger=t, started=started,
            trackers={ch: GapTracker(cfg.rate.chunk_samples) for ch in channels},
            last_data=dict.fromkeys(channels), last_means={}, pending={ch: [] for ch in channels})

    def _new_capture(self, cfg: RunConfig, interval: float, stream: _Stream, number: int) -> Capture:
        channels = {}
        for setup in cfg.channels:
            s = self._psu.status(setup.channel, measure=False)
            settings = ChannelSettings(setup.channel, s.voltage_set, s.current_set, s.output,
                                       setup.voltage_range, setup.current_range)
            channels[setup.channel] = ChannelData(settings, cfg.samples)
        first = stream.samples_streamed
        started = stream.started + timedelta(seconds=first * interval)
        capture = Capture(self._psu.info, cfg, interval, started, channels, number, first)
        stream.samples_streamed += cfg.samples
        for ch in stream.channels:
            pending, stream.pending[ch] = stream.pending[ch], []
            for item in pending:
                self._feed(capture, stream, ch, item)
        return capture

    def _feed(self, capture: Capture, stream: _Stream, ch: int, item) -> None:
        """Append a data block or a count of missing samples; overflow is kept for the next capture."""
        data = capture.channels[ch]
        if isinstance(item, int):
            used = data.append_missing(item)
            rest = item - used
            if rest:
                stream.pending[ch].append(rest)
        else:
            used = data.append(item)
            if used < len(item):
                stream.pending[ch].append(item[used:])

    def _stream_into(self, capture: Capture, stream: _Stream, cfg: RunConfig) -> None:
        psu = self._psu
        channels = stream.channels
        last_progress = last_status = time.monotonic()
        while not capture.complete and not self._stop.is_set():
            self._run_queued_commands(stream)
            ready = psu.poll(channels)
            t_poll = time.monotonic()
            for ch in ready:
                block = psu.fetch(ch)
                if len(block) == 0:
                    continue
                if len(block) != stream.chunk_samples and not stream.size_warned:
                    stream.size_warned = True
                    capture.warnings.append(f"CH{ch} sent a chunk of {len(block):,} samples; expected "
                                            f"{stream.chunk_samples:,}. Gap positions may be inexact.")
                missing = stream.trackers[ch].missing_before(t_poll)
                if missing:
                    self._feed(capture, stream, ch, missing)
                self._feed(capture, stream, ch, block)
                stream.last_data[ch] = t_poll
                with np.errstate(invalid="ignore"):
                    stream.last_means[ch] = (float(np.nanmean(block[:, 0])), float(np.nanmean(block[:, 1])))
            if cfg.output_on_at_start and not stream.outputs_switched and all(stream.last_data.values()):
                # FastLog is running (its first chunk took ~0.6 s), so the turn-on is captured after a baseline.
                stream.outputs_switched = True
                for ch in channels:
                    if not capture.channels[ch].settings.output:
                        psu.set_output(ch, True)
                        self._emit_status(ch, stream)
            self._check_stall(stream, t_poll)
            now = time.monotonic()
            if now - last_progress >= PROGRESS_PERIOD:
                last_progress = now
                self.progress.emit(capture)
            if now - last_status >= LIVE_STATUS_PERIOD and stream.last_means:
                last_status = now
                for ch in channels:
                    self._emit_status(ch, stream)
            if not ready:
                time.sleep(POLL_INTERVAL)

    def _check_stall(self, stream: _Stream, now: float) -> None:
        for ch in stream.channels:
            last = stream.last_data[ch]
            silent = now - (last if last is not None else stream.t_trigger + FIRST_CHUNK_GRACE - STALL_TIMEOUT)
            if silent < STALL_TIMEOUT or now - stream.last_stall_check < 0.5:
                continue
            stream.last_stall_check = now
            if not self._psu.fastlog_running(ch):
                raise FastLogStopped(
                    f"FastLog on CH{ch} stopped delivering data and the instrument has switched it off.\n\n"
                    "If the instrument display shows \"Internal critical error -23 · FastLog disabled\", "
                    "FastLog stays off until the instrument restarts. Use Restart Instrument (about "
                    "40 s), then run again.")
            if silent > STALL_GIVE_UP:
                raise InstrumentError(f"No FastLog data from CH{ch} for {silent:.0f} s although FastLog "
                                      "reports running. Check the instrument, then run again.")
