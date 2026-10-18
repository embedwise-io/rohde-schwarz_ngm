"""FastLog driver for Rohde & Schwarz NGM201 / NGM202 power supplies.

FastLog samples the voltage and current of a channel at up to 500 kSa/s. With the SCPI target,
the instrument collects 0.25 s of samples and makes them available as one chunk; every 250 ms
the chunk is replaced by the next one. A chunk that isn't read before it is replaced is lost
without any error or status flag, so the host must keep polling, and it detects losses from the
time between chunks (see capture.GapTracker).

Measured behaviour this driver relies on (NGM202, firmware 04.013):

- Chunks are little-endian float32 pairs (V, I), exactly rate × 0.25 s samples each.
- Both channels can log at once only if their rates total 500 kSa/s or less. Exceeding that
  stops FastLog with "Internal critical error -23" on the front panel, not in the SCPI error
  queue, and FastLog stays dead until the instrument restarts. FastLogConfig refuses such
  configurations.
- FLOG:TRIG 1 arms a channel, and a *TRG starts all armed channels together, so the channels'
  samples are time-aligned. But *TRG toggles an internal FastLog-enable state that FLOG 0 doesn't
  reset: after a triggered capture, the next *TRG does nothing and the one after starts FastLog.
  start() therefore triggers, checks FLOG?, and triggers once more if needed. A single channel is
  started with FLOG 1, which always works.
- Right after FLOG 0, FLOG:TARG and FLOG:SRAT can fail with -221 "Settings conflict"; they are
  retried briefly, and only sent when they differ from the current setting.
- Polling STAT:OPER:INST:ISUM<n>:EVEN? (bit 12) is cheaper for the instrument than polling
  FLOG:DATA? directly: a 1 MB chunk then transfers in ~95 ms instead of ~130 ms.
- Out-of-range readings are clipped at about 1.13-1.14 × the measurement range, not NaN.
- Channel select and query go in one message ("INST OUT1;:FLOG:DATA?"). Two separate writes
  wait ~40 ms on TCP delayed ACK.

The driver has no GUI dependencies:

    from ngm_daq.instrument import ChannelSetup, FastLogConfig, NgmPowerSupply, SampleRate

    with NgmPowerSupply("TCPIP::10.101.0.120::5025::SOCKET") as psu:
        cfg = FastLogConfig((ChannelSetup(1, current_range=0.1),), SampleRate.R500K)
        interval = psu.prepare_fastlog(cfg)
        psu.arm(cfg.channel_numbers)
        psu.start(cfg.channel_numbers)
        ...  # psu.poll() / psu.fetch(1) at least every 250 ms
        psu.stop_fastlog(cfg.channel_numbers)
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass

import numpy as np
from RsInstrument import RsInstrument

SUPPORTED_MODELS = ("NGM201", "NGM202")
CHUNK_PERIOD = 0.25            # s of samples per FastLog chunk, and the chunk cadence
MAX_TOTAL_RATE = 500e3         # Sa/s summed over both channels
FASTLOG_DATA_BIT = 1 << 12     # STAT:OPER:INST:ISUM<n> "FastLog data available"

VOLTAGE_MAX = 20.05            # V
CURRENT_MIN = 0.001            # A
CURRENT_MAX = 6.01             # A, at up to 6 V
CURRENT_MAX_ABOVE_6V = 3.01    # A
VOLTAGE_RANGES = (5.0, 20.0)                # V; SENS:VOLT:RANG
CURRENT_RANGES = (0.01, 0.1, 1.0, 10.0)     # A; SENS:CURR:RANG

# STAT:QUES:INST:ISUM<n>:COND bits.
QUES_CC = 1 << 0               # "Voltage" bit: set in constant-current mode
QUES_CV = 1 << 1               # "Current" bit: set in constant-voltage mode
TRIP_BITS = {1 << 3: "OPP", 1 << 4: "OTP", 1 << 9: "OVP", 1 << 10: "OCP"}

PROTOCOLS = ("Socket", "HiSLIP", "VXI-11")
DEFAULT_TIMEOUT_MS = 5000
START_CONFIRM_TIMEOUT = 0.4    # s to wait for FLOG? = 1 after a trigger (it takes ~45 ms)
SETTING_RETRY = (0.1, 0.2, 0.4, 0.8)   # s between retries of FastLog settings after -221


class InstrumentError(RuntimeError):
    """The instrument rejected a command, or returned or did something unexpected."""


class SampleRate(enum.Enum):
    R500K = ("S500k", 500e3)
    R250K = ("S250k", 250e3)
    R50K = ("S050k", 50e3)
    R10K = ("S010k", 10e3)
    R1K = ("S001k", 1e3)
    R100 = ("S100", 100.0)

    def __init__(self, scpi: str, hz: float):
        self.scpi = scpi
        self.hz = hz

    @property
    def chunk_samples(self) -> int:
        return round(self.hz * CHUNK_PERIOD)

    @property
    def label(self) -> str:
        return format_rate(self.hz)

    @classmethod
    def from_scpi(cls, text: str) -> SampleRate:
        key = text.strip().upper()
        for rate in cls:
            if rate.scpi.upper() == key:
                return rate
        raise InstrumentError(f"Unknown FastLog sample rate '{text}'")


def format_rate(hz: float) -> str:
    return f"{hz / 1e3:g} kSa/s" if hz >= 1e3 else f"{hz:g} Sa/s"


def range_label(value: float | None, unit: str) -> str:
    """'Auto' for None, else e.g. '10 mA' or '5 V'."""
    if value is None:
        return "Auto"
    if value < 1:
        return f"{value * 1e3:g} m{unit}"
    return f"{value:g} {unit}"


def max_current(voltage: float) -> float:
    """Highest current limit the instrument accepts at this voltage setpoint."""
    return CURRENT_MAX if voltage <= 6.0 else CURRENT_MAX_ABOVE_6V


def connection_for(target: str, protocol: str) -> tuple[str, str]:
    """(VISA resource string, RsInstrument SelectVisa value) for an address and protocol.

    Raw sockets use RsInstrument's built-in socket I/O, so no VISA installation is needed.
    HiSLIP and VXI-11 use pyvisa-py. A full resource string overrides the protocol.
    """
    target = target.strip()
    if "::" in target:
        return target, "socket" if target.upper().endswith("::SOCKET") else "@py"
    return {
        "Socket": (f"TCPIP::{target}::5025::SOCKET", "socket"),
        "HiSLIP": (f"TCPIP::{target}::hislip0::INSTR", "@py"),
        "VXI-11": (f"TCPIP::{target}::inst0::INSTR", "@py"),
    }[protocol]


@dataclass(frozen=True)
class DeviceInfo:
    resource: str
    manufacturer: str
    model: str
    part_number: str
    serial: str
    firmware: str
    options: frozenset[str]

    @classmethod
    def parse(cls, resource: str, idn: str, opt: str) -> DeviceInfo:
        # e.g. "Rohde&Schwarz,NGM202,3638.4472k03/103062,04.013 0092C027A93"
        parts = [p.strip() for p in idn.split(",")] + [""] * 4
        part_number, _, serial = parts[2].partition("/")
        options = frozenset(o.strip() for o in opt.split(",") if o.strip() not in ("", "0"))
        return cls(resource, parts[0], parts[1], part_number, serial or parts[2], parts[3], options)

    @property
    def channels(self) -> tuple[int, ...]:
        return (1, 2) if self.model.endswith("2") else (1,)

    @property
    def has_ranges(self) -> bool:
        """Measurement range selection needs the K104 (DVM) option."""
        return "K104" in self.options

    @property
    def firmware_version(self) -> str:
        return self.firmware.split()[0] if self.firmware else ""

    def summary(self) -> str:
        opts = ", ".join(sorted(self.options)) or "none"
        return f"{self.model} · S/N {self.serial}\nFirmware {self.firmware_version}\nOptions {opts}"


@dataclass(frozen=True)
class ChannelSetup:
    """Measurement settings for one channel in a FastLog capture. A range of None means autorange."""
    channel: int
    voltage_range: float | None = None
    current_range: float | None = None


@dataclass(frozen=True)
class FastLogConfig:
    channels: tuple[ChannelSetup, ...]
    rate: SampleRate

    @property
    def channel_numbers(self) -> tuple[int, ...]:
        return tuple(c.channel for c in self.channels)

    def problems(self) -> list[str]:
        """Reasons the instrument can't run this configuration (empty if it can)."""
        problems = []
        if not self.channels:
            problems.append("Select at least one channel to measure.")
        if len(set(self.channel_numbers)) != len(self.channels):
            problems.append("A channel is listed twice.")
        if self.rate.hz * len(self.channels) > MAX_TOTAL_RATE:
            problems.append(f"Both channels together can log at most {format_rate(MAX_TOTAL_RATE)}; "
                            f"use {format_rate(MAX_TOTAL_RATE / 2)} or lower per channel.")
        return problems


@dataclass(frozen=True)
class ChannelStatus:
    channel: int
    output: bool
    voltage_set: float
    current_set: float
    voltage: float                   # measured; NaN while unknown or with the output off
    current: float
    condition: int                   # STAT:QUES:INST:ISUM<n>:COND
    voltage_range: float | None      # None = autorange
    current_range: float | None

    @property
    def mode(self) -> str:
        if not self.output:
            return ""
        if self.condition & QUES_CC:
            return "CC"
        return "CV" if self.condition & QUES_CV else ""

    @property
    def trips(self) -> tuple[str, ...]:
        return tuple(name for bit, name in TRIP_BITS.items() if self.condition & bit)


class NgmPowerSupply:
    """One session to an NGM201/NGM202. Not thread-safe: use it from one thread."""

    def __init__(self, resource: str, visa_select: str = "socket", timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.resource = resource
        self.visa_select = visa_select
        self.timeout_ms = timeout_ms
        self._inst: RsInstrument | None = None
        self.info: DeviceInfo | None = None

    def __enter__(self) -> NgmPowerSupply:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- Session -------------------------------------------------------------------------

    def open(self) -> DeviceInfo:
        try:
            inst = RsInstrument(self.resource, id_query=False, reset=False,
                                options=f"SelectVisa='{self.visa_select}'")
        except Exception as e:
            raise InstrumentError(f"Could not open {self.resource}: {e}") from e
        # Errors are checked explicitly where they matter, rather than after every command.
        inst.instrument_status_checking = False
        inst.visa_timeout = self.timeout_ms
        inst.opc_timeout = self.timeout_ms
        self._inst = inst
        try:
            idn = self.query("*IDN?")
            model = idn.split(",")[1].strip() if idn.count(",") >= 1 else idn
            if model not in SUPPORTED_MODELS:
                raise InstrumentError(f"Unsupported instrument '{idn}'. Expected an NGM201 or NGM202 "
                                      "(FastLog is not available on the NGL series).")
            self.info = DeviceInfo.parse(self.resource, idn, self.query("*OPT?"))
            self.write("*CLS")
        except Exception:
            self.close()
            raise
        return self.info

    def close(self) -> None:
        if self._inst is not None:
            try:
                self._inst.close()
            except Exception:
                pass
            finally:
                self._inst = None
                self.info = None

    @property
    def is_open(self) -> bool:
        return self._inst is not None

    def write(self, cmd: str) -> None:
        self._inst.write_str(cmd)

    def query(self, cmd: str) -> str:
        return self._inst.query_str(cmd).strip()

    def query_values(self, cmd: str) -> list[str]:
        """Responses of a ';'-joined compound query, e.g. 'VOLT?;:CURR?' -> ['4.2', '0.1']."""
        return [part.strip() for part in self.query(cmd).split(";")]

    def opc(self) -> None:
        self.query("*OPC?")

    def errors(self) -> list[str]:
        """Drain the instrument error queue."""
        errs = []
        for _ in range(32):
            err = self.query("SYST:ERR?")
            if err.startswith(("0,", "+0,")):
                break
            errs.append(err)
        return errs

    def check_errors(self, context: str) -> None:
        if errs := self.errors():
            raise InstrumentError(f"{context}:\n" + "\n".join(errs))

    def restart(self) -> None:
        """Reboot the instrument (~35 s), e.g. to recover FastLog after an internal error. Closes the session."""
        try:
            self.write("SYST:RESTart")
        finally:
            self.close()

    # --- Channel control -----------------------------------------------------------------

    def status(self, ch: int, measure: bool = True) -> ChannelStatus:
        """Setpoints, output state, condition and ranges; measured V and I if measure is True.

        The measurement (READ?) takes ~80 ms, so it is skipped while FastLog runs.
        """
        values = self.query_values(
            f"INST OUT{ch};:VOLT?;:CURR?;:OUTP?;:STAT:QUES:INST:ISUM{ch}:COND?;"
            ":SENS:VOLT:RANG?;:SENS:VOLT:RANG:AUTO?;:SENS:CURR:RANG?;:SENS:CURR:RANG:AUTO?")
        v_set, i_set, out, cond, v_rng, v_auto, i_rng, i_auto = values[:8]
        voltage = current = math.nan
        if measure:
            v, _, i = self.query(f"READ? (@{ch})").partition(",")
            voltage, current = _float(v), _float(i)
        return ChannelStatus(
            channel=ch, output=out == "1", voltage_set=float(v_set), current_set=float(i_set),
            voltage=voltage, current=current, condition=int(float(cond)),
            voltage_range=None if v_auto == "1" else float(v_rng),
            current_range=None if i_auto == "1" else float(i_rng))

    def set_voltage(self, ch: int, volts: float) -> float:
        """Set the voltage setpoint; returns the value the instrument accepted."""
        self.write(f"INST OUT{ch};:VOLT {volts:.4f}")
        self.check_errors(f"CH{ch} rejected voltage {volts:g} V")
        return float(self.query(f"INST OUT{ch};:VOLT?"))

    def set_current(self, ch: int, amps: float) -> float:
        """Set the current limit; returns the value the instrument accepted."""
        self.write(f"INST OUT{ch};:CURR {amps:.5f}")
        self.check_errors(f"CH{ch} rejected current limit {amps:g} A")
        return float(self.query(f"INST OUT{ch};:CURR?"))

    def set_output(self, ch: int, on: bool) -> bool:
        """Switch one channel's output; returns the resulting state."""
        self.write(f"INST OUT{ch};:OUTP {1 if on else 0}")
        self.check_errors(f"CH{ch} output {'on' if on else 'off'} failed")
        return self.query(f"INST OUT{ch};:OUTP?") == "1"

    def set_ranges(self, ch: int, voltage_range: float | None, current_range: float | None) -> None:
        """Set the measurement ranges (None = autorange). Each change takes ~0.2 s to settle."""
        cmds = [f"INST OUT{ch}"]
        for name, value in (("VOLT", voltage_range), ("CURR", current_range)):
            if value is None:
                cmds.append(f"SENS:{name}:RANG:AUTO 1")
            else:
                cmds += [f"SENS:{name}:RANG:AUTO 0", f"SENS:{name}:RANG {value:g}"]
        self.write(";:".join(cmds))
        self.opc()
        self.check_errors(f"CH{ch} rejected the measurement ranges")

    # --- FastLog -------------------------------------------------------------------------

    def prepare_fastlog(self, cfg: FastLogConfig) -> float:
        """Stop any FastLog, apply ranges and rate, and set up status reporting. Returns the sample interval (s)."""
        if problems := cfg.problems():
            raise InstrumentError("\n".join(problems))
        channels = self.info.channels
        unknown = [c for c in cfg.channel_numbers if c not in channels]
        if unknown:
            raise InstrumentError(f"{self.info.model} has no channel {unknown[0]}.")
        self.stop_fastlog(channels)
        self._check_trigger_targets()
        for setup in cfg.channels:
            ch = setup.channel
            if self.info.has_ranges:
                self.set_ranges(ch, setup.voltage_range, setup.current_range)
            self._set_fastlog_target(ch, cfg.rate)
            self.write(f"STAT:OPER:INST:ISUM{ch}:ENAB {FASTLOG_DATA_BIT};"
                       f":STAT:OPER:INST:ISUM{ch}:PTR {FASTLOG_DATA_BIT};:STAT:OPER:INST:ISUM{ch}:NTR 0")
        self.opc()
        self.check_errors("Instrument rejected the FastLog configuration")
        intervals = set()
        for ch in cfg.channel_numbers:
            rate = SampleRate.from_scpi(self.query(f"INST OUT{ch};:FLOG:SRAT?"))
            if rate is not cfg.rate:
                raise InstrumentError(f"CH{ch} FastLog rate is {rate.label}, expected {cfg.rate.label}.")
            intervals.add(float(self.query("FLOG:STIM?")))
        self.poll(cfg.channel_numbers)      # clear stale "data available" events
        if len(intervals) != 1:
            raise InstrumentError(f"Channels report different sample intervals: {sorted(intervals)}")
        return intervals.pop()

    def _set_fastlog_target(self, ch: int, rate: SampleRate) -> None:
        """FastLog to SCPI at `rate`, retrying the -221 'Settings conflict' that follows a recent FLOG 0."""
        for wait in (0.0, *SETTING_RETRY):
            time.sleep(wait)
            target, current = self.query_values(f"INST OUT{ch};:FLOG:TARG?;:FLOG:SRAT?")
            cmds = []
            if target.upper() != "SCPI":
                cmds.append("FLOG:TARG SCPI")
            if current.upper() != rate.scpi.upper():
                cmds.append(f"FLOG:SRAT {rate.scpi}")
            if not cmds:
                return
            self.write(f"INST OUT{ch};:" + ";:".join(cmds))
            self.opc()
            errs = self.errors()
            if any(not e.startswith("-221") for e in errs):
                raise InstrumentError(f"CH{ch} rejected the FastLog settings:\n" + "\n".join(errs))
        raise InstrumentError(f"CH{ch} kept refusing the FastLog settings (settings conflict). Try again.")

    def _check_trigger_targets(self) -> None:
        """*TRG starts FastLog, but it also fires anything else set to 'Triggered'. Refuse rather than surprise."""
        armed = []
        for ch in self.info.channels:
            out_trig, arb_trig = self.query_values(f"INST OUT{ch};:OUTP:TRIG?;:ARB:TRIG?")
            if out_trig == "1":
                armed.append(f"CH{ch} output")
            if arb_trig == "1":
                armed.append(f"CH{ch} arbitrary waveform")
        if self.query("LOG:TRIG?") == "1":
            armed.append("data logging")
        if armed:
            raise InstrumentError(
                "These instrument functions are set to 'Triggered' and would also react to the trigger "
                f"that starts FastLog: {', '.join(armed)}.\nTurn 'Triggered' off for them on the instrument "
                "(channel menu > Output / Arbitrary, device menu > Logging) and run again.")

    def arm(self, channels: tuple[int, ...]) -> None:
        """Prepare FastLog on the channels for start(). Several channels are armed for a common trigger."""
        triggered = 1 if len(channels) > 1 else 0
        for ch in channels:
            self.write(f"INST OUT{ch};:FLOG 0;:FLOG:TRIG {triggered}")
        self.opc()
        self.check_errors("Arming FastLog failed")

    def start(self, channels: tuple[int, ...]) -> float:
        """Start FastLog on the armed channels, together. Returns time.monotonic() of the start command."""
        if len(channels) == 1:
            t = time.monotonic()
            self.write(f"INST OUT{channels[0]};:FLOG 1")
            return t
        for _ in range(2):
            # A *TRG either starts the armed channels or only flips the instrument's internal trigger
            # state (see the module docstring); in that case the next one starts them.
            t = time.monotonic()
            self.write("*TRG")
            running = self._wait_running(channels)
            if all(running):
                return t
            if any(running):
                self.stop_fastlog(channels)
                raise InstrumentError("The trigger started FastLog on only some channels. Run again.")
        raise InstrumentError("FastLog didn't start on the trigger. Check the instrument, then run again.")

    def _wait_running(self, channels: tuple[int, ...]) -> list[bool]:
        query = ";:".join(f"INST OUT{ch};:FLOG?" for ch in channels)
        deadline = time.monotonic() + START_CONFIRM_TIMEOUT
        while True:
            running = [v == "1" for v in self.query_values(query)]
            if all(running) or time.monotonic() > deadline:
                return running
            time.sleep(0.01)

    def poll(self, channels: tuple[int, ...]) -> list[int]:
        """Channels that have a new chunk ready. Reading the event registers clears them."""
        values = self.query_values(";:".join(f"STAT:OPER:INST:ISUM{ch}:EVEN?" for ch in channels))
        return [ch for ch, v in zip(channels, values) if int(float(v)) & FASTLOG_DATA_BIT]

    def fetch(self, ch: int) -> np.ndarray:
        """The pending chunk of one channel as an (n, 2) float32 array of (V, I) pairs; empty if none."""
        raw = self._inst.query_bin_block(f"INST OUT{ch};:FLOG:DATA?")
        if len(raw) % 8:
            raise InstrumentError(f"CH{ch} FastLog block of {len(raw)} bytes is not a whole number of samples.")
        return np.frombuffer(raw, dtype="<f4").reshape(-1, 2)

    def fastlog_running(self, ch: int) -> bool:
        return self.query(f"INST OUT{ch};:FLOG?") == "1"

    def stop_fastlog(self, channels: tuple[int, ...]) -> None:
        """Stop and disarm FastLog on the channels."""
        for ch in channels:
            self.write(f"INST OUT{ch};:FLOG 0;:FLOG:TRIG 0")
        self.opc()


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan
