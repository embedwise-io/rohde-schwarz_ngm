"""Minimal R&S NGM202 SCPI simulator for exercising ngm-daq without hardware.

Implements the commands ngm_daq.instrument sends, over a raw socket:

    python tools/ngm_simulator.py                 # listens on 127.0.0.1:5025
    ngm-daq --resource TCPIP::127.0.0.1::5025::SOCKET

FastLog behaves like the real instrument (firmware 04.013): a chunk of 0.25 s of samples becomes
available every 250 ms, about 0.3 s after its last sample was taken, and only the newest chunk is
kept, so a client that falls behind loses data silently. Starting FastLog with more than 500 kSa/s
in total "crashes" FastLog (internal error -23, reported only on stdout here) until SYST:RESTart.

Signals: with its output on, CH1 behaves like a Bluetooth device (about 2 mA with periodic radio
bursts up to ~25 mA, and an inrush that hits the current limit at turn-on). CH2 has no load.
Readings are clipped at 1.14 × the measurement range, as on the instrument.
"""

from __future__ import annotations

import argparse
import math
import re
import socketserver
import threading
import time

import numpy as np

CHUNK = 0.25
DELIVERY_LATENCY = 0.3          # s from a chunk's last sample until it can be read
START_LATENCY = 0.07            # s from FLOG 1 / *TRG until the first sample
CLIP = 1.14
RATES = {"S500K": 500e3, "S250K": 250e3, "S050K": 50e3, "S50K": 50e3, "S010K": 10e3, "S10K": 10e3,
         "S001K": 1e3, "S1K": 1e3, "S100": 100.0}
RATE_NAMES = {500e3: "S500k", 250e3: "S250k", 50e3: "S050k", 10e3: "S010k", 1e3: "S001k", 100.0: "S100"}
V_RANGES = (5.0, 20.0)
I_RANGES = (0.01, 0.1, 1.0, 10.0)
FASTLOG_BIT = 1 << 12


class Channel:
    def __init__(self, number: int, load: bool):
        self.number = number
        self.load = load
        self.voltage_history = [(-math.inf, 1.0)]    # (time set, value): samples use the value at their time
        self.current = 0.1
        self.output = False
        self.t_on = -math.inf
        self.t_off = -math.inf
        self.v_range, self.v_auto = 20.0, True
        self.i_range, self.i_auto = 10.0, True
        self.flog = False
        self.flog_trig = False
        self.rate = 50e3
        self.target = "SCPI"
        self.t_start = 0.0
        self.delivered = -1         # index of the last chunk read
        self.signalled = -1         # index of the last chunk reported in the event register
        self.oper_event = 0

    @property
    def voltage(self) -> float:
        return self.voltage_history[-1][1]

    @voltage.setter
    def voltage(self, value: float) -> None:
        self.voltage_history.append((time.monotonic(), value))

    # --- Signal model --------------------------------------------------------------------

    def signal(self, t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        n = len(t)
        on = (t >= self.t_on) & ((self.t_off < self.t_on) | (t < self.t_off))
        since = t - self.t_on if math.isfinite(self.t_on) else np.full(n, -1.0)
        times, values = zip(*self.voltage_history)
        vset = np.asarray(values)[np.searchsorted(np.asarray(times), t, side="right") - 1]
        v = np.where(on, vset * (1 - np.exp(-np.clip(since, 0, None) / 1e-4)), -1.8e-4)
        i = rng.normal(0, 1e-5, n)
        if self.load:
            phase = np.mod(since, 0.1)
            burst = (phase < 0.0045) & (np.mod(phase, 0.0015) < 0.0009)
            demand = np.where(burst, 0.022, 0.0019) + 0.0003 * np.sin(2 * np.pi * 50 * t)
            inrush = (since >= 0) & (since < 0.0006)
            demand = np.where(inrush, 10.0, demand)
            i = np.where(on, np.minimum(demand, self.current), i) + rng.normal(0, 3e-5, n)
            v = v - np.where(on & burst, 0.02, 0.0)
        elif self.output:
            i = np.where(on, rng.normal(2e-5, 1e-5, n), i)
        v = v + rng.normal(0, 0.004 if self.load else 3e-4, n) * on
        return self._clip(v, self.v_range, self.v_auto, V_RANGES), self._clip(i, self.i_range, self.i_auto, I_RANGES)

    @staticmethod
    def _clip(x: np.ndarray, rng: float, auto: bool, ranges: tuple[float, ...]) -> np.ndarray:
        if auto:
            rng = next((r for r in ranges if np.abs(x).max() < r), ranges[-1])
        return np.clip(x, -CLIP * rng, CLIP * rng).astype(np.float32)

    # --- FastLog -------------------------------------------------------------------------

    def chunk_samples(self) -> int:
        return round(self.rate * CHUNK)

    def newest_chunk(self, now: float) -> int:
        """Index of the newest chunk that can be read now (-1 if none)."""
        if not self.flog:
            return -1
        return math.floor((now - self.t_start - START_LATENCY - DELIVERY_LATENCY) / CHUNK) - 1

    def start(self, now: float) -> None:
        self.flog = True
        self.t_start = now
        self.delivered = self.signalled = -1


class SimulatedNGM:
    def __init__(self, model: str):
        self.model = model
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.command_log: list[str] = []      # every command that isn't a query, for tests
        self.reset()

    def reset(self) -> None:
        self.channels = {1: Channel(1, load=True), 2: Channel(2, load=False)}
        self.selected = 1
        self.errors: list[str] = []
        self.broken = False
        # Like the instrument, *TRG toggles a FastLog-enable trigger state that FLOG 0 doesn't reset:
        # only every other trigger starts the armed channels.
        self.trigger_state = False

    def log(self, msg: str) -> None:
        print(f"[sim {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def error(self, code: int, text: str) -> None:
        self.errors.append(f'{code},"{text}"')

    # --- Dispatch ------------------------------------------------------------------------

    def handle(self, message: str) -> bytes | None:
        """Execute one message (';'-separated commands); return the joined response, if any."""
        replies: list[bytes] = []
        with self.lock:
            for part in message.split(";"):
                part = part.strip()
                if not part:
                    continue
                if "?" not in part:
                    self.command_log.append(part.lstrip(":"))
                reply = self._dispatch(part.lstrip(":"))
                if reply is not None:
                    replies.append(reply if isinstance(reply, bytes) else reply.encode())
        if not replies:
            return None
        return b";".join(replies) + b"\n"

    def _channel_arg(self, arg: str) -> tuple[str, Channel]:
        m = re.search(r"\(@(\d)\)", arg)
        if m:
            arg = arg[:m.start()].rstrip(", ")
            return arg, self.channels[int(m.group(1))]
        return arg, self.channels[self.selected]

    @staticmethod
    def _bool(arg: str) -> bool:
        return arg.strip().upper() in ("1", "ON")

    def _dispatch(self, cmd: str) -> str | bytes | None:
        head, _, arg = cmd.partition(" ")
        h = head.upper()
        h = re.sub(r"^SOUR(CE)?:", "", h)
        arg, ch = self._channel_arg(arg)
        now = time.monotonic()

        if h == "*IDN?":
            return f"Rohde&Schwarz,{self.model},3638.4472k03/000000,04.013 SIMULATED"
        if h == "*OPT?":
            return "K103,K104,K106"
        if h in ("*CLS", "*RST"):
            self.errors.clear()
            if h == "*RST":
                self.reset()
            return None
        if h == "*OPC?":
            return "1"
        if h == "*STB?":
            return "0"
        if h == "*TRG":
            self.trigger_state = not self.trigger_state
            if self.trigger_state:
                for c in self.channels.values():
                    if c.flog_trig and not c.flog:
                        c.start(now)
                self._check_rate_limit()
            return None
        if h in ("SYST:ERR?", "SYST:ERR:NEXT?"):
            return self.errors.pop(0) if self.errors else '0,"No error"'
        if h in ("SYST:REST", "SYST:RESTART"):
            self.log("restart")
            self.reset()
            return None
        if h in ("INST", "INST:SEL", "INSTRUMENT"):
            m = re.fullmatch(r"(OUT|OUTP|OUTPUT|CH)([12])", arg.strip().upper())
            if not m:
                self.error(-224, "Illegal parameter value")
            else:
                self.selected = int(m.group(2))
            return None
        if h == "INST:NSEL":
            self.selected = int(arg)
            return None
        if h in ("VOLT", "VOLT?", "CURR", "CURR?"):
            name = "voltage" if h.startswith("VOLT") else "current"
            if h.endswith("?"):
                a = arg.strip().upper()
                if a.startswith("MIN"):
                    return "0.000000E+00" if name == "voltage" else "1.000000E-03"
                if a.startswith("MAX"):
                    return "2.005000E+01" if name == "voltage" else "6.010000E+00"
                return f"{getattr(ch, name):.6E}"
            value = float(arg)
            limit = 20.05 if name == "voltage" else (6.01 if ch.voltage <= 6 else 3.01)
            if not 0 <= value <= limit:
                self.error(-222, "Data out of range")
            else:
                setattr(ch, name, value)
            return None
        if h in ("OUTP", "OUTP:STAT", "OUTP?", "OUTP:STAT?"):
            if h.endswith("?"):
                return "1" if ch.output else "0"
            on = self._bool(arg)
            if on and not ch.output:
                ch.t_on = now
            elif ch.output and not on:
                ch.t_off = now
            ch.output = on
            return None
        if h in ("OUTP:TRIG?", "ARB:TRIG?", "LOG:TRIG?"):
            return "0"
        if h == "READ?":
            if not ch.output:
                return "NAN,NAN"
            v, i = ch.signal(np.array([now]), self.rng)
            time.sleep(0.08)
            return f"{v[0]:.6E},{i[0]:.6E}"
        m = re.fullmatch(r"SENS:(VOLT|CURR):RANG(:UPP)?(:AUTO)?(\?)?", h)
        if m:
            attr = "v" if m.group(1) == "VOLT" else "i"
            ranges = V_RANGES if attr == "v" else I_RANGES
            if m.group(3):
                if m.group(4):
                    return "1" if getattr(ch, f"{attr}_auto") else "0"
                setattr(ch, f"{attr}_auto", self._bool(arg))
                return None
            if m.group(4):
                return f"{getattr(ch, f'{attr}_range'):.6E}"
            try:
                value = float(arg)
            except ValueError:
                self.error(-148, "Character data not allowed")
                return None
            setattr(ch, f"{attr}_range", next((r for r in ranges if value <= r * 1.0001), ranges[-1]))
            setattr(ch, f"{attr}_auto", False)
            return None
        m = re.fullmatch(r"STAT:(OPER|QUES):INST:ISUM([12]):(COND\?|EVEN\?|ENAB|PTR|NTR)", h)
        if m:
            c = self.channels[int(m.group(2))]
            if m.group(1) == "QUES" and m.group(3) == "COND?":
                if not c.output:
                    return "2"
                return "1" if c.load and c.current < 0.0025 else "2"
            if m.group(3) == "EVEN?":
                if m.group(1) == "QUES":
                    return "0"
                newest = c.newest_chunk(now) if not self.broken else -1
                value = c.oper_event
                if newest > c.signalled:
                    value |= FASTLOG_BIT
                    c.signalled = newest
                c.oper_event = 0
                return str(value)
            return None
        if h.startswith("FLOG"):
            return self._flog(h, arg, ch, now)
        self.error(-113, f"Undefined header;{cmd}")
        return None

    def _flog(self, h: str, arg: str, ch: Channel, now: float) -> str | bytes | None:
        if h in ("FLOG", "FLOG:STAT"):
            if self._bool(arg):
                ch.start(now)
                self._check_rate_limit()
            else:
                ch.flog = False
            return None
        if h in ("FLOG?", "FLOG:STAT?"):
            return "1" if ch.flog and not self.broken else "0"
        if h == "FLOG:TRIG":
            ch.flog_trig = self._bool(arg)
            return None
        if h == "FLOG:TRIG?":
            return "1" if ch.flog_trig else "0"
        if h == "FLOG:TARG":
            ch.target = arg.strip().upper()
            return None
        if h == "FLOG:TARG?":
            return ch.target
        if h == "FLOG:SRAT":
            key = arg.strip().upper()
            if key not in RATES:
                self.error(-224, "Illegal parameter value")
            else:
                ch.rate = RATES[key]
            return None
        if h == "FLOG:SRAT?":
            return RATE_NAMES[ch.rate]
        if h == "FLOG:STIM?":
            return f"{1 / ch.rate:.6E}"
        if h == "FLOG:DATA?":
            newest = ch.newest_chunk(now) if not self.broken else -1
            if newest <= ch.delivered or ch.target != "SCPI":
                return b"#10"
            ch.delivered = newest
            n = ch.chunk_samples()
            t = ch.t_start + START_LATENCY + (newest * n + np.arange(n)) / ch.rate
            v, i = ch.signal(t, self.rng)
            data = np.column_stack((v, i)).astype("<f4").tobytes()
            size = str(len(data))
            return f"#{len(size)}{size}".encode() + data
        self.error(-113, f"Undefined header;{h}")
        return None

    def _check_rate_limit(self) -> None:
        total = sum(c.rate for c in self.channels.values() if c.flog)
        if total > 500e3 and not self.broken:
            self.broken = True
            for c in self.channels.values():
                c.flog = False
            self.log("FastLog: Internal critical error -23. FastLog disabled. (SYST:RESTart to recover)")


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        sim: SimulatedNGM = self.server.sim
        sim.log(f"client connected: {self.client_address[0]}:{self.client_address[1]}")
        for line in self.rfile:
            message = line.decode(errors="replace").strip()
            if not message:
                continue
            if self.server.verbose:
                sim.log(f"<< {message}")
            reply = sim.handle(message)
            if reply is not None:
                self.wfile.write(reply)
        sim.log("client disconnected")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, sim: SimulatedNGM, verbose: bool):
        super().__init__(address, Handler)
        self.sim = sim
        self.verbose = verbose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5025)
    parser.add_argument("--model", default="NGM202", choices=("NGM201", "NGM202"))
    parser.add_argument("-v", "--verbose", action="store_true", help="log every message")
    args = parser.parse_args()
    sim = SimulatedNGM(args.model)
    with Server((args.host, args.port), sim, args.verbose) as server:
        sim.log(f"{args.model} listening on {args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
