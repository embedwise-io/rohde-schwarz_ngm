"""Captured FastLog data: sample buffers, lost and clipped samples, plotting summaries and statistics.

A Capture holds one acquisition: for each channel, float32 voltage and current arrays (the
instrument's native format), filled chunk by chunk while FastLog streams. Alongside the samples
it records:

- missing samples: chunks the host didn't read before the instrument replaced them. They are
  stored as NaN, and GapTracker works out how many there were from the time between chunks;
- clipped samples: readings pinned at the measurement range's limit (see clipped_mask);
- min/max pyramids, so that a plot of hundreds of millions of samples redraws in milliseconds.

Nothing here depends on Qt; the worker thread appends while the GUI thread reads. Appends only
write beyond the published sample count, so a reader that snapshots `n` first sees consistent data.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import __version__
from .instrument import (
    CHUNK_PERIOD, CURRENT_RANGES, VOLTAGE_RANGES, ChannelSetup, DeviceInfo, SampleRate, range_label,
)

# A reading at or beyond CLIP_FACTOR × range is clipped. Measured clip levels are 1.134-1.141 × range.
CLIP_FACTOR = 1.12
# With autorange the range in use is unknown, so a reading counts as clipped when it sits inside a
# range's clip window and repeats the same value at least AUTO_CLIP_RUN times (a real signal varies).
AUTO_CLIP_WINDOW = (1.12, 1.16)
AUTO_CLIP_RUN = 3
AUTO_CLIP_TOLERANCE = 2e-4     # × range: "same value"

PYRAMID_BASE = 64              # samples per level-0 min/max block
PYRAMID_FACTOR = 8             # blocks per block of the next level
STATS_BLOCK = 1 << 21          # samples per block when computing statistics

QUANTITIES = ("voltage", "current", "power")
UNITS = {"voltage": "V", "current": "A", "power": "W"}

# Bits of the per-sample flags column in CSV exports.
FLAG_VOLTAGE_CLIPPED = 1
FLAG_CURRENT_CLIPPED = 2
FLAG_MISSING = 4


class TriggerMode(enum.Enum):
    SINGLE = "Single"
    AUTO = "Auto (continuous)"
    MANUAL = "Manual"


@dataclass(frozen=True)
class RunConfig:
    channels: tuple[ChannelSetup, ...]
    rate: SampleRate
    duration: float                    # s per capture
    mode: TriggerMode = TriggerMode.SINGLE
    output_on_at_start: bool = False   # switch measured outputs on once the first chunk has arrived
    output_off_at_end: bool = False    # switch measured outputs off when the run ends

    @property
    def samples(self) -> int:
        return max(1, round(self.duration * self.rate.hz))


# --- Interval bookkeeping ----------------------------------------------------------------


class IntervalList:
    """Sorted, non-overlapping [start, stop) sample-index runs, built by appending in order."""

    def __init__(self):
        self._starts: list[int] = []
        self._stops: list[int] = []

    def add(self, start: int, stop: int) -> None:
        if stop <= start:
            return
        if self._stops and start <= self._stops[-1]:
            self._starts[-1] = min(self._starts[-1], start)
            self._stops[-1] = max(self._stops[-1], stop)
        else:
            self._starts.append(start)
            self._stops.append(stop)

    def add_mask(self, mask: np.ndarray, offset: int) -> None:
        """Add the runs of True in mask, whose element 0 is sample `offset`."""
        if not mask.any():
            return
        edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            self.add(offset + int(a), offset + int(b))

    def array(self) -> np.ndarray:
        n = len(self._stops)    # snapshot: the worker thread may be appending
        return np.column_stack((self._starts[:n], self._stops[:n])).astype(np.int64).reshape(-1, 2)

    def count_in(self, start: int, stop: int) -> tuple[int, int]:
        """(samples, runs) of this list inside [start, stop)."""
        runs = self.array()
        if len(runs) == 0:
            return 0, 0
        a = np.clip(runs[:, 0], start, stop)
        b = np.clip(runs[:, 1], start, stop)
        inside = b > a
        return int((b - a)[inside].sum()), int(inside.sum())

    def __len__(self) -> int:
        return len(self._stops)


def merge_intervals(*arrays: np.ndarray) -> np.ndarray:
    """Union of several (n, 2) interval arrays."""
    runs = np.concatenate([a for a in arrays if len(a)] or [np.empty((0, 2), np.int64)])
    if len(runs) == 0:
        return runs
    runs = runs[np.argsort(runs[:, 0], kind="stable")]
    out = [list(runs[0])]
    for a, b in runs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out, dtype=np.int64)


# --- Clipping ----------------------------------------------------------------------------


def clipped_mask(x: np.ndarray, fixed_range: float | None, ranges: tuple[float, ...]) -> np.ndarray:
    """Samples pinned at the measurement limit.

    Fixed range: |x| >= CLIP_FACTOR × range. Autorange: for some range below the largest, |x| lies
    in that range's clip window and the value repeats for AUTO_CLIP_RUN samples. That catches the
    ~20 µs a lower range stays clipped before autorange switches up, but not a real signal passing
    through the same level.
    """
    a = np.abs(x)
    if fixed_range is not None:
        return a >= CLIP_FACTOR * fixed_range
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < AUTO_CLIP_RUN:
        return mask
    for r in ranges[:-1]:
        window = (a >= AUTO_CLIP_WINDOW[0] * r) & (a <= AUTO_CLIP_WINDOW[1] * r)
        if not window.any():
            continue
        # same[j]: samples j and j+1 are both in the window and (nearly) equal.
        same = window[:-1] & window[1:] & (np.abs(np.diff(x)) <= AUTO_CLIP_TOLERANCE * r)
        core = same[:-1] & same[1:]                 # sample j+1 sits in a run of >= 3
        run = np.zeros(len(x), dtype=bool)
        run[1:-1] |= core
        run[:-2] |= core
        run[2:] |= core
        mask |= run
    return mask


# --- Min/max pyramid ---------------------------------------------------------------------


class MinMaxPyramid:
    """Block minima and maxima of a growing series at several resolutions, for fast envelope plots.

    Level k holds one (min, max) pair per PYRAMID_BASE × PYRAMID_FACTOR**k samples. NaN samples are
    ignored unless a whole block is NaN, which then stays NaN so the plot shows a gap.
    """

    def __init__(self, capacity: int):
        self.block_sizes: list[int] = []
        self.mins: list[np.ndarray] = []
        self.maxs: list[np.ndarray] = []
        self.counts: list[int] = []        # blocks written per level (the last may be partial)
        size = PYRAMID_BASE
        while True:
            blocks = max(1, -(-capacity // size))
            self.block_sizes.append(size)
            self.mins.append(np.full(blocks, np.nan, np.float32))
            self.maxs.append(np.full(blocks, np.nan, np.float32))
            self.counts.append(0)
            if blocks <= 4:
                break
            size *= PYRAMID_FACTOR

    def extend(self, start: int, values: np.ndarray) -> None:
        """Update with samples [start, start + len(values)); start must be a multiple of PYRAMID_BASE.

        `values` must run to the end of the valid data, so the last partial block is complete as far
        as it goes; the next call recomputes it.
        """
        b = PYRAMID_BASE
        first = start // b
        n = len(values)
        if n == 0:
            return
        last = first + -(-n // b)
        pad = (last - first) * b - n
        v = values if not pad else np.concatenate((values, np.full(pad, np.nan, values.dtype)))
        v = v.reshape(-1, b)
        with np.errstate(invalid="ignore"):
            self.mins[0][first:last] = np.fmin.reduce(v, axis=1)
            self.maxs[0][first:last] = np.fmax.reduce(v, axis=1)
        self.counts[0] = last
        for level in range(1, len(self.block_sizes)):
            f = PYRAMID_FACTOR
            lo = first // f
            hi = -(-last // f)
            first, last = lo, hi
            child_lo, child_hi = lo * f, min(hi * f, self.counts[level - 1])
            mins = self.mins[level - 1][child_lo:child_hi]
            maxs = self.maxs[level - 1][child_lo:child_hi]
            pad = (hi - lo) * f - len(mins)
            if pad:
                mins = np.concatenate((mins, np.full(pad, np.nan, np.float32)))
                maxs = np.concatenate((maxs, np.full(pad, np.nan, np.float32)))
            self.mins[level][lo:hi] = np.fmin.reduce(mins.reshape(-1, f), axis=1)
            self.maxs[level][lo:hi] = np.fmax.reduce(maxs.reshape(-1, f), axis=1)
            self.counts[level] = hi

    def level_for(self, samples_per_bin: float) -> int | None:
        """Coarsest level whose blocks are at most half a bin; None if raw samples should be used."""
        level = None
        for k, size in enumerate(self.block_sizes):
            if size * 2 <= samples_per_bin:
                level = k
        return level

    def binned(self, level: int, start: int, stop: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin min and max over samples [start, stop) using one level; edges are bin starts (sample indices)."""
        size = self.block_sizes[level]
        b0 = start // size
        b1 = min(-(-stop // size), self.counts[level])
        if b1 <= b0:
            nan = np.full(len(edges), np.nan, np.float32)
            return nan, nan.copy()
        idx = np.clip(edges // size - b0, 0, b1 - b0 - 1)
        idx = np.maximum.accumulate(idx)
        mins = self.mins[level][b0:b1]
        maxs = self.maxs[level][b0:b1]
        with np.errstate(invalid="ignore"):
            return np.fmin.reduceat(mins, idx), np.fmax.reduceat(maxs, idx)


def binned_raw(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin min and max of raw samples; edges are bin starts relative to values[0]."""
    if len(values) == 0:
        nan = np.full(len(edges), np.nan, np.float32)
        return nan, nan.copy()
    idx = np.maximum.accumulate(np.clip(edges, 0, len(values) - 1))
    with np.errstate(invalid="ignore"):
        return np.fmin.reduceat(values, idx), np.fmax.reduceat(values, idx)


# --- Channel data ------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSettings:
    """What a channel was set to when the capture started (for metadata)."""
    channel: int
    voltage_set: float
    current_set: float
    output: bool
    voltage_range: float | None
    current_range: float | None


class ChannelData:
    """One channel's samples in a capture. Appended to by one thread; readable from another."""

    def __init__(self, settings: ChannelSettings, capacity: int):
        self.settings = settings
        self.capacity = capacity
        self.voltage = np.empty(capacity, np.float32)
        self.current = np.empty(capacity, np.float32)
        self.n = 0
        self.missing = IntervalList()
        self.voltage_clipped = IntervalList()
        self.current_clipped = IntervalList()
        self.pyramids = {q: MinMaxPyramid(capacity) for q in QUANTITIES}

    @property
    def channel(self) -> int:
        return self.settings.channel

    @property
    def full(self) -> bool:
        return self.n >= self.capacity

    def series(self, quantity: str, start: int, stop: int) -> np.ndarray:
        """Samples [start, stop) of 'voltage', 'current' or 'power' (computed)."""
        if quantity == "power":
            return self.voltage[start:stop] * self.current[start:stop]
        return getattr(self, quantity)[start:stop]

    def append(self, block: np.ndarray) -> int:
        """Append (V, I) rows; returns how many fit (the rest belongs to the next capture)."""
        k = min(len(block), self.capacity - self.n)
        if k <= 0:
            return 0
        start = self.n
        self.voltage[start:start + k] = block[:k, 0]
        self.current[start:start + k] = block[:k, 1]
        self._detect_clipping(start, start + k)
        self._update(start, start + k)
        return k

    def append_missing(self, count: int) -> int:
        """Append lost samples as NaN; returns how many fit."""
        k = min(count, self.capacity - self.n)
        if k <= 0:
            return 0
        start = self.n
        self.voltage[start:start + k] = np.nan
        self.current[start:start + k] = np.nan
        self.missing.add(start, start + k)
        self._update(start, start + k)
        return k

    def _detect_clipping(self, start: int, stop: int) -> None:
        # Two samples of context so autorange runs that straddle chunks are found.
        ctx = min(start, AUTO_CLIP_RUN - 1)
        s = self.settings
        for values, rng, ranges, out in (
                (self.voltage, s.voltage_range, VOLTAGE_RANGES, self.voltage_clipped),
                (self.current, s.current_range, CURRENT_RANGES, self.current_clipped)):
            mask = clipped_mask(values[start - ctx:stop], rng, ranges)
            out.add_mask(mask, start - ctx)

    def _update(self, start: int, stop: int) -> None:
        base = (start // PYRAMID_BASE) * PYRAMID_BASE
        for q in QUANTITIES:
            self.pyramids[q].extend(base, self.series(q, base, stop))
        self.n = stop      # publish last, after the data and summaries are written

    def flags(self, start: int, stop: int) -> np.ndarray:
        """Per-sample FLAG_* bits for [start, stop)."""
        flags = np.zeros(stop - start, np.uint8)
        for runs, bit in ((self.voltage_clipped, FLAG_VOLTAGE_CLIPPED), (self.current_clipped, FLAG_CURRENT_CLIPPED),
                          (self.missing, FLAG_MISSING)):
            for a, b in runs.array():
                a, b = max(a, start), min(b, stop)
                if b > a:
                    flags[a - start:b - start] |= bit
        return flags


class GapTracker:
    """Infers lost chunks from arrival times: chunks arrive every CHUNK_PERIOD, and a chunk replaced
    before it was read leaves a longer interval. Arrival jitter must stay below half a period."""

    def __init__(self, chunk_samples: int):
        self.chunk_samples = chunk_samples
        self.last: float | None = None

    def missing_before(self, arrival: float) -> int:
        """Samples lost between the previous chunk and one that arrived at `arrival` (s, monotonic)."""
        lost = 0
        if self.last is not None:
            periods = round((arrival - self.last) / CHUNK_PERIOD)
            lost = max(0, periods - 1) * self.chunk_samples
        self.last = arrival
        return lost


# --- Capture -----------------------------------------------------------------------------


@dataclass
class Capture:
    device: DeviceInfo
    config: RunConfig
    interval: float                    # s, read back from the instrument
    started: datetime                  # host time of the trigger (approximate start of sampling)
    channels: dict[int, ChannelData]
    number: int = 1                    # capture number within the run
    first_sample: int = 0              # index of sample 0 in the FastLog stream (Auto mode frames)
    warnings: list[str] = field(default_factory=list)

    @property
    def target(self) -> int:
        return next(iter(self.channels.values())).capacity

    @property
    def n(self) -> int:
        """Samples available on every channel."""
        return min(ch.n for ch in self.channels.values())

    @property
    def complete(self) -> bool:
        return all(ch.full for ch in self.channels.values())

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.interval

    def duration(self, n: int | None = None) -> float:
        return (self.n if n is None else n) * self.interval

    def metadata(self) -> dict[str, str]:
        d = self.device
        meta = {
            "software": f"ngm-daq {__version__}",
            "instrument": f"{d.manufacturer} {d.model}",
            "serial": d.serial,
            "firmware": d.firmware,
            "options": ",".join(sorted(d.options)),
            "resource": d.resource,
            "started": self.started.isoformat(timespec="milliseconds"),
            "trigger": self.config.mode.value,
            "capture_number": str(self.number),
            "sample_interval_s": f"{self.interval:.9g}",
            "sample_rate_hz": f"{self.sample_rate:.9g}",
            "channels": ",".join(f"CH{c}" for c in self.channels),
        }
        for c, ch in self.channels.items():
            s = ch.settings
            meta[f"ch{c}_voltage_setpoint_V"] = f"{s.voltage_set:g}"
            meta[f"ch{c}_current_limit_A"] = f"{s.current_set:g}"
            meta[f"ch{c}_output_at_start"] = "on" if s.output else "off"
            meta[f"ch{c}_voltage_range"] = range_label(s.voltage_range, "V")
            meta[f"ch{c}_current_range"] = range_label(s.current_range, "A")
        return meta


# --- Statistics --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stats:
    samples: int                     # in the window
    valid: int                       # not missing
    mean: dict[str, float]
    std: dict[str, float]
    min: dict[str, float]
    max: dict[str, float]
    energy_j: float                  # sum of V × I × dt over valid samples
    charge_c: float                  # sum of I × dt over valid samples
    missing: tuple[int, int]         # (samples, runs)
    voltage_clipped: tuple[int, int]
    current_clipped: tuple[int, int]


def window_stats(ch: ChannelData, start: int, stop: int, interval: float) -> Stats:
    """Statistics of samples [start, stop), ignoring missing ones. Works in blocks to bound memory."""
    stop = min(stop, ch.n)
    count = 0
    sums = dict.fromkeys(QUANTITIES, 0.0)
    squares = dict.fromkeys(QUANTITIES, 0.0)
    lo = dict.fromkeys(QUANTITIES, math.inf)
    hi = dict.fromkeys(QUANTITIES, -math.inf)
    for a in range(start, stop, STATS_BLOCK):
        b = min(stop, a + STATS_BLOCK)
        v = ch.voltage[a:b].astype(np.float64)
        i = ch.current[a:b].astype(np.float64)
        ok = np.isfinite(v) & np.isfinite(i)
        if not ok.all():
            v, i = v[ok], i[ok]
        if len(v) == 0:
            continue
        count += len(v)
        for q, x in (("voltage", v), ("current", i), ("power", v * i)):
            sums[q] += float(x.sum())
            squares[q] += float(np.dot(x, x))
            lo[q] = min(lo[q], float(x.min()))
            hi[q] = max(hi[q], float(x.max()))
    mean, std = {}, {}
    for q in QUANTITIES:
        if count:
            mean[q] = sums[q] / count
            std[q] = math.sqrt(max(0.0, squares[q] / count - mean[q] ** 2))
        else:
            mean[q] = std[q] = lo[q] = hi[q] = math.nan
    return Stats(
        samples=max(0, stop - start), valid=count, mean=mean, std=std, min=lo, max=hi,
        energy_j=sums["power"] * interval, charge_c=sums["current"] * interval,
        missing=ch.missing.count_in(start, stop),
        voltage_clipped=ch.voltage_clipped.count_in(start, stop),
        current_clipped=ch.current_clipped.count_in(start, stop))


def samples_between(t0: float, t1: float, interval: float, n: int) -> tuple[int, int]:
    """Sample range [start, stop) whose times (index × interval) lie in [t0, t1], within [0, n)."""
    start = min(n, max(0, math.ceil(t0 / interval - 1e-6)))
    stop = min(n, math.floor(t1 / interval + 1e-6) + 1)
    return start, max(start, stop)


@dataclass(frozen=True)
class BoxStats:
    """One quantity's samples inside a selection box: in its time span and between its levels."""
    span: int                        # valid (not missing) samples in the time span
    count: int                       # of those, the ones between the levels
    mean: float
    std: float
    min: float
    max: float


def box_stats(ch: ChannelData, quantity: str, start: int, stop: int, lo: float, hi: float) -> BoxStats:
    """Statistics of samples [start, stop) of a quantity with values in [lo, hi], ignoring missing ones."""
    stop = min(stop, ch.n)
    span = count = 0
    total = squares = 0.0
    low, high = math.inf, -math.inf
    for a in range(start, stop, STATS_BLOCK):
        b = min(stop, a + STATS_BLOCK)
        if quantity == "power":
            x = ch.voltage[a:b].astype(np.float64) * ch.current[a:b]
        else:
            x = getattr(ch, quantity)[a:b].astype(np.float64)
        span += int(np.isfinite(x).sum())
        x = x[(x >= lo) & (x <= hi)]                 # NaN compares false: missing samples drop out
        if len(x) == 0:
            continue
        count += len(x)
        total += float(x.sum())
        squares += float(np.dot(x, x))
        low, high = min(low, float(x.min())), max(high, float(x.max()))
    if count == 0:
        return BoxStats(span, 0, math.nan, math.nan, math.nan, math.nan)
    mean = total / count
    return BoxStats(span, count, mean, math.sqrt(max(0.0, squares / count - mean ** 2)), low, high)
