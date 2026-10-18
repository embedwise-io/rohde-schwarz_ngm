"""Save captures (or a slice of one) to CSV or HDF5."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

from .capture import FLAG_CURRENT_CLIPPED, FLAG_MISSING, FLAG_VOLTAGE_CLIPPED, Capture, IntervalList

# Export formats and their file suffixes; the first suffix is the default.
FORMATS = {"CSV": (".csv",), "HDF5": (".h5", ".hdf5")}

# Modern HDF5 file format (checksummed metadata, newer chunk index) that HDF5 1.14 tools can still
# read. A low bound of "latest" (HDF5 2.0) would make files unreadable by tools built on HDF5 1.x.
HDF5_LIBVER = ("v114", "latest")
HDF5_COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
BLOCK = 500_000                  # rows per write, and per progress report

FLAGS_DESCRIPTION = (f"{FLAG_VOLTAGE_CLIPPED} = voltage clipped (beyond the measurement range), "
                     f"{FLAG_CURRENT_CLIPPED} = current clipped, "
                     f"{FLAG_MISSING} = missing (not received from the instrument; V and I are NaN)")

Progress = Callable[[int, int], bool]   # (rows done, total) -> False to cancel


class Cancelled(Exception):
    """The progress callback asked to stop; the partial file has been removed."""


def format_for(path: str | Path) -> str | None:
    """Export format implied by the file suffix, or None if it isn't one of FORMATS."""
    suffix = Path(path).suffix.lower()
    return next((name for name, suffixes in FORMATS.items() if suffix in suffixes), None)


def save(path: str | Path, capture: Capture, start: int = 0, stop: int | None = None,
         progress: Progress | None = None) -> int:
    """Write samples [start:stop) of every channel, with metadata, in the format given by the suffix.

    Returns the number of samples written per channel.
    """
    path = Path(path)
    fmt = format_for(path)
    if fmt is None:
        suffixes = ", ".join(s for group in FORMATS.values() for s in group)
        raise ValueError(f"Unknown export format for '{path.name}'; use one of {suffixes}.")
    stop = capture.n if stop is None else min(stop, capture.n)
    start = max(0, min(start, stop))
    meta = capture.metadata()
    meta["samples"] = str(stop - start)
    meta["first_sample_index"] = str(start)
    try:
        if fmt == "HDF5":
            _save_hdf5(path, capture, start, stop, meta, progress)
        else:
            _save_csv(path, capture, start, stop, meta, progress)
    except Cancelled:
        path.unlink(missing_ok=True)
        raise
    return stop - start


def _report(progress: Progress | None, done: int, total: int) -> None:
    if progress is not None and progress(done, total) is False:
        raise Cancelled()


def _save_csv(path: Path, capture: Capture, start: int, stop: int, meta: dict[str, str],
              progress: Progress | None) -> None:
    """Metadata as '#' comments, then a plain header, so pandas.read_csv(path, comment="#") works."""
    channels = list(capture.channels.values())
    columns = ["time_s"]
    for ch in channels:
        c = ch.channel
        columns += [f"ch{c}_voltage_V", f"ch{c}_current_A", f"ch{c}_flags"]
    row_format = "%.6f" + ",%.7g,%.7g,%d" * len(channels) + "\n"
    total = stop - start
    with open(path, "w", newline="") as f:
        f.writelines(f"# {k}: {v}\n" for k, v in meta.items())
        f.write(f"# flags: {FLAGS_DESCRIPTION}\n")
        f.write(",".join(columns) + "\n")
        for a in range(start, stop, BLOCK):
            b = min(stop, a + BLOCK)
            cols = [np.arange(a, b) * capture.interval]
            for ch in channels:
                cols += [ch.voltage[a:b], ch.current[a:b], ch.flags(a, b)]
            rows = zip(*(c.tolist() for c in cols))
            f.write("".join(row_format % row for row in rows))
            _report(progress, b - start, total)
        if total == 0:
            _report(progress, 0, 0)


def _save_hdf5(path: Path, capture: Capture, start: int, stop: int, meta: dict[str, str],
               progress: Progress | None) -> None:
    """One group per channel with float32 'voltage' and 'current' datasets (the instrument's native
    format, so lossless), and [start, stop) sample-index runs of missing and clipped samples.

    Samples are evenly spaced, so time is not stored: t = t0_s + arange(n) * sample_interval_s.
    """
    total = stop - start
    n = total
    with h5py.File(path, "w", libver=HDF5_LIBVER) as f:
        f.attrs.update(meta)
        f.attrs.update({
            "sample_interval_s": capture.interval,
            "sample_rate_hz": float(f"{capture.sample_rate:.9g}"),
            "t0_s": start * capture.interval,
            "samples": n,
            "first_sample_index": start,
        })
        datasets = []
        for ch in capture.channels.values():
            g = f.create_group(f"ch{ch.channel}")
            s = ch.settings
            g.attrs.update({
                "voltage_setpoint_V": s.voltage_set, "current_limit_A": s.current_set,
                "output_at_start": "on" if s.output else "off",
                "voltage_range": meta[f"ch{ch.channel}_voltage_range"],
                "current_range": meta[f"ch{ch.channel}_current_range"],
            })
            for name, unit in (("voltage", "V"), ("current", "A")):
                ds = g.create_dataset(name, shape=(n,), dtype=np.float32, chunks=(min(max(n, 1), 1 << 16),)
                                      if n else None, **(HDF5_COMPRESSION if n else {}))
                ds.attrs["units"] = unit
                datasets.append((ds, getattr(ch, name)))
            for name, runs in (("missing", ch.missing), ("voltage_clipped", ch.voltage_clipped),
                               ("current_clipped", ch.current_clipped)):
                ds = g.create_dataset(name, data=_slice_runs(runs, start, stop))
                ds.attrs["description"] = "[start, stop) sample indices relative to t0_s"
        done = 0
        for a in range(start, stop, BLOCK):
            b = min(stop, a + BLOCK)
            for ds, values in datasets:
                ds[a - start:b - start] = values[a:b]
            done = b - start
            _report(progress, done, total)
        if total == 0:
            _report(progress, 0, 0)


def _slice_runs(runs: IntervalList, start: int, stop: int) -> np.ndarray:
    arr = runs.array()
    if len(arr) == 0:
        return np.empty((0, 2), np.int64)
    a = np.clip(arr[:, 0], start, stop)
    b = np.clip(arr[:, 1], start, stop)
    keep = b > a
    return np.column_stack((a[keep] - start, b[keep] - start)).astype(np.int64)
