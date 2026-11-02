import math

import numpy as np
import pytest

from ngm_daq.capture import (
    PYRAMID_BASE, ChannelData, ChannelSettings, GapTracker, IntervalList, MinMaxPyramid, binned_raw, box_stats,
    clipped_mask, merge_intervals, samples_between, window_stats,
)
from ngm_daq.instrument import CHUNK_PERIOD, CURRENT_RANGES


def settings(current_range=0.1, voltage_range=5.0):
    return ChannelSettings(1, 4.2, 0.1, True, voltage_range, current_range)


def test_interval_list_merges_adjacent_and_overlapping():
    runs = IntervalList()
    runs.add(0, 5)
    runs.add(5, 8)        # adjacent: merged
    runs.add(10, 12)
    runs.add(9, 11)       # overlaps the last run and starts earlier
    assert runs.array().tolist() == [[0, 8], [9, 12]]
    assert runs.count_in(4, 10) == (5, 2)


def test_interval_mask():
    runs = IntervalList()
    runs.add_mask(np.array([0, 1, 1, 0, 1], bool), offset=100)
    assert runs.array().tolist() == [[101, 103], [104, 105]]


def test_merge_intervals():
    a = np.array([[0, 5], [10, 20]])
    b = np.array([[3, 7], [30, 31]])
    assert merge_intervals(a, b).tolist() == [[0, 7], [10, 20], [30, 31]]


@pytest.mark.parametrize("capacity,chunks", [(100_000, [1000, 37, 64, 5000, 12345]), (777, [100, 100, 577])])
def test_pyramid_matches_brute_force(capacity, chunks):
    rng = np.random.default_rng(1)
    data = rng.normal(size=capacity).astype(np.float32)
    data[500:700] = np.nan                 # a gap spanning whole and partial blocks
    pyr = MinMaxPyramid(capacity)
    n = 0
    for k in chunks:
        stop = min(capacity, n + k)
        base = (n // PYRAMID_BASE) * PYRAMID_BASE
        pyr.extend(base, data[base:stop])
        n = stop
    for level, size in enumerate(pyr.block_sizes):
        blocks = -(-n // size)
        for j in range(0, blocks, max(1, blocks // 50)):
            seg = data[j * size:min(n, (j + 1) * size)]
            if np.isnan(seg).all():
                assert math.isnan(pyr.mins[level][j])
            else:
                assert pyr.mins[level][j] == np.nanmin(seg)
                assert pyr.maxs[level][j] == np.nanmax(seg)


def test_pyramid_binned_covers_range():
    data = np.arange(1_000_000, dtype=np.float32)
    pyr = MinMaxPyramid(len(data))
    pyr.extend(0, data)
    edges = np.linspace(200_000, 800_000, 100, endpoint=False).astype(np.int64)
    level = pyr.level_for(6000)
    lo, hi = pyr.binned(level, 200_000, 800_000, edges)
    assert len(lo) == 100
    size = pyr.block_sizes[level]
    assert lo[0] <= 200_000 and lo[0] > 200_000 - size
    assert hi[-1] >= 799_999 and hi[-1] < 800_000 + size
    assert np.all(np.diff(lo) > 0)


def test_binned_raw():
    lo, hi = binned_raw(np.array([1, 5, 2, 8, 3, 0], np.float32), np.array([0, 2, 4]))
    assert lo.tolist() == [1, 2, 0] and hi.tolist() == [5, 8, 3]


def test_clipped_fixed_range():
    x = np.array([0.0, 0.1, 0.1125, 0.11407, -0.11411, 0.05], np.float32)
    assert clipped_mask(x, 0.1, CURRENT_RANGES).tolist() == [False, False, True, True, True, False]


def test_clipped_autorange_needs_repeated_value():
    upshift = np.full(10, 0.0113391, np.float32)                 # 10 mA range pinned during an upshift
    passing = np.linspace(0.0112, 0.0116, 10).astype(np.float32)  # a real signal crossing the same level
    x = np.concatenate(([0.002], upshift, [0.04, 0.03], passing, [0.002])).astype(np.float32)
    mask = clipped_mask(x, None, CURRENT_RANGES)
    assert mask[1:11].all()
    assert not mask[11:].any() and not mask[0]


def test_channel_append_and_missing():
    ch = ChannelData(settings(), capacity=10_000)
    block = np.column_stack((np.full(4000, 4.2), np.full(4000, 0.002))).astype(np.float32)
    assert ch.append(block) == 4000
    assert ch.append_missing(2500) == 2500
    assert ch.append(block) == 3500                 # capacity reached; the rest is for the next capture
    assert ch.full and ch.n == 10_000
    assert ch.missing.array().tolist() == [[4000, 6500]]
    assert np.isnan(ch.voltage[4000:6500]).all()
    flags = ch.flags(3999, 6501)
    assert flags[0] == 0 and flags[1] == 4 and flags[-2] == 4 and flags[-1] == 0


def test_clipping_found_across_chunks():
    ch = ChannelData(settings(current_range=None), capacity=100)
    pinned = np.full(6, 0.0113391, np.float32)
    first = np.column_stack((np.full(4, 4.2, np.float32), np.r_[0.001, 0.001, pinned[:2]].astype(np.float32)))
    second = np.column_stack((np.full(5, 4.2, np.float32), np.r_[pinned[2:], 0.001].astype(np.float32)))
    ch.append(first)
    ch.append(second)
    assert ch.current_clipped.array().tolist() == [[2, 8]]


def test_gap_tracker():
    g = GapTracker(chunk_samples=125_000)
    assert g.missing_before(10.0) == 0
    assert g.missing_before(10.0 + CHUNK_PERIOD + 0.06) == 0       # jitter
    assert g.missing_before(10.31 + 3 * CHUNK_PERIOD - 0.05) == 2 * 125_000


def test_window_stats_energy_and_charge():
    ch = ChannelData(settings(), capacity=1000)
    v = np.full(1000, 4.0, np.float32)
    i = np.r_[np.full(500, 0.01), np.full(500, 0.03)].astype(np.float32)
    ch.append(np.column_stack((v, i)))
    s = window_stats(ch, 0, 1000, interval=1e-3)
    assert s.mean["current"] == pytest.approx(0.02)
    assert s.mean["power"] == pytest.approx(0.08)
    assert s.energy_j == pytest.approx(0.08 * 1.0)            # 1000 samples × 1 ms = 1 s at 80 mW
    assert s.charge_c == pytest.approx(0.02)
    assert s.std["current"] == pytest.approx(0.01, rel=1e-5)
    assert s.min["current"] == pytest.approx(0.01) and s.max["current"] == pytest.approx(0.03)


def test_window_stats_skips_missing():
    ch = ChannelData(settings(), capacity=300)
    ch.append(np.column_stack((np.full(100, 1.0), np.full(100, 1.0))).astype(np.float32))
    ch.append_missing(100)
    ch.append(np.column_stack((np.full(100, 3.0), np.full(100, 1.0))).astype(np.float32))
    s = window_stats(ch, 0, 300, interval=1.0)
    assert s.valid == 200 and s.samples == 300
    assert s.mean["voltage"] == pytest.approx(2.0)
    assert s.missing == (100, 1)


def test_samples_between():
    assert samples_between(0.0105, 0.0200, 1e-3, 1000) == (11, 21)      # samples at 11 ms ... 20 ms
    assert samples_between(-1.0, 0.5, 1e-3, 100) == (0, 100)
    assert samples_between(0.2, 0.3, 1e-3, 100) == (100, 100)          # beyond the data: empty


def test_box_stats_counts_samples_between_the_levels():
    ch = ChannelData(settings(), capacity=400)
    v = np.full(400, 4.0, np.float32)
    i = np.tile(np.array([0.001, 0.010, 0.020, 0.050], np.float32), 100)
    ch.append(np.column_stack((v[:300], i[:300])))
    ch.append_missing(100)
    s = box_stats(ch, "current", 0, 400, 0.005, 0.030)
    assert s.span == 300 and s.count == 150
    assert s.min == pytest.approx(0.010) and s.max == pytest.approx(0.020)
    assert s.mean == pytest.approx(0.015) and s.std == pytest.approx(0.005, rel=1e-5)
    p = box_stats(ch, "power", 4, 8, 0.0, 0.1)                          # 4, 40, 80 mW; 200 mW is above
    assert p.count == 3 and p.max == pytest.approx(0.080)
    empty = box_stats(ch, "voltage", 0, 400, 5.0, 6.0)
    assert empty.count == 0 and math.isnan(empty.mean)
