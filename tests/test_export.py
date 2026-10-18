from datetime import datetime

import h5py
import numpy as np
import pytest

from ngm_daq import export
from ngm_daq.capture import Capture, ChannelData, ChannelSettings, RunConfig
from ngm_daq.instrument import ChannelSetup, DeviceInfo, SampleRate

DEVICE = DeviceInfo("TCPIP::127.0.0.1::5025::SOCKET", "Rohde&Schwarz", "NGM202", "3638.4472k03", "103062",
                    "04.013 0092C027A93", frozenset({"K103", "K104"}))


def make_capture(n=10_000):
    cfg = RunConfig((ChannelSetup(1, 5.0, 0.01), ChannelSetup(2)), SampleRate.R1K, duration=n / 1e3)
    channels = {}
    for c in (1, 2):
        ch = ChannelData(ChannelSettings(c, 4.2, 0.1, True, 5.0 if c == 1 else None, 0.01 if c == 1 else None), n)
        v = np.full(n, 4.2 if c == 1 else 3.0, np.float32)
        i = np.full(n, 0.002, np.float32)
        if c == 1:
            i[100:110] = 0.0113391          # clipped on the 10 mA range
        ch.append(np.column_stack((v[:5000], i[:5000])))
        if c == 1:
            ch.append_missing(250)
            ch.append(np.column_stack((v[5250:], i[5250:])))
        else:
            ch.append(np.column_stack((v[5000:], i[5000:])))
        channels[c] = ch
    return Capture(DEVICE, cfg, 1e-3, datetime(2026, 10, 18, 16, 30).astimezone(), channels)


def test_csv_round_trip(tmp_path):
    cap = make_capture()
    path = tmp_path / "cap.csv"
    assert export.save(path, cap, 50, 5300) == 5250
    lines = path.read_text().splitlines()
    meta = dict(line[2:].split(": ", 1) for line in lines if line.startswith("# "))
    assert meta["instrument"] == "Rohde&Schwarz NGM202" and meta["first_sample_index"] == "50"
    assert meta["ch1_current_range"] == "10 mA" and meta["ch2_current_range"] == "Auto"
    header_lines = sum(line.startswith("#") for line in lines)
    data = np.genfromtxt(path, delimiter=",", skip_header=header_lines, names=True)
    assert data.dtype.names == ("time_s", "ch1_voltage_V", "ch1_current_A", "ch1_flags",
                                "ch2_voltage_V", "ch2_current_A", "ch2_flags")
    assert len(data) == 5250
    assert data["time_s"][0] == pytest.approx(0.050)
    assert data["ch1_flags"][50] == export.FLAG_CURRENT_CLIPPED      # sample 100
    assert np.isnan(data["ch1_voltage_V"][5000 - 50]) and data["ch1_flags"][5000 - 50] == export.FLAG_MISSING
    assert data["ch2_voltage_V"][5000 - 50] == pytest.approx(3.0)


def test_hdf5_round_trip(tmp_path):
    cap = make_capture()
    path = tmp_path / "cap.h5"
    assert export.save(path, cap, 50, 5300) == 5250
    with h5py.File(path) as f:
        assert f.attrs["t0_s"] == pytest.approx(0.05) and f.attrs["samples"] == 5250
        assert f.attrs["sample_interval_s"] == pytest.approx(1e-3)
        v = f["ch1/voltage"][...]
        assert v.dtype == np.float32 and len(v) == 5250
        assert np.array_equal(v, cap.channels[1].voltage[50:5300], equal_nan=True)
        assert f["ch1/missing"][...].tolist() == [[4950, 5200]]
        assert f["ch1/current_clipped"][...].tolist() == [[50, 60]]
        assert f["ch2/missing"].shape == (0, 2)
        assert f["ch1"].attrs["current_range"] == "10 mA"


def test_cancel_removes_file(tmp_path):
    cap = make_capture()
    path = tmp_path / "cap.csv"
    with pytest.raises(export.Cancelled):
        export.save(path, cap, progress=lambda done, total: False)
    assert not path.exists()


def test_unknown_suffix(tmp_path):
    with pytest.raises(ValueError):
        export.save(tmp_path / "cap.txt", make_capture())
