"""Shared fixtures: a Qt application (offscreen) and the instrument simulator on a free port."""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ngm_simulator  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sim():
    simulator = ngm_simulator.SimulatedNGM("NGM202")
    server = ngm_simulator.Server(("127.0.0.1", 0), simulator, verbose=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield simulator, server.server_address[1]
    server.shutdown()
    server.server_close()


def wait_for(app, predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")
