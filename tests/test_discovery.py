"""Instrument discovery: merging what mDNS, PyVISA and LXI pages report, without any SCPI."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ngm_daq import discovery
from ngm_daq.discovery import FoundInstrument, host_of


@pytest.fixture(autouse=True)
def no_lan_scan():
    """Overrides conftest's stub: these tests use the real discover(), with its sources faked."""


LXI_PAGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<LXIDevice xmlns="http://www.lxistandard.org/InstrumentIdentification/1.0">
  <Manufacturer>Rohde&amp;Schwarz</Manufacturer>
  <Model>NGM202</Model>
  <SerialNumber>103062</SerialNumber>
  <FirmwareRevision>04.013 0092C027A93</FirmwareRevision>
</LXIDevice>"""


class FakeVisa:
    """A resource manager that only answers list_resources, and records every call."""

    def __init__(self, resources):
        self.resources, self.calls = resources, []

    def list_resources(self, query):
        self.calls.append(("list_resources", query))
        return self.resources

    def __getattr__(self, name):           # open_resource or anything else would talk to an instrument
        raise AssertionError(f"discovery called {name}")


def test_merges_mdns_and_visa_and_asks_lxi_only_when_needed(monkeypatch):
    monkeypatch.setattr(discovery, "_browse_mdns", lambda add, browse_time: add(
        "10.101.0.120", manufacturer="Rohde&Schwarz", model="NGM202", serial="103062"))
    asked = []
    monkeypatch.setattr(discovery, "lxi_identification",
                        lambda host, timeout=None: asked.append(host) or {"model": "NGM201", "serial": "1"})
    rm = FakeVisa(("TCPIP::10.101.0.120::INSTR", "TCPIP::10.101.0.9::inst0::INSTR", "USB0::0x0AAD::0x0197::1::INSTR"))
    found = discovery.discover(rm, browse_time=0.05)
    assert found == [
        FoundInstrument("10.101.0.9", model="NGM201", serial="1"),     # VISA only: identity from its LXI page
        FoundInstrument("10.101.0.120", "Rohde&Schwarz", "NGM202", "103062"),    # in IP order
        FoundInstrument("USB0::0x0AAD::0x0197::1::INSTR"),
    ]
    assert asked == ["10.101.0.9"]                     # mDNS already said who 10.101.0.120 is
    assert rm.calls == [("list_resources", "?*::INSTR")]
    assert found[1].identity == "NGM202 · S/N 103062"


def test_lxi_identification_page():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            ok = self.path == "/lxi/identification/"
            self.send_response(200 if ok else 404)
            self.end_headers()
            if ok:
                self.wfile.write(LXI_PAGE)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        host = f"127.0.0.1:{server.server_address[1]}"
        assert discovery.lxi_identification(host) == {
            "manufacturer": "Rohde&Schwarz", "model": "NGM202", "serial": "103062"}
    finally:
        server.shutdown()
        server.server_close()
    assert discovery.lxi_identification("127.0.0.1:9", timeout=0.5) == {}      # nothing there


def test_host_of():
    assert host_of("TCPIP::10.101.0.120::5025::SOCKET") == "10.101.0.120"
    assert host_of("TCPIP0::10.101.0.120::hislip0::INSTR") == "10.101.0.120"
    assert host_of(" 10.101.0.120 ") == "10.101.0.120"
    assert host_of("USB0::0x0AAD::0x0197::1::INSTR") == "USB0::0x0AAD::0x0197::1::INSTR"
