"""Find instruments on the LAN, with their model and serial number, without sending them any command.

A SCPI query such as *IDN? would put an instrument into Remote and lock its front panel, which
a scan shouldn't do to an instrument someone may be using. So identity comes from what LXI
instruments publish anyway:

- mDNS announcements (_lxi._tcp, _scpi-raw._tcp, ...), whose TXT records carry Model and
  SerialNumber. The NGM202 (firmware 04.013) announces _lxi and _scpi-raw with both;
- the LXI identification page, http://<host>/lxi/identification/, for an instrument found only
  by PyVISA's VXI-11 broadcast.

PyVISA's own scan also runs, so instruments that announce nothing (or USB ones) are still listed.
"""

from __future__ import annotations

import ipaddress
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

MDNS_SERVICES = ("_lxi._tcp.local.", "_hislip._tcp.local.", "_vxi-11._tcp.local.", "_scpi-raw._tcp.local.")
BROWSE_TIME = 2.0            # s to listen for mDNS announcements
LXI_TIMEOUT = 1.5            # s for an LXI identification page
LXI_MAX_BYTES = 65_536


@dataclass(frozen=True)
class FoundInstrument:
    address: str             # IP address, or a full VISA resource string (e.g. USB)
    manufacturer: str = ""
    model: str = ""
    serial: str = ""

    @property
    def identity(self) -> str:
        """E.g. "NGM202 · S/N 103062"; empty if the instrument didn't say."""
        if self.model and self.serial:
            return f"{self.model} · S/N {self.serial}"
        return self.model


def host_of(address: str) -> str:
    """The host of an IP address or TCPIP resource string (TCPIP::10.1.2.3::...); others unchanged."""
    m = re.match(r"\s*TCPIP\d*::([^:]+)::", address, re.IGNORECASE)
    return m.group(1) if m else address.strip()


def _sort_key(address: str):
    try:
        return (0, ipaddress.ip_address(address).packed, "")
    except ValueError:
        return (1, b"", address)


def discover(rm=None, browse_time: float = BROWSE_TIME) -> list[FoundInstrument]:
    """Instruments found on the LAN (and USB, if PyUSB is installed), IP addresses in order."""
    found: dict[str, dict[str, str]] = {}
    lock = threading.Lock()

    def add(address: str, **identity: str) -> None:
        with lock:
            entry = found.setdefault(address, {})
            for key, value in identity.items():
                if value and not entry.get(key):
                    entry[key] = value

    threads = [threading.Thread(target=_browse_mdns, args=(add, browse_time), daemon=True)]
    if rm is not None:
        threads.append(threading.Thread(target=_scan_visa, args=(add, rm), daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join(browse_time + 10)
    with lock:
        snapshot = {a: dict(i) for a, i in found.items()}
    for address, identity in snapshot.items():
        if not identity.get("model") and _sort_key(address)[0] == 0:
            identity.update(lxi_identification(address))
    return [FoundInstrument(a, **i) for a, i in sorted(snapshot.items(), key=lambda kv: _sort_key(kv[0]))]


def _scan_visa(add, rm) -> None:
    try:
        resources = rm.list_resources("?*::INSTR")
    except Exception:
        return
    for resource in resources:
        add(host_of(resource))


def _browse_mdns(add, browse_time: float) -> None:
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return

    class Listener:
        def add_service(self, zc, service_type, name):
            info = zc.get_service_info(service_type, name, 1500)
            if info is None:
                return
            props = {k.decode(errors="replace"): (v or b"").decode(errors="replace")
                     for k, v in info.properties.items()}
            for address in info.parsed_addresses():
                if ":" not in address:              # IPv4 only, as typed in the Address box
                    add(address, manufacturer=props.get("Manufacturer", ""), model=props.get("Model", ""),
                        serial=props.get("SerialNumber", ""))

        def update_service(self, *args):
            pass

        def remove_service(self, *args):
            pass

    try:
        zc = Zeroconf()
    except Exception:
        return
    try:
        browsers = [ServiceBrowser(zc, t, Listener()) for t in MDNS_SERVICES]
        time.sleep(browse_time)
        for b in browsers:
            b.cancel()
    finally:
        zc.close()


def lxi_identification(host: str, timeout: float = LXI_TIMEOUT) -> dict[str, str]:
    """Manufacturer, model and serial from an LXI instrument's identification page, or {}."""
    try:
        with urllib.request.urlopen(f"http://{host}/lxi/identification/", timeout=timeout) as r:
            root = ET.fromstring(r.read(LXI_MAX_BYTES))
    except Exception:
        return {}
    fields = {el.tag.rsplit("}", 1)[-1]: (el.text or "").strip() for el in root}
    return {"manufacturer": fields.get("Manufacturer", ""), "model": fields.get("Model", ""),
            "serial": fields.get("SerialNumber", "")}
