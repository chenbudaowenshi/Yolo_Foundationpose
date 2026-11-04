#!/usr/bin/env python3
"""Loopback-only NTP test source. Offsets are simulated; the OS clock is unchanged."""
from __future__ import annotations
import argparse
import ipaddress
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock_sync import ntp_timestamp


class LocalNtpTestServer:
    def __init__(self, port=0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.settimeout(.1)
        self.port = self.socket.getsockname()[1]
        self.offset_ms = 0.0
        self.synchronized = True
        self.respond = True
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='local-ntp-test-source', daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, peer = self.socket.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            offset_ns = int(self.offset_ms * 1e6)
            received = time.time_ns() + offset_ns
            if len(data) < 48 or data[0] & 7 != 3 or not self.respond:
                continue
            packet = bytearray(48)
            packet[0] = 0x24 | (0 if self.synchronized else 0xc0)
            packet[1] = 1
            packet[3] = 0xec  # precision -20
            packet[12:16] = b'TEST'
            packet[16:24] = ntp_timestamp(received)
            packet[24:32] = data[40:48]
            packet[32:40] = ntp_timestamp(received)
            packet[40:48] = ntp_timestamp(time.time_ns() + offset_ns)
            try:
                self.socket.sendto(packet, peer)
            except OSError:
                break

    def close(self):
        self._stop.set()
        self.thread.join(timeout=.3)
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=41234)
    parser.add_argument('--offset-ms', type=float, default=0)
    parser.add_argument('--duration', type=float, default=None)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('test server requires an unprivileged port')
    server = LocalNtpTestServer(args.port)
    server.offset_ms = args.offset_ms
    print(json.dumps({'server':'127.0.0.1','port':server.port,'simulated_offset_ms':args.offset_ms,
                      'os_clock_changed':False}), flush=True)
    started = time.monotonic()
    try:
        while args.duration is None or time.monotonic() - started < args.duration:
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == '__main__': main()
