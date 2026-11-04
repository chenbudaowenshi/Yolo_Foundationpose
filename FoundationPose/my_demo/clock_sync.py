"""OS clock-sync readiness and a fresh NTP measurement; never rewrites image time.

The OS service disciplines CLOCK_REALTIME. This module only verifies it against
one explicitly selected NTP source. Loopback test mode simulates clock offsets
without changing the host clock. NTP arithmetic follows RFC 5905 section 8.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import threading
import time

NTP_EPOCH = 2208988800
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'config/clock_sync.json'


@dataclass(frozen=True)
class ClockSettings:
    mode: str = 'required'
    server: str | None = None
    port: int = 123
    max_error_bound_ms: float = 2.0
    interval_s: float = 5.0
    timeout_s: float = .5
    samples: int = 3
    max_status_age_s: float = 15.0
    query_timeout_s: float = 8.0
    jump_limit_ms: float = 5.0

    def __post_init__(self):
        if self.mode not in {'required', 'local', 'test'}:
            raise ValueError('clock mode must be required, local or test')
        if self.server is not None:
            validate_server(self.server)
        if not 1 <= self.port <= 65535 or not 1 <= self.samples <= 10:
            raise ValueError('invalid NTP port or sample count')
        for name in ('max_error_bound_ms', 'interval_s', 'timeout_s', 'max_status_age_s', 'query_timeout_s', 'jump_limit_ms'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.mode == 'test':
            if self.server is None or not ipaddress.ip_address(self.server).is_loopback or self.port < 1024:
                raise ValueError('test mode requires a loopback IP and an unprivileged port')
        if self.mode == 'required' and self.port != 123:
            raise ValueError('production NTP must use port 123; use test mode for local simulation')


def validate_server(value):
    if not isinstance(value, str) or not value or len(value) > 253:
        raise ValueError('NTP server must be an IP address or hostname')
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        if not all(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
                   for label in value.rstrip('.').split('.')):
            raise ValueError('invalid NTP hostname')
        return value.lower().rstrip('.')


def add_clock_arguments(parser):
    parser.add_argument('--clock-config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--clock-sync-mode', choices=('required', 'local', 'test'), default=None)
    parser.add_argument('--clock-server', default=os.environ.get('SB_NTP_SERVER'))
    parser.add_argument('--clock-port', type=int, default=None)
    parser.add_argument('--clock-max-error-ms', type=float, default=None)


def settings_from_args(args):
    values = json.loads(args.clock_config.read_text()) if args.clock_config else {}
    values.pop('note', None)
    for field, attribute in (('mode', 'clock_sync_mode'), ('server', 'clock_server'),
                             ('port', 'clock_port'), ('max_error_bound_ms', 'clock_max_error_ms')):
        value = getattr(args, attribute, None)
        if value is not None:
            values[field] = value
    return ClockSettings(**values)


def ntp_timestamp(ns):
    seconds, remainder = divmod(int(ns), 1000000000)
    return struct.pack('!II', (seconds + NTP_EPOCH) & 0xffffffff, (remainder << 32) // 1000000000)


def timestamp_ns(data, reference_ns):
    seconds, fraction = struct.unpack('!II', data)
    reference_seconds = reference_ns // 1000000000 + NTP_EPOCH
    era = (reference_seconds - seconds + (1 << 31)) // (1 << 32)
    return (seconds + era * (1 << 32) - NTP_EPOCH) * 1000000000 + (fraction * 1000000000 >> 32)


def decode_ntp_response(data, request_stamp, t1_ns, t4_ns, elapsed_ns):
    if len(data) < 48:
        raise ValueError('short NTP response')
    leap, version, mode = data[0] >> 6, (data[0] >> 3) & 7, data[0] & 7
    if leap == 3 or mode != 4 or version not in (3, 4) or not 1 <= data[1] <= 15:
        raise ValueError('NTP server is unsynchronized, refused the query, or returned an invalid mode')
    if data[24:32] != request_stamp:
        raise ValueError('NTP originate timestamp does not match this request')
    if data[32:40] == bytes(8) or data[40:48] == bytes(8):
        raise ValueError('missing server timestamps')
    if abs((t4_ns - t1_ns) - elapsed_ns) > 5000000:
        raise ValueError('local wall clock changed during the NTP measurement')
    t2 = timestamp_ns(data[32:40], t1_ns)
    t3 = timestamp_ns(data[40:48], t4_ns)
    delay_ns = (t4_ns - t1_ns) - (t3 - t2)
    if t3 < t2 or delay_ns < -1000:
        raise ValueError('inconsistent NTP timestamp ordering')
    offset_ms = ((t2 - t1_ns) + (t3 - t4_ns)) / 2e6  # source minus local clock
    delay_ms = max(0, delay_ns) / 1e6
    return {'offset_ms': offset_ms, 'round_trip_ms': delay_ms,
            'estimated_error_bound_ms': abs(offset_ms) + delay_ms / 2,
            'stratum': int(data[1]), 'leap': leap}


def probe_ntp(server, port=123, timeout_s=.5):
    addresses = socket.getaddrinfo(server, port, type=socket.SOCK_DGRAM)
    family, kind, protocol, _, address = addresses[0]
    with socket.socket(family, kind, protocol) as sock:
        sock.settimeout(timeout_s)
        sock.connect(address)  # Only accept datagrams from the selected endpoint.
        packet = bytearray(48)
        packet[0] = 0x23  # version 4, client mode
        before = time.monotonic_ns()
        t1 = time.time_ns()
        packet[40:48] = ntp_timestamp(t1)
        sock.send(packet)
        data = sock.recv(512)
        t4 = time.time_ns()
        elapsed = time.monotonic_ns() - before
    result = decode_ntp_response(data, bytes(packet[40:48]), t1, t4, elapsed)
    result['server_address'] = address[0]
    return result


def properties(text):
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)


def source_matches(server, service):
    expected = validate_server(server)
    for value in (service.get('server_name'), service.get('server_address')):
        if value and validate_server(value) == expected:
            return True
    try:
        ipaddress.ip_address(expected)
        return False
    except ValueError:
        # chronyc -n reports an IP even when chronyd was configured with a name.
        addresses = {entry[4][0] for entry in socket.getaddrinfo(expected, 123, type=socket.SOCK_DGRAM)}
        return service.get('server_address') in addresses


def read_system_service():
    environment = dict(os.environ, LC_ALL='C')
    def run(command):
        return subprocess.run(command, text=True, capture_output=True, timeout=2, env=environment)
    try:
        result = run(['chronyc', '-n', 'tracking'])
        if result.returncode == 0:
            values = dict((k.strip(), v.strip()) for k, v in
                          (line.split(':', 1) for line in result.stdout.splitlines() if ':' in line))
            match = re.search(r'\(([^)]+)\)', values.get('Reference ID', ''))
            return {'backend': 'chrony', 'synchronized': values.get('Leap status', '').lower() == 'normal',
                    'server_address': match.group(1) if match else None, 'server_name': None}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    state = run(['timedatectl', 'show', '--property=NTP', '--property=NTPSynchronized'])
    source = run(['timedatectl', 'show-timesync', '--property=ServerName', '--property=ServerAddress'])
    values, selected = properties(state.stdout), properties(source.stdout)
    return {'backend': 'systemd-timesyncd',
            'synchronized': state.returncode == 0 and source.returncode == 0
                            and values.get('NTP') == 'yes' and values.get('NTPSynchronized') == 'yes',
            'server_address': selected.get('ServerAddress'), 'server_name': selected.get('ServerName'),
            'detail': (state.stderr + source.stderr).strip()[:500]}


def check_clock(settings, service_reader=read_system_service, probe=probe_ntp):
    report = {'ready': False, 'mode': settings.mode, 'server': settings.server,
              'max_error_bound_ms': settings.max_error_bound_ms}
    if settings.mode == 'local':
        return dict(report, ready=True, reason='same_host_only', scope='no cross-machine alignment claim')
    if not settings.server:
        return dict(report, reason='server_not_configured')
    target = settings.server
    if settings.mode == 'required':
        try:
            service = service_reader()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return dict(report, reason='system_service_unavailable', detail=str(exc))
        report['system_service'] = service
        if not service.get('synchronized'):
            return dict(report, reason='system_service_not_synchronized')
        try:
            matched = source_matches(settings.server, service)
        except (OSError, ValueError) as exc:
            return dict(report, reason='source_verification_failed', detail=str(exc))
        if not matched:
            return dict(report, reason='system_service_uses_different_source')
        target = service.get('server_address') or target
    readings, errors = [], []
    for _ in range(settings.samples):
        try:
            readings.append(probe(target, settings.port, settings.timeout_s))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if not readings:
        return dict(report, reason='ntp_probe_failed', errors=errors)
    # Lowest RTT reduces the influence of transient packet queuing; retain all
    # samples and the RTT-derived bound instead of claiming exact clock equality.
    best = min(readings, key=lambda row: row['round_trip_ms'])
    ready = best['estimated_error_bound_ms'] <= settings.max_error_bound_ms
    return dict(report, **best, ready=ready, reason='synchronized' if ready else 'clock_error_bound_exceeded',
                samples=readings, errors=errors,
                scope='loopback simulated source; OS clock unchanged' if settings.mode == 'test' else 'local host versus selected NTP source')


class ClockSyncMonitor:
    """Check outside the ROS executor, with a hard subprocess timeout including DNS."""
    def __init__(self, settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._generation = 0
        self._checked_at = None
        self._report = {'ready': False, 'mode': settings.mode, 'server': settings.server, 'reason': 'checking_clock'}
        self._thread = None
        if settings.mode != 'local':
            self._thread = threading.Thread(target=self._run, name='clock-sync-check', daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                generation = self._generation
            try:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--settings-json',
                    json.dumps(asdict(self.settings))], capture_output=True, text=True,
                    timeout=self.settings.query_timeout_s)
                report = json.loads(result.stdout)
                if not isinstance(report.get('ready'), bool):
                    raise ValueError('invalid clock checker result')
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                report = {'ready': False, 'mode': self.settings.mode, 'server': self.settings.server,
                          'reason': 'clock_checker_failed', 'detail': str(exc)[:500]}
            with self._lock:
                if generation == self._generation:
                    self._report, self._checked_at = report, time.monotonic()
            self._wake.wait(self.settings.interval_s)
            self._wake.clear()

    def snapshot(self):
        if self.settings.mode == 'local':
            return {'ready': True, 'mode': 'local', 'reason': 'same_host_only', 'check_age_s': 0.0}
        with self._lock:
            report = dict(self._report)
            age = None if self._checked_at is None else time.monotonic() - self._checked_at
        report['check_age_s'] = age
        if age is None or age > self.settings.max_status_age_s:
            report.update(ready=False, reason='clock_status_stale' if age is not None else 'checking_clock')
        return report

    def invalidate(self, reason):
        with self._lock:
            self._generation += 1
            self._report = {'ready': False, 'mode': self.settings.mode, 'reason': reason, 'server': self.settings.server}
            self._checked_at = time.monotonic()
        self._wake.set()

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=self.settings.query_timeout_s + 1)


@dataclass(frozen=True)
class ClockDecision:
    ready: bool
    epoch: int
    changed: bool
    jumped: bool


class ClockEpochGuard:
    def __init__(self, jump_limit_ms=5.0):
        self.limit_ns = int(jump_limit_ms * 1e6)
        self.previous = None
        self.ready = False
        self.epoch = 0

    def observe(self, healthy, wall_ns, mono_ns, uncertainty_ns=0):
        jumped = False
        if self.previous is not None:
            wall, mono, uncertainty = self.previous
            jumped = abs((wall_ns-wall) - (mono_ns-mono)) > self.limit_ns + uncertainty + uncertainty_ns
        self.previous = wall_ns, mono_ns, uncertainty_ns
        ready = bool(healthy and not jumped)
        changed = jumped or ready != self.ready
        if changed:
            self.epoch += 1
        self.ready = ready
        return ClockDecision(ready, self.epoch, changed, jumped)


def sample_clock_pair(wall_clock=time.time_ns, monotonic_clock=time.monotonic_ns):
    before = monotonic_clock()
    wall = wall_clock()
    after = monotonic_clock()
    return wall, (before+after)//2, (after-before)//2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_clock_arguments(parser)
    parser.add_argument('--settings-json', default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        settings = ClockSettings(**json.loads(args.settings_json)) if args.settings_json else settings_from_args(args)
        result = check_clock(settings)
    except (ValueError, OSError) as exc:
        result = {'ready': False, 'reason': 'clock_configuration_error', 'detail': str(exc)}
    print(json.dumps(result, allow_nan=False))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
