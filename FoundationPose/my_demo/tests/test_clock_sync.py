"""Deterministic NTP and runtime clock-gating tests; no sockets or clock changes."""
from dataclasses import replace
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock_sync import (ClockSettings, check_clock, ClockEpochGuard, decode_ntp_response,
                        ntp_timestamp, timestamp_ns, properties, validate_server)

NOW = 1800000000000000000


def response(offset_ms=0):
    t1, t4 = NOW, NOW + 2200000
    t2 = NOW + 1000000 + int(offset_ms * 1e6)
    t3 = t2 + 200000
    packet = bytearray(48)
    packet[0], packet[1] = 0x24, 2
    packet[24:32] = ntp_timestamp(t1)
    packet[32:40] = ntp_timestamp(t2)
    packet[40:48] = ntp_timestamp(t3)
    return packet, t1, t4


def service():
    return {'synchronized': True, 'backend': 'test_fixture', 'server_address': '192.0.2.1', 'server_name': None}


class NtpTests(unittest.TestCase):
    def test_known_unix_epoch_encoding(self):
        self.assertEqual(ntp_timestamp(0), struct.pack('!II', 2208988800, 0))
        self.assertEqual(ntp_timestamp(500000000), struct.pack('!II', 2208988800, 0x80000000))
    def test_timestamp_roundtrip_across_2036_era(self):
        for value in (NOW, 2200000000123456789):
            self.assertLessEqual(abs(timestamp_ns(ntp_timestamp(value), value) - value), 1)
    def test_signed_offsets_and_delay(self):
        for offset in (0, 20, -20, 400, -400):
            packet, t1, t4 = response(offset)
            result = decode_ntp_response(packet, bytes(packet[24:32]), t1, t4, t4-t1)
            self.assertAlmostEqual(result['offset_ms'], offset, places=5)
            self.assertAlmostEqual(result['round_trip_ms'], 2, places=5)
    def test_replayed_response_is_rejected(self):
        packet, t1, t4 = response()
        with self.assertRaisesRegex(ValueError, 'originate'):
            decode_ntp_response(packet, ntp_timestamp(NOW-1000000), t1, t4, t4-t1)
    def test_unsynchronized_and_invalid_server_packets(self):
        for first, stratum in ((0xe4,2), (0x23,2), (0x24,0), (0x24,16)):
            packet, t1, t4 = response(); packet[0]=first; packet[1]=stratum
            with self.assertRaises(ValueError): decode_ntp_response(packet,bytes(packet[24:32]),t1,t4,t4-t1)
    def test_clock_step_during_probe_is_rejected(self):
        packet,t1,t4=response()
        with self.assertRaisesRegex(ValueError,'clock changed'):
            decode_ntp_response(packet,bytes(packet[24:32]),t1,t4+20000000,t4-t1)
    def test_short_packet_is_rejected(self):
        with self.assertRaises(ValueError): decode_ntp_response(bytes(40),bytes(8),NOW,NOW,0)


class ReadinessTests(unittest.TestCase):
    def setUp(self): self.settings=ClockSettings(server='192.0.2.1')
    def probe(self,offset=0,delay=.2):
        return lambda *args: {'offset_ms':offset,'round_trip_ms':delay,
            'estimated_error_bound_ms':abs(offset)+delay/2,'server_address':'192.0.2.1','stratum':2,'leap':0}
    def test_system_sync_flag_alone_is_insufficient(self):
        for offset in (-400,-20,20,400):
            result=check_clock(self.settings,service,self.probe(offset))
            self.assertFalse(result['ready']); self.assertEqual(result['reason'],'clock_error_bound_exceeded')
    def test_healthy_source_can_publish(self):
        self.assertTrue(check_clock(self.settings,service,self.probe(.1))['ready'])
    def test_network_uncertainty_is_included(self):
        self.assertFalse(check_clock(self.settings,service,self.probe(0,10))['ready'])
    def test_os_service_must_use_the_selected_source(self):
        result=check_clock(self.settings,lambda:dict(service(),server_address='192.0.2.2'),self.probe())
        self.assertEqual(result['reason'],'system_service_uses_different_source')
    def test_os_service_must_be_synchronized(self):
        result=check_clock(self.settings,lambda:dict(service(),synchronized=False),self.probe())
        self.assertFalse(result['ready'])
    def test_missing_source_does_not_probe_the_network(self):
        def forbidden(*args): raise AssertionError('unexpected probe')
        result=check_clock(ClockSettings(),forbidden,forbidden)
        self.assertEqual(result['reason'],'server_not_configured')
    def test_missing_responses_fail_closed(self):
        def timeout(*args): raise TimeoutError('no NTP reply')
        self.assertEqual(check_clock(self.settings,service,timeout)['reason'],'ntp_probe_failed')
    def test_loopback_test_mode_cannot_target_a_robot(self):
        with self.assertRaises(ValueError): ClockSettings(mode='test',server='192.0.2.1',port=41234)
        with self.assertRaises(ValueError): ClockSettings(mode='test',server='127.0.0.1',port=123)
    def test_local_mode_is_explicit_and_does_not_query(self):
        def forbidden(*args): raise AssertionError('unexpected network/service access')
        self.assertEqual(check_clock(ClockSettings(mode='local'),forbidden,forbidden)['reason'],'same_host_only')
    def test_server_configuration_rejects_injected_directives(self):
        for server in ('host\nFallbackNTP=evil', '-bad', 'a b', 'ntp://host'):
            with self.assertRaises(ValueError): validate_server(server)


class ClockEpochTests(unittest.TestCase):
    def test_unsync_and_recovery_invalidate_old_jobs(self):
        guard=ClockEpochGuard()
        self.assertFalse(guard.observe(False,NOW,0).ready)
        first=guard.observe(True,NOW+1000000,1000000)
        self.assertTrue(first.ready)
        lost=guard.observe(False,NOW+2000000,2000000)
        restored=guard.observe(True,NOW+3000000,3000000)
        self.assertGreater(restored.epoch, first.epoch)
        self.assertFalse(lost.ready); self.assertTrue(restored.ready)
    def test_forward_and_backward_steps_stop_publication(self):
        for offset in (20000000,-20000000,400000000,-400000000):
            guard=ClockEpochGuard(); first=guard.observe(True,NOW,0)
            stepped=guard.observe(True,NOW+1000000+offset,1000000)
            self.assertFalse(stepped.ready); self.assertTrue(stepped.jumped)
            self.assertGreater(stepped.epoch,first.epoch)
    def test_sampling_scheduler_delay_is_not_a_clock_step(self):
        guard=ClockEpochGuard(); guard.observe(True,NOW,0)
        observed=guard.observe(True,NOW+20000000,10000000,uncertainty_ns=10000000)
        self.assertFalse(observed.jumped)
    def test_steady_clock_does_not_create_new_epochs(self):
        guard=ClockEpochGuard(); first=guard.observe(True,NOW,0)
        self.assertEqual(guard.observe(True,NOW+5000000,5000000).epoch,first.epoch)


if __name__=='__main__': unittest.main()
