"""Persistent-sequence invariants, including a real child-process crash."""
import json
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'my_demo'), str(ROOT/'sb_vision_handoff_20260910')]
from sequence_store import PersistentSequence, SequenceStateError, MAX_SEQUENCE
from sb_pose_contract import PublicationGate
from shuangbi_grasp.contracts import ObjectPoseSample
from shuangbi_grasp.target_manager import TargetValidator


class SequenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sb_sequence_test_', dir='/tmp')
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def open(self, **kwargs):
        return PersistentSequence(self.directory, domain_id=10, topic='/object_pose', block_size=4, **kwargs)

    def test_missing_history_requires_explicit_bootstrap(self):
        with self.assertRaisesRegex(SequenceStateError, 'not initialized'):
            self.open()
        with self.open(start_after=800) as store:
            self.assertEqual(store.next(), 801)

    def test_receiver_800_accepts_first_new_value_and_restart(self):
        validator = TargetValidator()
        def sample(seq, stamp):
            return ObjectPoseSample('bottle','camera_optical',[0,0,.5],[1,0,0,0],stamp,stamp,sequence=seq)
        validator.accept(sample(800, 10))
        with self.open(start_after=800) as first:
            new = sample(first.next(),11)
            self.assertEqual(new.sequence,801)
            self.assertIsNone(validator.check(new,11)); validator.accept(new)
        with self.open() as resumed:
            new = sample(resumed.next(),12)
            self.assertGreater(new.sequence,801)
            self.assertIsNone(validator.check(new,12)); validator.accept(new)
        self.assertEqual(validator.check(sample(800,13),13),'out_of_order')

    def test_normal_restart_skips_unused_reserved_numbers(self):
        with self.open(start_after=0) as first:
            self.assertEqual([first.next(),first.next()],[1,2])
            saved=json.loads(first.path.read_text())['reserved_through']
        with self.open() as second:
            self.assertEqual(second.next(),saved+1)

    def test_killed_process_recovers_without_shutdown_save(self):
        code='''
import sys
sys.path.insert(0,sys.argv[2])
from sequence_store import PersistentSequence
s=PersistentSequence(sys.argv[1],domain_id=10,topic='/object_pose',start_after=800,block_size=4)
print(s.next(),flush=True)
sys.stdin.read()
'''
        process=subprocess.Popen([sys.executable,'-c',code,str(self.directory),str(ROOT/'my_demo')],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertTrue(select.select([process.stdout],[],[],5)[0],'child did not acquire/publish in time')
            self.assertEqual(process.stdout.readline().strip(),'801')
            process.kill(); process.wait(timeout=5)  # deliberately no allocator.close()
        finally:
            if process.poll() is None: process.kill()
            process.communicate(timeout=5)
        with self.open() as resumed:
            self.assertEqual(resumed.next(),805)

    def test_duplicate_publisher_cannot_take_same_stream(self):
        with self.open(start_after=0) as first:
            with self.assertRaisesRegex(SequenceStateError,'Another publisher'):
                self.open(start_after=0)
            self.assertEqual(first.next(),1)

    def test_domains_and_topics_have_independent_state(self):
        with self.open(start_after=800) as first:
            with PersistentSequence(self.directory,domain_id=11,topic='/object_pose',start_after=0) as other:
                self.assertNotEqual(first.path,other.path); self.assertEqual(other.next(),1)
            with PersistentSequence(self.directory,domain_id=10,topic='/other/object_pose',start_after=0) as other:
                self.assertNotEqual(first.path,other.path); self.assertEqual(other.next(),1)

    def test_repeated_bootstrap_flag_never_rewinds(self):
        with self.open(start_after=800) as first: first.next()
        with self.open(start_after=0) as second: self.assertEqual(second.next(),805)
        with self.open(start_after=1000) as third: self.assertEqual(third.next(),1001)

    def test_corruption_is_not_silently_reinitialized(self):
        with self.open(start_after=0) as store: path=store.path
        path.write_text('{unfinished')
        with self.assertRaisesRegex(SequenceStateError,'Cannot trust'):
            self.open(start_after=800)

    def test_checksum_detects_changed_high_water_mark(self):
        with self.open(start_after=800) as store: path=store.path
        data=json.loads(path.read_text()); data['reserved_through']=0; path.write_text(json.dumps(data))
        with self.assertRaisesRegex(SequenceStateError,'checksum'):
            self.open()

    def test_deleted_state_with_existing_marker_is_refused(self):
        with self.open(start_after=800) as store: path=store.path
        path.unlink()
        with self.assertRaisesRegex(SequenceStateError,'disappeared'):
            self.open(start_after=0)

    def test_corrupt_lock_marker_is_refused(self):
        with self.open(start_after=800) as store: path=store.lock_path
        path.write_text('partial marker')
        with self.assertRaisesRegex(SequenceStateError,'lock marker'):
            self.open()

    def test_hot_path_does_not_write_each_frame(self):
        with self.open(start_after=0) as store:
            with patch('sequence_store.os.fsync',side_effect=AssertionError('per-frame fsync')):
                self.assertEqual([store.next() for _ in range(4)],[1,2,3,4])
            self.assertEqual(store.next(),5)
            self.assertEqual(json.loads(store.path.read_text())['reserved_through'],8)

    def test_failed_reservation_never_returns_an_uncommitted_number(self):
        with self.open(start_after=0) as store:
            for _ in range(4): store.next()
            with patch('sequence_store.os.replace',side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(SequenceStateError,'durably reserve'): store.next()
            with self.assertRaisesRegex(SequenceStateError,'failed'): store.next()
            self.assertEqual(store.current,4)
        with self.open() as resumed: self.assertEqual(resumed.next(),5)

    def test_uint32_exhaustion_does_not_wrap(self):
        with self.open(start_after=MAX_SEQUENCE-2) as store:
            self.assertEqual(store.next(),MAX_SEQUENCE-1)
            self.assertEqual(store.next(),MAX_SEQUENCE)
            with self.assertRaisesRegex(SequenceStateError,'exhausted'): store.next()
        with self.assertRaisesRegex(SequenceStateError,'exhausted'): self.open()

    def test_rejected_observation_does_not_allocate_a_number(self):
        with self.open(start_after=800) as store:
            gate=PublicationGate(sequence_allocator=store)
            frame=SimpleNamespace(stamp_ns=1800000000000000000)
            self.assertIsNone(gate.admit(frame,frame.stamp_ns+400000000))
            self.assertEqual(store.current,800)
            self.assertEqual(gate.admit(frame,frame.stamp_ns),801)
            self.assertIsNone(gate.admit(frame,frame.stamp_ns))
            self.assertEqual(store.current,801)


if __name__=='__main__': unittest.main()
