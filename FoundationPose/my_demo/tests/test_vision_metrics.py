import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_metrics import Distribution, VisionMetrics, ProcessGpuSampler
from sb_vision_pipeline import PosePipeline, PosePipelineError
from test_sb_handoff import FakeBackend, scene, load_object_profile, ROOT, EPOCH


class PipelineEventTests(unittest.TestCase):
    def setUp(self):
        self.profile=load_object_profile(ROOT/'my_demo/config/bottle_current.json')

    def test_first_tracking_and_reset_are_distinct_without_resetting_attempts(self):
        p=PosePipeline(FakeBackend(),self.profile)
        first=p.process(scene()[0]); tracking=p.process(scene(EPOCH+100_000_000)[0])
        p.reset_target('manual_reset',trigger_mono_ns=1234)
        repeated=p.process(scene(EPOCH+200_000_000)[0])
        self.assertEqual((first.mode,tracking.mode,repeated.mode),('initial_registration','tracking','reregistration'))
        self.assertEqual((first.registration_attempt,repeated.registration_attempt),(1,2))
        self.assertEqual(repeated.trigger_reason,'manual_reset')
        self.assertEqual(repeated.trigger_mono_ns,1234)
        self.assertEqual(repeated.operation,'register')
        self.assertFalse(repeated.full_search_requested)

    def test_failed_initial_registration_is_followed_by_full_retry(self):
        backend=FakeBackend();backend.pose[2,3]+=.2
        p=PosePipeline(backend,self.profile)
        first=p.process(scene()[0])
        backend.pose=scene()[2].copy()
        second=p.process(scene(EPOCH+100_000_000)[0])
        self.assertIsNone(first.pose_center)
        self.assertEqual(second.mode,'reregistration')
        self.assertEqual(second.trigger_reason,'registration_quality_rejected')
        self.assertTrue(second.full_search_requested)
        self.assertEqual(second.status,'valid')

    def test_occlusion_reason_and_original_tracking_policy(self):
        backend=FakeBackend();p=PosePipeline(backend,self.profile)
        p.process(scene()[0]);backend.detections=[]
        p.process(scene(EPOCH+400_000_000)[0]);backend.detections=[scene()[1]]
        result=p.process(scene(EPOCH+500_000_000)[0])
        self.assertEqual(result.trigger_reason,'target_not_observed')
        self.assertEqual(result.mode,'reregistration')

    def test_exception_retains_operation_and_elapsed_time(self):
        class Broken(FakeBackend):
            def register(self,*args): raise RuntimeError('GPU test failure')
        p=PosePipeline(Broken(),self.profile)
        with self.assertRaises(PosePipelineError) as raised: p.process(scene()[0])
        result=raised.exception.result
        self.assertEqual(result.mode,'initial_registration')
        self.assertEqual(result.status,'inference_error')
        self.assertIn('GPU test failure',result.error)
        self.assertGreaterEqual(result.timings['pose_s'],0)
        self.assertGreaterEqual(result.timings['pipeline_s'],result.timings['pose_s'])

    def test_memory_probe_failure_does_not_change_pose_result(self):
        class BrokenProbe(FakeBackend):
            def memory_snapshot(self): raise RuntimeError('probe unavailable')
        result=PosePipeline(BrokenProbe(),self.profile).process(scene()[0])
        self.assertEqual(result.status,'valid')
        self.assertFalse(result.gpu_memory['available'])

    def test_actual_backend_search_flag_is_recorded(self):
        class ForcedBackend(FakeBackend):
            def registration_details(self): return {'force_full_requested':True,'search_executed':True,'full_search':True}
        result=PosePipeline(ForcedBackend(),self.profile).process(scene()[0])
        self.assertTrue(result.full_search_requested)
        self.assertTrue(result.search['full_search'])

    def test_detector_is_decimated_but_tracking_runs_on_each_frame(self):
        backend = FakeBackend()
        calls = []
        original = backend.detect
        def detect(frame):
            calls.append(frame.stamp_ns)
            return original(frame)
        backend.detect = detect
        pipeline = PosePipeline(backend, self.profile, detector_hz=10)
        first = pipeline.process(scene()[0])
        second = pipeline.process(scene(EPOCH + 50_000_000)[0])
        third = pipeline.process(scene(EPOCH + 100_000_000)[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual((first.mode, second.mode, third.mode),
                         ('initial_registration', 'tracking', 'tracking'))
        self.assertEqual(backend.tracks, 2)
        self.assertFalse(second.detection_fresh)
        self.assertTrue(third.detection_fresh)

    def test_detector_miss_uses_short_grace_then_stops(self):
        backend = FakeBackend()
        pipeline = PosePipeline(backend, self.profile, detector_hz=10, miss_grace_s=.30)
        self.assertEqual(pipeline.process(scene()[0]).status, 'valid')
        backend.detections = []
        # The detector is not due yet, so tracking can bridge this short gap.
        bridged = pipeline.process(scene(EPOCH + 50_000_000)[0])
        self.assertEqual(bridged.status, 'valid')
        # At the detector refresh, the miss is visible and no stale pose is emitted.
        missed = pipeline.process(scene(EPOCH + 100_000_000)[0])
        self.assertEqual(missed.status, 'no_detection')

    def test_camera_gap_keeps_pose_seed_for_local_track(self):
        backend = FakeBackend()
        pipeline = PosePipeline(backend, self.profile)
        pipeline.process(scene()[0])
        pipeline.process(scene(EPOCH + 100_000_000)[0])
        result = pipeline.process(scene(EPOCH + 700_000_000)[0])
        self.assertEqual(result.mode, 'tracking')
        self.assertEqual(backend.registers, 1)
        self.assertEqual(backend.tracks, 2)


def record(mode='tracking', published=False, **extra):
    return dict(mode=mode,operation='track',status='valid' if published else 'no_detection',
                stamp_ns=EPOCH,sequence=1,estimate_valid=True,published=published,
                timings={'pose_s':.01}, **extra)


class MetricSummaryTests(unittest.TestCase):
    def test_recovery_ends_at_publication_not_at_registration_completion(self):
        m=VisionMetrics({'input_source':'synthetic'},now_ns=0)
        first=record('initial_registration',False,trigger_reason='startup',trigger_mono_ns=1,
                     operation_start_mono_ns=20_000_000)
        first['operation']='register';first['status']='stale_future_or_duplicate_result'
        m.record(first,now_ns=100_000_000)
        self.assertEqual(m.recovery_duration.count,0)
        good=m.record(record(published=True),now_ns=200_000_000)
        self.assertAlmostEqual(good['completed_recovery']['duration_s'],.2)
        self.assertAlmostEqual(good['completed_recovery']['first_registration_start_to_output_s'],.18)

    def test_repeated_missing_frames_do_not_restart_recovery_timer(self):
        m=VisionMetrics({},now_ns=0);m.record(record(published=True),now_ns=100_000_000)
        m.begin_recovery('manual_reset',200_000_000)
        m.record(record(),now_ns=400_000_000);m.record(record(),now_ns=600_000_000)
        done=m.record(record(published=True),now_ns=700_000_000)
        self.assertEqual(done['completed_recovery']['reason'],'manual_reset')
        self.assertAlmostEqual(done['completed_recovery']['duration_s'],.5)

    def test_no_output_is_pending_not_zero_latency_success(self):
        m=VisionMetrics({},now_ns=0);m.record(record(),now_ns=100_000_000)
        result=m.snapshot(final=True,now_ns=1_000_000_000)
        self.assertIsNone(result['time_to_first_valid_output_s'])
        self.assertIsNone(result['recovery_duration_seconds']['p50'])
        self.assertEqual(result['pending_recovery']['elapsed_s'],1.)
        self.assertEqual(result['operations']['initial_registration']['attempts'],0)

    def test_camera_silence_starts_timeout_episode(self):
        m=VisionMetrics({},now_ns=0);m.record(record(published=True),now_ns=100_000_000)
        self.assertFalse(m.check_output_timeout(.3,now_ns=300_000_000))
        self.assertTrue(m.check_output_timeout(.3,now_ns=500_000_000))
        self.assertEqual(m.active_recovery['started_mono_ns'],400_000_000)

    def test_search_request_and_execution_are_separate(self):
        m=VisionMetrics({},now_ns=0)
        m.record(record('reregistration',full_search_requested=True,search={'full_search':False}),now_ns=10)
        stats=m.snapshot(now_ns=20)['operations']['reregistration']
        self.assertEqual(stats['full_search_requests'],1)
        self.assertEqual(stats['full_search_executions'],0)

    def test_bounded_quantiles_are_labelled_and_max_retains_whole_run(self):
        d=Distribution(capacity=3)
        for value in (1000,0,1,2,3,4): d.add(value)
        summary=d.snapshot()
        self.assertEqual(summary['count'],6)
        self.assertEqual(summary['max'],1000)
        self.assertEqual(summary['p50'],3)
        self.assertEqual(summary['quantile_scope'],'latest_bounded_samples')

    def test_torch_session_peak_survives_probe_failure_and_lower_later_peak(self):
        m=VisionMetrics({},now_ns=0)
        m.record(record(gpu_memory={'available':True,'peak_allocated_bytes':900}),now_ns=1)
        m.record(record(gpu_memory={'available':False,'error':'probe failed'}),now_ns=2)
        m.record(record(gpu_memory={'available':True,'peak_allocated_bytes':800}),now_ns=3)
        self.assertEqual(m.torch_memory['peak_allocated_bytes'],900)
        self.assertTrue(m.torch_memory['available'])
        self.assertNotIn('error',m.torch_memory)

    def test_shutdown_discard_does_not_invent_a_new_outage(self):
        m=VisionMetrics({},now_ns=0);m.record(record(published=True),now_ns=1)
        event=record();event['status']='shutdown_discard'
        m.record(event,now_ns=2)
        self.assertIsNone(m.active_recovery)

    def test_model_load_is_visible_even_if_first_frame_has_no_target(self):
        m=VisionMetrics({},now_ns=0)
        event=record('none');event['timings']={'model_load_s':7.5,'yolo_s':.4}
        m.record(event,now_ns=8_000_000_000)
        self.assertEqual(m.snapshot(now_ns=8_000_000_000)['model_load_seconds']['p50'],7.5)

    def test_nvml_counts_only_current_pid_once(self):
        own=NS(pid=os.getpid(),usedGpuMemory=100)
        other=NS(pid=os.getpid()+1,usedGpuMemory=9000)
        module=NS(nvmlInit=lambda:None,nvmlShutdown=lambda:None,nvmlDeviceGetCount=lambda:1,
                  nvmlDeviceGetHandleByIndex=lambda i:i,nvmlDeviceGetName=lambda h:'test GPU',
                  nvmlDeviceGetUUID=lambda h:'test UUID',nvmlDeviceGetComputeRunningProcesses=lambda h:[own,other],
                  nvmlDeviceGetGraphicsRunningProcesses=lambda h:[own],NVMLError=RuntimeError)
        with patch.dict(sys.modules,{'pynvml':module}):
            sampler=ProcessGpuSampler(.001)
            end=time.monotonic()+1
            while sampler.snapshot()['sample_count']==0 and time.monotonic()<end: time.sleep(.001)
            sampler.close();result=sampler.snapshot()
        self.assertEqual(result['process_peak_bytes'],100)


if __name__=='__main__': unittest.main()
