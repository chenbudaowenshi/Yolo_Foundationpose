#!/usr/bin/env python3
"""Real publisher process restarts with one continuously running ROS receiver.

Synthetic camera/pose backend; isolated topic and loopback Domain 75. The
receiver's TargetValidator is never reset. No robot or physical camera is used.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_demo'),str(ROOT/'my_demo/tests'),str(ROOT/'my_demo/scripts'),
              str(ROOT/'sb_vision_handoff_20260910')]
from test_sb_handoff import FakeBackend


class Backend(FakeBackend):
    def __init__(self, profile, args): super().__init__()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--child':
        from sb_vision_node import main as vision_main
        return vision_main(sys.argv[2:],backend_factory=Backend)
    os.environ.update(ROS_DOMAIN_ID='75',ROS_LOCALHOST_ONLY='0',RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
        CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces></General>'
                       '<Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>'
                       '<Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    import rclpy
    from rclpy.node import Node
    from sb_communication.msg import ObjectPose
    from sb_pose_contract import stamp_ns
    from shuangbi_grasp.contracts import ObjectPoseSample
    from shuangbi_grasp.target_manager import TargetValidator
    from run_realsense_viewer import RealSenseViewerNode,build_argument_parser
    from test_ros_handoff import FakeCamera
    topic='/sequence_restart_test/object_pose'
    output=ROOT/'runs/sb_vision/sequence_restart_20260919'
    output.mkdir(parents=True,exist_ok=True)
    os.environ['ROS_LOG_DIR']=str(output/'ros_logs')
    with tempfile.TemporaryDirectory(prefix='sb_restart_stream_',dir='/tmp') as directory:
        rclpy.init()
        camera=receiver=None
        children=[]; logs=[]
        try:
            camera=RealSenseViewerNode(build_argument_parser().parse_args(
                ['--width','100','--height','100','--publish-fps','20']),FakeCamera)
            receiver=Node('sequence_restart_receiver')
            validator=TargetValidator()
            validator.accept(ObjectPoseSample('bottle','head_d435i_optical_frame',[0,0,.7],[1,0,0,0],
                             time.time()-1,time.monotonic()-1,sequence=800))
            samples=[]; rejected=[]
            def receive(msg):
                sample=ObjectPoseSample(msg.object_id,msg.header.frame_id,
                    [msg.pose.position.x,msg.pose.position.y,msg.pose.position.z],
                    [msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z],
                    stamp_ns(msg.header)/1e9,time.monotonic(),confidence=msg.confidence,
                    covariance=np.asarray(msg.covariance).reshape(6,6),sequence=msg.sequence,
                    observation_delay=(receiver.get_clock().now().nanoseconds-stamp_ns(msg.header))/1e9)
                reason=validator.check(sample,sample.received)
                if reason: rejected.append({'sequence':msg.sequence,'reason':reason})
                else: validator.accept(sample)
                samples.append(msg.sequence)
            receiver.create_subscription(ObjectPose,topic,receive,10)
            def spin(seconds):
                end=time.monotonic()+seconds
                while time.monotonic()<end:
                    rclpy.spin_once(camera,timeout_sec=.001)
                    rclpy.spin_once(receiver,timeout_sec=.001)
            def until(predicate,timeout=8):
                end=time.monotonic()+timeout
                while not predicate() and time.monotonic()<end: spin(.02)
                assert predicate(), f'timeout; samples={samples[-5:]}; rejected={rejected}'
            def launch(label,bootstrap=False,duration=30,alias=False):
                command=[sys.executable,str(Path(__file__).resolve()),'--child',
                    '--sequence-state-dir',directory,'--clock-sync-mode','local',
                    '--object-pose-topic','/sequence_requested_alias' if alias else topic,
                    '--max-runtime-seconds',str(duration)]
                if bootstrap: command+=['--sequence-start-after','800']
                if alias: command+=['--ros-args','-r','/sequence_requested_alias:='+topic]
                log=(output/(label+'.log')).open('w'); logs.append(log)
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
                return child
            first=launch('first',bootstrap=True,duration=1.5)
            until(lambda:len(samples)>=8)
            assert samples[0]>800,samples[:3]  # DDS discovery can omit an initial publication.
            until(lambda:first.poll() is not None)
            assert first.returncode==0,(output/'first.log').read_text()
            spin(.15)
            before_normal=samples[-1]; normal_index=len(samples)
            # Remap a different requested name onto the same actual ROS topic.
            second=launch('after_normal_exit',alias=True)
            until(lambda:len(samples)>=normal_index+8)
            first_after_normal=samples[normal_index]
            assert first_after_normal>before_normal
            second.kill(); second.wait(timeout=5)  # no graceful close/save can run
            spin(.15)
            before_crash=samples[-1]; crash_index=len(samples)
            third=launch('after_sigkill')
            until(lambda:len(samples)>=crash_index+8)
            first_after_crash=samples[crash_index]
            assert first_after_crash>before_crash
            duplicate=launch('duplicate_publisher',alias=True)
            until(lambda:duplicate.poll() is not None)
            assert duplicate.returncode!=0
            assert 'Another publisher owns sequence stream' in (output/'duplicate_publisher.log').read_text()
            count=len(samples); until(lambda:len(samples)>=count+5)
            assert not rejected,rejected
            assert all(a<b for a,b in zip(samples,samples[1:])),samples
            assert len(list(Path(directory).glob('*.json')))==1, 'remapped topic used a different state file'
            replay=ObjectPoseSample('bottle','head_d435i_optical_frame',[0,0,.7],[1,0,0,0],
                                   time.time(),time.monotonic(),sequence=800)
            replay_reason=validator.check(replay,replay.received)
            assert replay_reason=='out_of_order'
            result={'status':'PASS','scope':'synthetic data over real ROS; publisher child processes genuinely restart',
                'receiver_initial_sequence':800,'first_received_sequence':samples[0],
                'before_normal_exit_last':before_normal,'after_normal_exit_first':first_after_normal,
                'before_sigkill_last':before_crash,'after_sigkill_first':first_after_crash,
                'receiver_reset_count':0,'accepted_live_messages':len(samples),'live_rejections':rejected,
                'duplicate_publisher_rejected':True,'topic_remapping_keeps_stream_identity':True,
                'replayed_sequence_800_rejection':replay_reason,'robot_connected':False}
            (output/'ros_restart_validation.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
            return 0
        finally:
            for child in children:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
            for log in logs: log.close()
            if camera: camera.close(); camera.destroy_node()
            if receiver: receiver.destroy_node()
            if rclpy.ok(): rclpy.shutdown()


if __name__=='__main__': raise SystemExit(main())
