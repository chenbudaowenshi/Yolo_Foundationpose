#!/usr/bin/env python3
"""Real camera/GPU regression with explicitly injected detector gaps.

Does not move the camera or robot. The original selector/tracker run unchanged.
Only this harness masks worker.busy and detector output for timed test phases.
"""
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

os.environ["D435I_FOPOSE_REEXEC"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_demo"))
import d435i_autonomous_vision as app

output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
first_valid = None
states = []


def phase():
    if first_valid is None:
        return "initializing"
    elapsed = time.perf_counter() - first_valid
    if 2 <= elapsed < 2.3:
        return "short_detection_gap"
    if 4 <= elapsed < 5.2:
        return "detector_paused"
    if 7 <= elapsed < 9.5:
        return "long_detection_gap"
    return "recovery" if elapsed >= 9.5 else "normal"


original_find = app.base.find_targets


def injected_find(*args, **kwargs):
    if phase() in {"short_detection_gap", "long_detection_gap"}:
        return [], None
    return original_find(*args, **kwargs)


class AuditWorker(app.LatestPoseRegistrationWorker):
    @property
    def busy(self):
        return phase() == "detector_paused" or super().busy


class AuditPublisher(app.VisionStatePublisher):
    def publish(self, state, now_s, *, force=False):
        global first_valid
        if first_valid is None and state.get("pose_valid"):
            first_valid = time.perf_counter()
        state = dict(state, test_phase=phase(),
                     test_elapsed_s=None if first_valid is None else time.perf_counter() - first_valid)
        states.append(state)
        super().publish(state, now_s, force=force)


app.base.find_targets = injected_find
app.LatestPoseRegistrationWorker = AuditWorker
app.VisionStatePublisher = AuditPublisher
app.read_key_from_stdin = lambda *args: ord("q") if first_valid is not None and time.perf_counter() - first_valid >= 17 else 255
parser = app.build_parser()
args = parser.parse_args([
    "--trust-checkpoints", "--serial", "327122073383", "--no-display", "--record-preview",
    "--record-dir", str(output / "recordings"), "--autonomous-output-dir", str(output),
    "--vision-state-file", str(output / "latest_state.json"), "--max-runtime-seconds", "60",
])
app.validate_args(args, parser)
code = app.run(args)
(output / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))

failures = []
def check(condition, message):
    if not condition:
        failures.append(message)

valid = [s for s in states if s.get("pose_valid")]
check(code == 0 and len(valid) > 10, "No sufficient valid measured output")
for state in valid:
    check(state["selected_target"]["observed_this_frame"] and state["detection_ran"], "Valid pose without current detection")
    check(state["observation_state"] == "measured" and 0 <= state["observation_age_s"] <= .5, "Valid pose with stale observation")
    check(state["pose_quality"]["depth_surface"]["valid"], "Valid pose without independent surface check")
# Phase boundaries are sampled at publication, after detection. Exclude the
# transition frame that genuinely detected just before a timed injection began.
# The global valid-output assertions above still check every transition frame.
paused = [s for s in states if s.get("test_elapsed_s") is not None and 4.1 <= s["test_elapsed_s"] <= 5.1]
check(any((s.get("observation_age_s") or 0) > .5 for s in paused), "Did not exercise stale observation during pause")
check(all(not s.get("pose_valid") for s in paused), "Accepted a cached pose during detector pause")
check(all(not (s.get("selected_target") or {}).get("observed_this_frame") for s in paused), "Cached detection still marked current")
check(len({s.get("observation_stamp_ns") for s in paused}) == 1, "Capture stamp changed without detection")
short = [s for s in states if s.get("test_elapsed_s") is not None and 2.1 <= s["test_elapsed_s"] <= 2.28]
check(any(s.get("status") == "pose_predicted" and s.get("pose_camera_object") is not None for s in short), "Short-term prediction not preserved")
check(all(not s.get("pose_valid") for s in short), "Gap emitted a valid pose")
check(any(s.get("test_phase") == "long_detection_gap" and s.get("selected_track_id") is None for s in states), "Original lost-target expiration did not run")
check(any(s.get("test_phase") == "recovery" and s.get("pose_valid") for s in states), "Did not recover a valid pose")
for before, after in zip(states, states[1:]):
    if before.get("selected_track_id") != after.get("selected_track_id"):
        check(after.get("registration_age_s") is None, "Registration age inherited across target change")
        check(after.get("pose_camera_object") is None, "Previous target pose inherited")
last = json.loads((output / "latest_state.json").read_text())
check(last["status"] == "stopped" and last["pose_camera_object"] is None, "Shutdown retained a valid pose")
report = dict(mode="real RGB-D and GPU, software-injected detection gaps and busy pause; no physical occlusion claim",
              state_count=len(states), valid_count=len(valid), paused_frames=len(paused),
              phases=dict(Counter(s.get("test_phase") for s in states)),
              statuses=dict(Counter(s.get("status") for s in states)),
              failures=sorted(set(failures)))
(output / "regression_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(bool(failures))
