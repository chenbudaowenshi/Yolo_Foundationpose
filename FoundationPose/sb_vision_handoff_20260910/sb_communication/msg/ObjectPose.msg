# 6D pose of a task object, from any source (manual test input, replay, camera).
# Frame 3.1 of the README: the ObjectPoseProvider -> TargetManager contract.

std_msgs/Header header          # stamp = when the pose was OBSERVED (not when it was sent); frame_id = its frame

string object_id                # stable id, e.g. "bottle"; TargetManager tracks one id at a time
geometry_msgs/Pose pose         # position [m] and orientation in header.frame_id

float64[36] covariance          # 6x6 row-major (x y z rx ry rz), all zeros = unknown
float32 confidence              # 0..1, 1 = fully trusted; a manual source publishes 1.0
builtin_interfaces/Time valid_until   # after this the pose must not be used; zero = use the freshness window

string source                   # "manual" | "replay" | "camera" ...; only for diagnostics
uint32 sequence                 # increases per publication; used to reject out-of-order / replayed messages
