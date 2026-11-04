#!/usr/bin/env python3
"""接口验收：对着视觉节点正在发的话题跑一遍，全 PASS 才算可以交付。

    # 瓶子放桌上不动，机器人不动
    python3 vision_interface_acceptance.py --static --duration 60

    # 手动把瓶子慢慢挪 20-30 cm
    python3 vision_interface_acceptance.py --moving --duration 30

Why this script and not "试一下看看": every gate below already exists in `TargetManager`, which is what the
robot actually runs.  Re-implementing those gates here would create a second definition of "acceptable" that
drifts from the first, so section 3 does the honest thing and REPLAYS the captured stream through the real
`TargetValidator` with production defaults.  If this script says the stream is clean, the robot will agree,
because it is the same code.

The rest of the script measures the things the validator cannot tell you: how noisy the source is, and how
far the yaw wanders.  Those two numbers are what `filter_tau` and the jump gate get tuned from, and they are
the reason the static capture is worth 60 seconds of somebody's time.

结论：退出码 0 = 可以交付。非 0 = 交付前先修，别开始联调。
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

# Find `shuangbi_grasp` next to this script's directory.  That one line covers both layouts this file has to
# live in: the repo (`ws/src/shuangbi_grasp/scripts/`) and the hand-off bundle given to the vision side, which
# ships only the three pure-numpy modules the replay needs.  The alternative was telling the other side to set
# PYTHONPATH, which silently REPLACES the ROS entries and loses rclpy - a five-minute detour for no reason.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import rclpy
from rclpy.node import Node
from sb_communication.msg import ObjectPose

from shuangbi_grasp.contracts import (
    DEFAULT_JUMP_ORIENTATION_RAD, DEFAULT_JUMP_POSITION_M, DEFAULT_MAX_OBSERVATION_DELAY_S, ObjectPoseSample,
)
from shuangbi_grasp.target_manager import TargetValidator, ValidatorConfig, quaternion_angle, bottle_axis_angle

#: 抓取侧拒绝倾斜超过这个角度的位姿（grasp_profile_bottle.yaml 的 max_tilt_deg）。
MAX_TILT_RAD = math.radians(15.0)
#: 静止拍摄下允许的逐轴位置标准差。头部 D435i 在 0.6-0.8 m 的深度噪声本来就有 1-2 cm，
#: 这里卡 10 mm 是"源本身别再额外抖"，不是相机精度指标。
STATIC_SIGMA_LIMIT = 0.010


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """(w,x,y,z) -> 3x3。与 shuangbi_control.model.quat_to_matrix 同定义，此处内联以便本脚本可独立运行。"""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("零四元数")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bottle_axis(rotation: np.ndarray) -> np.ndarray:
    """抓取侧怎么读姿态，这里就怎么读：只取第三列，且朝上翻正（grasp_pose_generator.py）。"""
    axis = rotation[:, 2]
    axis = axis / np.linalg.norm(axis)
    return -axis if axis[2] < 0.0 else axis


def yaw_about_axis(rotation: np.ndarray, axis: np.ndarray) -> float:
    """绕瓶轴的转角 [rad]，相对一个与瓶轴无关的固定基准。

    抓取侧不用这个值（圆柱的 yaw 不可观测，我们自己采样），但**跳变门用**：FoundationPose 对旋转对称体
    的 yaw 常常是随机游走的，逐帧乱跳会让整条流被判成跳变而全部丢弃。所以这里把它量出来。
    """
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    x_dir = rotation[:, 0] - np.dot(rotation[:, 0], axis) * axis
    if np.linalg.norm(x_dir) < 1e-9:
        return 0.0
    return math.atan2(float(np.dot(x_dir, v)), float(np.dot(x_dir, u)))


class Collector(Node):
    """默认 QoS 订阅——和 target_manager 一模一样，所以 QoS 不兼容会表现为"一条都收不到"，正是我们想测的。"""

    def __init__(self, topic: str) -> None:
        super().__init__("vision_interface_acceptance")
        self.samples: list[dict] = []
        self.malformed = 0
        self.create_subscription(ObjectPose, topic, self._on_msg, 10)

    def _on_msg(self, msg: ObjectPose) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        try:
            rotation = quat_to_matrix(msg.pose.orientation.w, msg.pose.orientation.x,
                                      msg.pose.orientation.y, msg.pose.orientation.z)
        except ValueError:
            self.malformed += 1
            return
        quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                         msg.pose.orientation.y, msg.pose.orientation.z])
        self.samples.append({
            "received": time.monotonic(), "stamp": stamp,
            "delay": (now_ros - stamp) if stamp > 0.0 else None,
            "seq": int(msg.sequence), "frame_id": msg.header.frame_id, "object_id": msg.object_id,
            "confidence": float(msg.confidence), "source": msg.source,
            "position": np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            "quat": quat, "quat_norm": float(np.linalg.norm(quat)),
            "covariance": np.asarray(msg.covariance, float).reshape(6, 6), "rotation": rotation})


def replay_through_validator(samples: list[dict]) -> tuple[int, dict, int]:
    """把抓到的流喂给生产环境那个校验器。返回 (接受数, 各拒绝理由计数, 构造失败数)。"""
    validator = TargetValidator(ValidatorConfig())
    accepted, malformed = 0, 0
    for s in samples:
        try:
            sample = ObjectPoseSample(
                object_id=s["object_id"] or "", frame_id=s["frame_id"] or "",
                position=s["position"], orientation=s["quat"], stamp=s["stamp"], received=s["received"],
                observation_delay=s["delay"], confidence=s["confidence"], covariance=s["covariance"],
                sequence=s["seq"])
        except ValueError:
            malformed += 1
            continue
        if validator.check(sample, s["received"]) is None:
            validator.accept(sample)
            accepted += 1
    return accepted, dict(validator.counts), malformed


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default="object_pose")
    parser.add_argument("--duration", type=float, default=60.0, help="采集时长 [s]")
    parser.add_argument("--min-hz", type=float, default=10.0, help="要求的最低发布频率")
    parser.add_argument("--static", action="store_true", help="瓶子静止：噪声项参与判定")
    parser.add_argument("--moving", action="store_true", help="瓶子在动：噪声项只报告不判定")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="丢弃开头这几秒（CUDA 首次 JIT、FoundationPose 首帧 register 天然慢，"
                             "拿它们判稳态是不公平的）")
    args = parser.parse_args()
    static = args.static and not args.moving

    rclpy.init()
    node = Collector(args.topic)
    print(f"采集 {args.topic} …… {args.duration:.0f} s（{'静止' if static else '移动/未声明'}），"
          f"丢弃前 {args.warmup:.0f} s 暖机")
    end = time.monotonic() + args.duration + args.warmup
    try:
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    raw, malformed = node.samples, node.malformed
    node.destroy_node()
    rclpy.shutdown()
    if raw and args.warmup > 0.0:
        cutoff = raw[0]["received"] + args.warmup
        samples = [s for s in raw if s["received"] >= cutoff] or raw
        print(f"  暖机丢弃 {len(raw) - len(samples)} 条，稳态样本 {len(samples)} 条")
    else:
        samples = raw

    checks: list[tuple[str, bool, str]] = []
    n = len(samples)

    # ---- 1. 收到了吗（QoS / 话题名） --------------------------------------------------------------
    checks.append(("收到消息（QoS 兼容、话题名对）", n > 0,
                   f"{n} 条" if n else "0 条 —— 话题名错，或发布端用了 BEST_EFFORT（我这边是 RELIABLE，不兼容且不报错）"))
    if n < 2:
        return report(checks, {})

    # ---- 2. 频率 ---------------------------------------------------------------------------------
    span = samples[-1]["received"] - samples[0]["received"]
    hz = (n - 1) / span if span > 0 else 0.0
    gaps = [samples[i + 1]["received"] - samples[i]["received"] for i in range(n - 1)]
    worst_gap = max(gaps)
    checks.append((f"频率 ≥ {args.min_hz:.0f} Hz", hz >= args.min_hz, f"实测 {hz:.1f} Hz"))
    # 0.25 s 是 max_age：任何一个比它长的空档，机器人那一刻就是"没有目标"。
    checks.append(("最大空档 < 0.25 s（max_age）", worst_gap < 0.25, f"最大 {worst_gap * 1000:.0f} ms"))

    # ---- 3. 时间戳语义 ---------------------------------------------------------------------------
    delays = [s["delay"] for s in samples if s["delay"] is not None]
    if not delays:
        checks.append(("header.stamp 已填", False, "全是 0 —— 必须填拍照时刻"))
    else:
        p95 = percentile(delays, 0.95)
        median = statistics.median(delays)
        negative = sum(1 for d in delays if d < -0.005)
        checks.append((f"观测延迟 p95 ≤ {DEFAULT_MAX_OBSERVATION_DELAY_S:.2f} s",
                       p95 <= DEFAULT_MAX_OBSERVATION_DELAY_S,
                       f"中位 {median * 1000:.0f} ms / p95 {p95 * 1000:.0f} ms / 最大 {max(delays) * 1000:.0f} ms"))
        # 负延迟只有两种可能：填的是未来时刻，或两台机器时钟没同步。两种都会让新鲜度判据失效。
        checks.append(("观测延迟非负（时钟同步）", negative == 0,
                       "全部非负" if negative == 0 else f"{negative} 条为负 —— 时钟没同步，或填的是发布时刻"))

    stamps_ok = all(s["stamp"] > 0 for s in samples) and all(samples[i + 1]["stamp"] > samples[i]["stamp"] for i in range(n - 1))
    checks.append(("header.stamp 有效且每帧更新", stamps_ok, "严格递增" if stamps_ok else "零时间、重复图像或时间倒退"))

    # ---- 4. 字段 ---------------------------------------------------------------------------------
    seq_ok = all(samples[i + 1]["seq"] > samples[i]["seq"] for i in range(n - 1))
    checks.append(("sequence 严格递增", seq_ok,
                   f"{samples[0]['seq']} → {samples[-1]['seq']}" if seq_ok else "有重复或回退 —— 会被判 out_of_order"))

    frames = {s["frame_id"] for s in samples}
    checks.append(("frame_id 非空且唯一", len(frames) == 1 and "" not in frames,
                   f"{frames}" + ("" if "" not in frames else "  空 frame_id 会被当成「已在规划系」，位姿全错")))

    ids = {s["object_id"] for s in samples}
    checks.append(("object_id 非空且唯一", len(ids) == 1 and "" not in ids, f"{ids}"))

    confs = [s["confidence"] for s in samples]
    checks.append(("confidence ∈ [0,1] 且 ≥ 0.2", all(0.2 <= c <= 1.0 for c in confs),
                   f"min {min(confs):.2f} / 中位 {statistics.median(confs):.2f} / max {max(confs):.2f}"))

    norms = [s["quat_norm"] for s in samples]
    checks.append(("四元数已归一化", all(abs(v - 1.0) < 1e-3 for v in norms),
                   f"模长 {min(norms):.4f} … {max(norms):.4f}"))
    checks.append(("没有畸形消息", malformed == 0, f"{malformed} 条构造失败"))

    # ---- 5. 几何：瓶轴与倾角 ---------------------------------------------------------------------
    axes = [bottle_axis(s["rotation"]) for s in samples]
    tilts = [math.acos(float(np.clip(abs(np.dot(a, [0, 0, 1])), -1, 1))) for a in axes]
    # 注意：frame_id 若是相机光学系，"竖直"在那个系里不是 z，这一项只在位姿已在机器人系里时才有意义。
    tilt_meaningful = any(k in (list(frames)[0] or "").lower() for k in ("base", "waist", "link")) \
        and "optical" not in (list(frames)[0] or "").lower()
    if tilt_meaningful:
        checks.append(("瓶轴倾角 < 15°（抓取侧的硬门限）", max(tilts) <= MAX_TILT_RAD,
                       f"最大 {math.degrees(max(tilts)):.1f}°"))

    # ---- 6. 噪声与跳变（决定 filter_tau 和跳变门怎么设） -------------------------------------------
    stack = np.array([s["position"] for s in samples])
    sigma = stack.std(axis=0)
    steps = [float(np.linalg.norm(stack[i + 1] - stack[i])) for i in range(n - 1)]
    quat_steps = [quaternion_angle(samples[i]["quat"], samples[i + 1]["quat"]) for i in range(n - 1)]
    yaws = np.unwrap([yaw_about_axis(s["rotation"], a) for s, a in zip(samples, axes)])
    yaw_steps = [abs(float(yaws[i + 1] - yaws[i])) for i in range(n - 1)]
    over_jump_pos = sum(1 for v in steps if v > DEFAULT_JUMP_POSITION_M)
    axis_steps = [bottle_axis_angle(samples[i]["quat"], samples[i + 1]["quat"]) for i in range(n - 1)]
    over_jump_ori = sum(1 for v in axis_steps if v > DEFAULT_JUMP_ORIENTATION_RAD)

    if static:
        checks.append((f"静止位置标准差 < {STATIC_SIGMA_LIMIT * 1000:.0f} mm（逐轴）",
                       bool(np.all(sigma < STATIC_SIGMA_LIMIT)),
                       f"σ = [{sigma[0] * 1000:.1f}, {sigma[1] * 1000:.1f}, {sigma[2] * 1000:.1f}] mm"))
        checks.append(("静止时逐帧位置跳变 < 50 mm（跳变门）", over_jump_pos == 0,
                       f"最大 {max(steps) * 1000:.1f} mm，超门限 {over_jump_pos} 帧"))

    # ---- 7. 回放：用生产环境的校验器判 ------------------------------------------------------------
    accepted, counts, replay_bad = replay_through_validator(samples)
    rate = accepted / n if n else 0.0
    # 95% 而不是 100%：偶发的一两帧超时是任何流水线都有的，机器人那一刻只是保持上一个目标，没有危害。
    # 持续性的问题会被上面的「最大空档」和「p95 延迟」抓住，不靠这一项。
    checks.append(("生产校验器接受率 ≥ 95%", rate >= 0.95,
                   f"{accepted}/{n} = {rate * 100:.1f}%" + (f"  拒绝: {counts}" if counts else "")))

    stats = {"n": n, "hz": hz, "sigma": sigma, "steps": steps, "quat_steps": quat_steps,
             "yaw_steps": yaw_steps, "yaws": yaws, "delays": delays, "counts": counts,
             "axis_steps": axis_steps, "over_jump_ori": over_jump_ori, "replay_bad": replay_bad, "static": static,
             "tilts": tilts, "tilt_meaningful": tilt_meaningful}
    return report(checks, stats)


def report(checks: list[tuple[str, bool, str]], stats: dict) -> int:
    print("\n" + "=" * 78)
    print("硬性验收项")
    print("=" * 78)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        print(f"        └─ {detail}")

    if stats:
        print("\n" + "=" * 78)
        print("调参用的测量值（不判定，但请连同结果一起发给我）")
        print("=" * 78)
        s = stats["sigma"]
        print(f"  样本数 / 频率        {stats['n']} 条 / {stats['hz']:.1f} Hz")
        print(f"  位置标准差 σ         [{s[0] * 1000:.2f}, {s[1] * 1000:.2f}, {s[2] * 1000:.2f}] mm")
        print(f"  逐帧位置变化         中位 {statistics.median(stats['steps']) * 1000:.2f} mm，"
              f"最大 {max(stats['steps']) * 1000:.2f} mm（跳变门 50 mm）")
        print(f"  逐帧姿态变化(四元数)  中位 {math.degrees(statistics.median(stats['quat_steps'])):.2f}°，"
              f"最大 {math.degrees(max(stats['quat_steps'])):.2f}°（完整四元数仅诊断）")
        print(f"  逐帧绕瓶轴 yaw 变化   中位 {math.degrees(statistics.median(stats['yaw_steps'])):.2f}°，"
              f"最大 {math.degrees(max(stats['yaw_steps'])):.2f}°")
        print(f"  yaw 总漂移            {math.degrees(float(stats['yaws'].max() - stats['yaws'].min())):.1f}°")
        if stats["tilt_meaningful"]:
            print(f"  瓶轴倾角             最大 {math.degrees(max(stats['tilts'])):.1f}°（门限 15°）")
        else:
            print("  瓶轴倾角             未判定（位姿在相机光学系里，「竖直」要变换后才有意义）")

        print(f"  瓶轴逐帧最大变化      {math.degrees(max(stats['axis_steps'])):.2f}°（门限 30°，忽略yaw及轴正负）")
        if stats["over_jump_ori"] > 0:
            print(f"  瓶轴跳变超门限帧数    {stats['over_jump_ori']}；检查实测瓶轴和目标关联")

    failed = [name for name, ok, _ in checks if not ok]
    print("\n" + "-" * 78)
    if not failed:
        print("  结论：全部通过，可以交付联调。")
        return 0
    print(f"  结论：未通过 {len(failed)} 项，交付前先修 ——")
    for name in failed:
        print(f"         · {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
