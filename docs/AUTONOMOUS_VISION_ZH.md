# 饮料抓取的全自动多实例视觉

## 当前实现目标

`examples/d435i_autonomous_vision.py` 是面向机器人部署的新入口。它与原来的单目标调试脚本分开，当前完成：

- 根据 `--target-class` 只搜索指令指定的饮料类型。
- 保留画面中该类型的全部实例，不再只取最高置信度的一瓶。
- 每瓶建立独立的 RGB-D 卡尔曼轨迹和持久 Track ID。
- 在所有深度有效、已确认的候选中选择相机距离最近者。
- 选中后锁定 Track ID，避免距离测量抖动导致抓取目标切换。
- 目标丢失后自动等待重现、稳定和重新注册，不要求按 `R`。
- FoundationPose 使用有界后台任务；注册期间继续读取和显示实时相机画面，不积压旧注册请求。
- 注册结果返回时检查 Track ID、位置变化、二维框 IoU 和位姿质量，过期结果直接丢弃。
- 注册失败采用退避重试；同一候选连续失败后暂时排除并选择下一瓶。
- 每秒10次原子更新机器人可读取的 `latest_state.json`。

当前“最近”指相机光心到目标三维中心的欧氏距离。加入手眼标定后，可把排序距离替换成机器人基座坐标距离，Track ID 和视觉主体不需要重写。

## 运行

七喜：

```bash
conda activate fopose
cd ~/YOLOv8

python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --record-preview \
  --trust-checkpoints
```

类别：

```text
0 = fenda
1 = cola
2 = qixi
```

机器人无显示运行：

```bash
python examples/d435i_autonomous_vision.py \
  --target-class 2 \
  --mask-source seg \
  --conf 0.15 \
  --no-display \
  --trust-checkpoints
```

生产流程不需要键盘输入。`R` 只用于人工调试时强制清除当前位姿并启动自动恢复；`Q`/`Esc`退出，`V`切换预览录像。

`--record-preview` 的视频与逐帧JSON保存在：

```text
runs/autonomous_vision/recordings/
```

## 自动状态

```text
searching
  ↓ 检测到深度有效的同类实例
stabilizing
  ↓ 连续稳定1秒
registering_recovery
  ↓ 后台注册和质量检查通过
pose_valid
  ↓ 短暂质量下降
pose_degraded
  ↓ 连续15帧无效
reacquiring
  ↓ 原Track ID重现并稳定
registering_recovery
```

目标移动后，程序继续用 IMU 直立轴和 KF3D 更新位姿；稳定后使用 `motion-refresh` 在后台刷新FoundationPose结果。默认注册冷却5秒，不允许形成逐帧注册风暴。

后台注册时界面显示：

```text
BACKGROUND REGISTRATION - live preview, YOLO temporarily paused
```

此时相机画面继续更新，YOLO暂停约1～2秒，已有轨迹和位姿保持，不会积压已经过时的检测/注册任务。

## 多实例和最近目标

界面中的每瓶目标显示：

```text
ID=3 OK   0.72m 0.95
ID=7 LOCK 0.58m 0.93
```

- `NEW`：刚出现，尚未连续确认。
- `OK`：已确认候选。
- `LOCK`：当前最近且已锁定的抓取目标。
- 距离后面的数值是YOLO置信度。

目标锁定后不会因为另一瓶短暂变近而切换。只有选中轨迹超过丢失宽限、被移除或连续注册失败并暂时排除时，才会选择下一瓶。

## 机器人视觉输出

默认文件：

```text
runs/autonomous_vision/latest_state.json
```

写入采用临时文件加原子替换，机器人不会读到只写了一半的JSON。核心字段：

```json
{
  "requested_class_name": "qixi",
  "selected_track_id": 7,
  "selected_target": {
    "track_id": 7,
    "position_camera_m": [0.10, -0.02, 0.65],
    "position_std_m": [0.004, 0.004, 0.009],
    "distance_camera_m": 0.658,
    "depth_valid_ratio": 0.72,
    "track_status": "tracking"
  },
  "status": "pose_valid",
  "pose_camera_object": [[1, 0, 0, 0.10], [0, 1, 0, -0.02], [0, 0, 1, 0.65], [0, 0, 0, 1]],
  "yaw_observable": false,
  "pose_quality": {},
  "tracks": []
}
```

机器人只能在以下条件全部满足时使用位姿：

```text
status == "pose_valid"
selected_track_id != null
pose_camera_object != null
selected_target.track_status == "tracking"
pose_quality.valid == true
```

`yaw_observable=false` 表示裸CAD与圆柱几何无法证明绕瓶身轴的真实朝向。对于不要求标签朝向的侧抓或顶抓，应按旋转对称目标处理。

## 主要自动化参数

```text
--registration-stable-seconds 1.0
--registration-cooldown-seconds 5.0
--registration-retry-base-seconds 1.0
--registration-max-attempts 3
--failed-track-exclusion-seconds 8.0
```

不建议首先通过放宽位姿IoU来解决丢失。`IoU=0.17、center-error=0.47`表示CAD框确实严重偏离，正确行为是自动恢复或换下一个候选，而不是把错误位姿交给机器人。

离线测试入口：

```bash
python examples/validate_autonomous_existing_data.py \
  --target-class 2 \
  --start 240 \
  --end 280 \
  --trust-checkpoints
```

## 现有数据验证结果

现有连续 RGB-D 数据 `frame_000240`～`frame_000280` 的实例计数烟雾测试：

| 类别 | 帧数 | 标注实例 | 预测实例 | 每帧实例数完全一致 | 预测中深度可用 |
|---|---:|---:|---:|---:|---:|
| fenda | 41 | 67 | 66 | 82.9% | 87.9% |
| cola | 41 | 70 | 71 | 97.6% | 85.9% |
| qixi | 41 | 56 | 63 | 82.9% | 88.9% |

七喜序列中检测顺序变化时，最近目标仍持续锁定同一Track ID。芬达/七喜仍存在多检、漏检或标注差异，因此这些结果只能证明多实例逻辑能够工作，不能证明货架部署精度已经达标。

现有训练结果最后一轮在当前随机验证集上约为：框Precision 0.945、框Recall 0.938、Mask Precision 0.939、Mask Recall 0.951、Mask mAP50 0.967。由于相邻视频帧可能同时进入训练和验证，这些指标可能偏乐观。

## 现在是否需要重新采集数据

代码开发和第一轮功能验证不需要重新采集。现有数据已经包含多瓶同类实例，可以验证：

- 全部实例输出；
- Track ID；
- 深度测量；
- 最近目标选择；
- 自动恢复接口。

完成第一轮实机测试后，再定向补采一个完全独立的货架测试集。不要从同一段视频随机抽帧到训练和测试两边。建议每个场景连续录制，但按整段采集批次划分：

- 不同货架层高和背景；
- 每类1、2、3、5瓶的组合；
- 同类紧邻、部分遮挡、前后排；
- 标签正面、侧面、背面；
- 0.35～1.0米工作距离；
- 正常光、背光、反光；
- 其他相似瓶作为困难负样本；
- 固定相机和最终机器人相机安装视角。

采集时保留同步RGB、原始深度、相机内参和时间戳。优先补采程序实际失败的场景，不要只增加与现有数据相似的清晰正面图片。

## 是否需要提取纹理

当前不需要。若夹爪从瓶身侧面或顶部抓取，圆柱绕自身轴的yaw通常不影响抓取，裸CAD配合“中心＋竖直轴＋yaw任意”的对称位姿更稳定。

只有以下需求出现时才做纹理/标签朝向：

- 必须让饮料正面标签朝向指定方向；
- 灵巧手抓取点依赖瓶身某个非对称细节；
- 需要区分几何完全相同但标签不同的实例朝向。

这时仅给OBJ随意贴一张图不够，需要准确UV、与实物一致的纹理，并用标签关键点或纹理感知姿态方法验证yaw。它不阻塞当前多实例自动视觉开发。

## 当前边界

- 当前距离是相机距离，不是机器人基座距离。
- 这是单条类别指令对应同类多实例；不同类别同时跟踪可在后续命令接口中扩展。
- 注册后台线程已经用真实保存的RGB-D数据验证，FoundationPose耗时约1.79秒期间主线程可继续运行。
- D435i对透明七喜瓶的深度不是每帧可靠；深度质量不足的候选不会被选择为可抓取目标。
- 最终“抓取成功”需要机器人状态、夹爪反馈和视觉确认共同决定，不应只由视觉框判断。
