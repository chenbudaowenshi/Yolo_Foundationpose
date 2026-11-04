# 首次注册、重注册、跟踪与显存统计

此统计针对 `sb_vision_pipeline.py` / `sb_vision_node.py` 的真实 FoundationPose 测量流水线，输出仍是原有 `sb_communication/ObjectPose`。
没有合并或改动首次/恢复时的搜索规则；统一的是事件结构与统计口径。

## 事件分类

| `mode` / `event_type` | 定义 |
|---|---|
| `initial_registration` | 当前进程第一次调用 register，包括最终质量未通过的尝试 |
| `reregistration` | 当前进程后续任意 register：重试、恢复、重置、换目标等 |
| `tracking` | 实际调用 FoundationPose track_one |
| `none` | 只有检测、无匹配目标等没有执行 FP 位姿操作的帧 |

`operation=register/track` 以及 `timings.register_s/track_s` 保留原始操作名称供兼容；新统计按上述明确类型分组。
`event_id`、`registration_attempt` 在进程内递增，目标重置不清零；`session_id` 区分进程，`target_generation` 区分目标选择代次。

`trigger_reason` 包括 `startup`、`manual_reset`、`target_not_observed`、`target_reselected`、`frame_gap`、`registration_quality_rejected`、`tracking_quality_failure`、`clock_unavailable`、`clock_recovered` 等。跟踪为 `continuous_tracking`。

每个注册事件还记录：

- `full_search_requested`：实际传给注册后端的 force_full 标记。
- `search.search_executed`：是否真正进入旋转搜索，深度不足的提前返回不算搜索完成。
- `search.full_search`：是否运行保守密集网格、全候选 refinement。
- `search.rotation_candidates / requested_topk / refined_candidates`：实际候选数量和裁剪范围。
- `estimate_valid`：计算及几何校验是否得到有效位姿。
- `published`、`sequence`、`published_mono_ns`：是否通过时效/序号门控并实际调用发布。

计算有效但已经过期、处于旧时钟代次或 reset 之前的在途结果，不计为有效发布。异常、失败与退出时完成但丢弃的任务也保留耗时记录。

## 时间统计口径

所有阶段耗时用单调时钟，原始相机采集时间不改写。

- `model_load_s`：后端首次构造、模型及 CUDA 资源初始化。
- `yolo_s`：当前帧检测。
- `pose_s`：FP register 或 track 调用及返回；不把模型加载混入注册核算。
- `quality_s`：位姿转换与几何校验。
- `pipeline_s`：本帧检测至计算结束。
- `worker_s`：完整工作线程任务，首次任务包含模型加载。
- `observation_delay_s`：当前输出检查时刻减原始采集时间。

每类操作给出样本数、P50/P95/最大值、有效估计数、实际发布数、全量请求/执行数与触发原因计数；失败尝试也进入耗时统计。
汇总顶层另列模型加载和仅已发布观测的延迟；模型加载可能发生在没有检测到目标的帧，不会因此漏记。
分位数最多保留最近 100,000 个样本，超过时明确标为 `latest_bounded_samples`；总次数与最大值仍是整段运行统计。短时验收使用 `whole_run`。

## 恢复时间

一次恢复从首次检测到输出中断、收到 reset、时钟状态变化或输出超时开始，到下一次通过门控的 `ObjectPose.publish()` 完成为止。
重复失败不会重置起点；期间追加的原因写入 `triggers`。首次启动从节点统计初始化计时，单独给出 `time_to_first_valid_output_s`。
恢复记录同时给出首个/最后一个注册调用开始到有效发布的耗时；恢复可以只靠 tracking 完成，不要求每次中断都重新注册。
始终没有恢复时保留 `pending_recovery`，不填 0 秒，也不记为完成。
该终点是发布端实际调用成功，不代表机器人执行或网络下游已经验收。

## 显存口径

- **PyTorch allocator**：在加载模型前清零统计，随后整次会话不因重置目标而清零；记录已分配和已保留显存的峰值。只覆盖 PyTorch 分配器。
- **NVML 本进程采样**：默认每 50 ms 读取本 PID 的 GPU 使用，覆盖 CUDA 上下文及非 PyTorch 分配；不把其他进程的显存相加，同一 PID 同一 GPU 不重复计数。
- NVML 值明确是采样峰值，可能漏掉采样间隔内的短暂尖峰；不可说成精确瞬时峰值。不可用时记录原因，不用合成测试数据补数。
- 保存 GPU 名称、UUID、采样周期及输入来源。实机报告标为 `live_camera`，回放/合成记录不得混入实机统计。

## 文件与运行

默认每次启动在 `runs/sb_vision/metrics/` 创建会话 JSONL 和 `.summary.json`，汇总每秒原子更新，正常关闭保存 `final=true`。
JSONL 包含生命周期、reset、frame 等记录；读取帧数据时筛选 `record_type == "frame"`。`/vision/status.performance` 同时提供汇总。

```bash
# 同机测试；跨机依旧需要 required 时钟同步配置。
SB_ROS_DOMAIN_ID=0 bash my_demo/scripts/run_sb_vision.sh --clock-sync-mode local \
  --metrics-source live_camera --metrics-jsonl runs/my_measurement/events.jsonl
```

首次启用发布序号仍须按 [序号说明](README_sequence_state.md) 配置下界，统计不会绕过或清零已有序号。

独立的实机统计脚本只向私有测试 pose/status 话题发布，不发送运动命令：

```bash
source my_demo/ros2_handoff_env.sh
export ROS_DOMAIN_ID=0
python my_demo/scripts/benchmark_registration.py --output-dir runs/my_benchmark --duration 50 --target-class 0 --resets 3
```

相机已经运行时直接订阅；相机空闲时可用 `--start-camera`。测试脚本的 `--force-full-search` 只在该测量进程中强制既有保守搜索路径，使用相同真实图像与 FP 模型，并显式记录覆盖条件；生产默认搜索策略不变。
