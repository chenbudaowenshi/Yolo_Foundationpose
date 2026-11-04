# ObjectPose 进程重启后的序号恢复

生产入口 `sb_vision_node.py` 使用持久化序号，不依赖退出时保存，也不要求接收端在每次视觉重启时reset。`ObjectPose.msg` 和接收端的乱序/时间戳校验保持不变。

## 初次启用与旧程序迁移

旧版本没有持久化记录，新的视觉进程不能凭空知道接收端已收到的最大序号。因此首次启用需要明确初始化下界，缺少它会报错并停止启动，不会默认发1：

```bash
# 示例：接收端已经接受到800；相机和视觉都在本机
bash my_demo/scripts/run_sb_vision.sh --clock-sync-mode local \
  --image-transport compressed --sequence-start-after 800
```

这个进程第一次分配的是801。对于确定没有旧记录的全新接收端，用 `--sequence-start-after 0`。填写的值必须不小于实际接收端最后接受的值；如果无法确定，应先与控制侧协调一次接收状态重置，再初始化新流。

以后正常退出、异常崩溃或被kill后，使用相同状态目录重新启动即可，不再需要初始化参数：

```bash
bash my_demo/scripts/run_sb_vision.sh --clock-sync-mode local --image-transport compressed
```

重复带上旧的 `--sequence-start-after` 不会降低已保存的序号上界。该参数只能初始化新流或提高下界，不能修复损坏/丢失的历史记录。

## 持久化方式

- 默认目录：`/home/robot/FoundationPose/.state/sb_vision/`。
- 可用 `--sequence-state-dir` 或环境变量 `SB_VISION_STATE_DIR` 指定固定目录。代码版本切换或移动checkout时，必须继续使用同一目录。
- 按**实际 ROS Domain ID＋解析/remap后的话题名**区分流。两个不同请求话题若remap到同一输出话题，会使用同一个序号状态。
- JSON保存版本、流身份、已预留序号上界与校验摘要。写入顺序为临时文件→文件fsync→原子replace→目录fsync。
- 每次先持久化预留4096个序号，之后在内存中分配；不逐帧写盘。范围耗尽时先提交下一个范围，再发新序号。
- 重启时跳过上次预留但可能尚未使用的部分。例如从800初始化，预留到4896，只发到810就崩溃，重启后从4897开始。**跳号是正常行为，接收端只要求严格递增。**
- 同一状态目录中的同一流使用生命周期独占文件锁，阻止两个进程交错发布。锁文件保持固定inode，不随JSON替换，也不在正常退出时删除。
- 锁中的初始化标记可发现JSON状态被单独删除的情况。记录损坏、校验失败、写入/fsync失败、状态消失时停止发布，不自动回到0。
- uint32耗尽时停止，不回绕。此时必须与接收端协调重新初始化流。

持久化依赖支持flock、原子rename和文件/目录fsync的本地文件系统。不要把生产状态放到 `/tmp`、自动清理的日志目录、网络文件系统或代码发布包内；不要删除锁文件、回滚旧备份或让同一话题的两个发布者使用不同状态目录。丢失整个目录后无法仅凭发送端恢复接收端历史，此时需要显式协调接收端。

## 与既有逻辑的关系

重注册、目标reset和时钟同步恢复继续使用同一分配器，均不重置序号。原始采集时间戳仍然透传；本功能不会放宽时间戳倒退、超时、跳变等检查。

偶发的号段持久化可能受磁盘延迟影响，发布前会再次检查帧年龄。若已过期，该序号可以跳过，旧位姿不会因为已经分配了序号而被发布。

`/vision/status.sequence_state`包含状态文件、流身份、当前分配值、预留上界、号段大小及失败状态。磁盘上记录的是预留上界，不是接收端实际收到的最后一帧。

## 验证

```bash
PYTHONPYCACHEPREFIX=/tmp/sb_sequence_pycache \
  /home/robot/anaconda3/envs/pose/bin/python -m unittest discover -s my_demo/tests -p test_sequence_store.py -v

source my_demo/ros2_handoff_env.sh
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_ros_sequence_restart.py
```

ROS测试使用独立的回环Domain 75、私有测试话题、合成相机/位姿后端及临时状态目录。接收端初始化为800后一直运行；发布端实际执行正常退出、SIGKILL、重启，还验证topic remap及重复发布进程的拒绝。不会使用真实相机或连接机器人。报告存入 `runs/sb_vision/sequence_restart_20260919/`。

原有合成ROS测试各用独立临时状态目录，不污染生产状态。`test_live_vision.py`使用真实视觉流，保留生产默认持久化目录，并接受相同的 `--sequence-state-dir` / `--sequence-start-after` 参数；首次运行时也需根据接收端状态初始化。

本轮结果：83项单元/回归测试通过，其中15项专门验证持久化。独立ROS进程测试中，接收端从800开始，先收到801，正常重启后收到4897，强制终止后再启动收到8993；接收端未reset，58条有效观测全部接受。重复发布者被拒绝，旧序号800的重放仍被判out_of_order。压缩链路和时钟失步恢复回归也通过。详见上述报告目录。
