# 视觉机时钟同步与本机验证

视觉节点会用本机系统时间检查远端图像采集时间，因此机器人、视觉机、控制机都必须对齐到同一授时源。系统服务负责校时，视觉代码负责确认同步状态并决定是否允许发布。`header.stamp`始终是原始采集时间；不会添加估计偏差或改成发布时刻。

当前按用户要求只搭建本机验证：没有连接机器人，没有修改系统时钟或现有NTP服务。本机原有`systemd-timesyncd`继续运行；正式三机授时源尚未指定，默认配置`config/clock_sync.json`的`server`为null。

## 运行模式

| 模式 | 用途 | 检查 |
|---|---|---|
| `required`（默认） | 后续跨机使用 | OS服务已同步、实际使用指定源、主动NTP测量达到阈值；未配置源则等待并停发 |
| `local` | 相机和视觉均在本机的测试 | 明确标记`same_host_only`；不查询NTP，不声称跨机时钟已对齐 |
| `test` | 本机可控偏差实验 | 仅允许回环IP及≥1024的端口，测量模拟授时源；不修改OS时钟 |

同机相机/视觉启动（序号状态需已初始化；首次启用见[序号恢复说明](README_sequence_state.md)）：

```bash
bash /home/robot/FoundationPose/my_demo/scripts/run_sb_vision.sh \
  --image-transport compressed --clock-sync-mode local
```

正式跨机时需要提供共同授时源，通过`--clock-server`或导出的`SB_NTP_SERVER`指定；默认`required`模式不会自动回退为`local`。

## 检查与停发行为

`clock_sync.py`通过chrony或systemd-timesyncd读取实际同步状态和选中的服务器。再向同一源发送NTP查询，检查应答来源、版本、模式、stratum、闰秒状态及请求时间戳匹配。

默认每5秒检查，查询3次并采用最低RTT样本。正offset表示授时源比本机快。接收/发送的四个时间按[NTP RFC 5905 §8](https://www.rfc-editor.org/rfc/rfc5905#section-8)计算偏差和往返时间。

默认要求`abs(offset) + RTT/2 ≤ 2 ms`，其中RTT/2用于保守计入路径不对称的不确定度；这不是仅检查`timedatectl`显示的yes。它是相对于指定源的测量界限，不能证明该源相对UTC的绝对准确度。共用同一稳定源时，两台客户端的相对误差预算可控制在约4ms，给现有5ms未来时间容差留出余量。WiFi排队可能让测量不能达标，应优化网络；不会自动放宽原有帧年龄门限。

- NTP源未配置、服务未同步、选错源、查询失败或偏差/路径不确定度超限：停止提交推理和发布ObjectPose，诊断中说明原因。
- 检查在后台子进程进行，含DNS在内的单轮检查最多8秒，不阻塞ROS图像回调；同步状态超过15秒未更新也不可用。
- 系统墙钟相对单调时钟发生超过5ms的跳变：清空待处理图像、使旧在途任务失效，并重新检查。采样调度延迟单独计入容差。
- 同步恢复后只使用新曝光帧，重新建立内部位姿初值。`sequence`保留原计数，不在同步恢复时清零。
- `/vision/status`新增`clock_sync`与`clock_epoch`。主要原因有`server_not_configured`、`system_service_not_synchronized`、`system_service_uses_different_source`、`clock_error_bound_exceeded`、`ntp_probe_failed`、`clock_status_stale`、`local_clock_jump`。

## 本机实验

独立测试授时源只监听127.0.0.1，不提供局域网授时服务：

```bash
# 终端A：模拟正常参考时钟；改为20或-400可模拟偏差
python3 my_demo/scripts/local_ntp_test_server.py --port 41234 --offset-ms 0
# 终端B：只读检查；通过退出0，不通过退出2
python3 my_demo/scripts/check_clock_sync.py \
  --clock-sync-mode test --clock-server 127.0.0.1 --clock-port 41234
```

自动的本机ROS验证使用合成相机和位姿后端，覆盖偏差、授时中断、恢复、在途任务及时间戳/序号：

```bash
export SB_DDS_INTERFACE=lo SB_ROS_DOMAIN_ID=74 SB_DDS_PEERS=127.0.0.1
source my_demo/ros2_handoff_env.sh
export ROS_LOG_DIR=/home/robot/FoundationPose/runs/sb_vision/clock_sync_20260919/ros_logs
/home/robot/anaconda3/envs/pose/bin/python my_demo/scripts/test_ros_clock_sync.py
```

既有`test_live_vision.py`和`test_ros_handoff.py`都是同机测试，已显式选择`local`；因此不会误把“同机测试通过”当成“三机NTP已配置”。

## 后续真实授时配置（本次未应用）

三台机器确定同一个固定LAN授时源后，先生成可审阅的systemd-timesyncd客户端配置：

```bash
python3 my_demo/scripts/configure_clock_sync.py --server <共同NTP服务器>
```

默认只在`runs/sb_vision/clock_sync_setup/`生成`90-sb-vision.conf`和`clock_sync.json`，不连接该服务器、不改服务。配置清空此前NTP列表和FallbackNTP，并设置16～64秒轮询。若存在DHCP/per-interface NTP，应确认实际选中的源符合配置；运行检查会拒绝其他源。

需要实际应用时，由有管理员权限的终端执行：

```bash
sudo python3 my_demo/scripts/configure_clock_sync.py --server <共同NTP服务器> --apply
python3 my_demo/scripts/check_clock_sync.py --clock-server <共同NTP服务器>
bash my_demo/scripts/run_sb_vision.sh --clock-server <共同NTP服务器> --image-transport compressed
```

`--apply`备份项目管理的旧drop-in，配置并重启timesyncd；如果chrony已经运行，会拒绝启动竞争的服务，需在chrony中配置相同源。应在正式控制启动前完成系统校时。每台机器都运行只读检查并保存报告，才能确认三机配置及测量结果；本机模拟测试不能替代这一部署步骤。

## 本轮验证结果

- 68项单元/回归测试通过。
- 本机真实UDP NTP报文与CycloneDDS测试通过：0ms参考时钟下正常发布；+20、-20、+400、-400ms四种偏差全部停发，归零后自动恢复。
- 授时端声明未同步、授时端不应答时均停发，恢复应答后继续发布。
- 1个跨同步状态变化的旧在途结果被丢弃；53条有效观测的采集时间戳保持原值，发布序号未重置。
- 报告：`runs/sb_vision/clock_sync_20260919/local_ntp_ros_validation.json`。这是模拟授时偏差/相机/位姿后端的本机集成测试，不是三台实机已完成系统校时的证明。
- `example_only/`用文档保留地址192.0.2.10验证配置生成；未应用、未连接此地址。默认项目授时服务器仍未设置。
