# FoundationPose 视觉交付文档包

生成日期：2026-09-23

本目录只包含按照 `/home/robot/list` 中 Markdown 需求整理的交付文档，不包含源代码、脚本、消息文件、JSON、模型、权重、日志或 rosbag。

## 阅读顺序

1. [交付总清单](DELIVERY_CHECKLIST.md)：按原始需求 §6、§9.4 标记完成状态和缺口。
2. [接口与字段语义](INTERFACE_CONTRACT.md)：说明话题、坐标、时间戳、置信度和序号。
3. [环境与依赖](ENVIRONMENT_AND_DEPENDENCIES.md)：回答原始 §8.3 九个问题。
4. [标定与模型](CALIBRATION_AND_MODEL.md)：手眼结果、mesh 原点和瓶子尺寸。
5. [运行模式](FOUNDATIONPOSE_RUNTIME.md)：register、track、重注册、目标丢失和多目标策略。
6. [验收结果与计划](ACCEPTANCE_STATUS.md)：已有证据、最终验收命令和待录数据。

## 原始需求

- [视觉模块对接需求单](requirements/vision_handoff.md)
- [FoundationPose 视觉交接与问题分析](requirements/vision_foundationpose_handoff_analysis.md)

`project_docs/` 保存当前项目说明，`reports/` 保存已有 Markdown 测试报告。非 Markdown 证据仍位于原项目 `/home/robot/FoundationPose/runs/`、`my_demo/config/` 和 `.state/`，具体路径写在验收文档中。
