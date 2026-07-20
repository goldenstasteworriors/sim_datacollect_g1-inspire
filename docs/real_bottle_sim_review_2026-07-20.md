# 真实瓶子 RGB-D → 0.7 m 桌面仿真评审（2026-07-20）

## 结论

当前计划不能用于真机。严格 40 mm 到点门槛下，右手在 pre-grasp 处仍差 44.2 mm，
安全门控停止了后续闭手；将门槛仅在仿真中放宽到 50 mm 后虽然走完闭手与抬升，
瓶子仍未被抬起，直线接近最大横向偏差为 49.8 mm。

两次仿真均未检测到非手指右臂与环境接触，最大接触力为 0 N；右臂相关 body origin
距 0.7 m 桌面最低约 57 mm。这个数值说明本次失败不是明显撞桌造成的，但 body origin
净空不能替代真实碰撞几何余量，因此仍需保留物理接触门控。

## 真实输入与几何

- 输入：`outputs/real_robot_dry_run/check_rgbd/capture.npz`（16:09 的最近可用瓶子帧）；
- 条件点：原图 `(310, 355)`，9×9 有效深度中值 `0.630 m`；
- 官方模型：GR00T-WholeBodyControl `g1_29dof_with_hand.xml`；
- 历史帧腰姿态假设：`[0, 0, 0] rad`，所以结果只作仿真评审；
- 瓶心：pelvis 系 `[0.4299, 0.0227, -0.0154] m`；
- Isaac 世界系瓶心：`[-0.1727, 0.4299, 0.7439] m`；
- HUG grasp 腕到瓶心距离：约 `0.141 m`。

17:00 保存的两帧已经看向地面且不再包含目标瓶。恢复有线网络后必须重新抓取 RGB-D
并同步读取腰部/右臂 LowState，不能把本报告的旧目标直接用于真机。

## 安全门控结果

| 检查 | 严格 40 mm | 仿真敏感性 50 mm |
|---|---:|---:|
| pre-grasp reached | 否，最终误差 44.2 mm | 是 |
| grasp reached | 未执行 | 是（宽松定义） |
| approach 最大横向偏差 | 未执行 | 49.8 mm |
| 瓶子最低抬升 | -3.9 mm | -3.9 mm |
| 非手指右臂最大接触力 | 0 N | 0 N |
| 最低 body-origin 桌面净空 | 57.7 mm | 57.2 mm |
| 严格任务结果 | 失败 | 失败 |

严格结果位于 `outputs/real_bottle_sim_review_20260720/episode_000000.failed/`；完整闭手
敏感性结果位于
`outputs/real_bottle_sim_review_relaxed_20260720/episode_000000.failed/`。

## Inspire 桥接接口

现有 Unitree Inspire DFX DDS 约定为：命令 `rt/inspire/cmd`、状态
`rt/inspire/state`，`MotorCmds.cmds[0:6]` 是右手小指、无名指、中指、食指、拇指弯曲、
拇指旋转，`cmds[6:12]` 是左手同序。硬件值为 `[0,1]`，其中 `0` 全闭、`1` 全开；
用户的 `inspire_modbus_hand.py --mode dds` 是 DDS 到左右手 Modbus IP 的桥。

本次网络中断时无法读取机器人上该脚本的实际版本；项目只记录接口，不创建或运行
`rt/inspire/cmd` publisher。重新连接后应先只读核对脚本与 `rt/inspire/state`，再讨论
真机 executor。
