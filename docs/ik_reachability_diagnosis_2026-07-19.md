# HUG 腕位姿可达性复核（2026-07-19）

## 结论

现有 100 组实验不能证明被拒绝的 HUG 腕位姿在物理上不可达。当前代码的“离线可达”
来自 Pinocchio URDF，而执行与实测来自 Isaac USD；两者在初始帧消除固定基座/末端偏移
后，随着关节运动仍出现数百毫米的 FK 分歧。因此先前的“8 个候选均未通过 G1 右臂
6D IK”只表示没有候选通过当前近似模型及工程阈值，不应称为真实 IK 不可达。

## 具体例子

`episode_000015.failed` 在向 pre-grasp 运动时：

- 在线末端跟踪误差在记录帧 6（约仿真步 60）首次超过 40 mm；
- pre-grasp 阶段结束时误差为 150.2 mm；
- 同一组实测关节角的 Pinocchio/Isaac FK 位置差在初始帧刚体对齐后由 0 增至
  872.4 mm。

这说明该例是“控制器未沿目标路径到达终点”，但不能据此断言记录帧 6 对应的路径点
或最终 pre-grasp 在真实 Isaac 运动学中不可达。要回答物理可达性，必须改用 Isaac
自身运动学逐点求解，再用求得的连续关节轨迹回放验证。

`episode_000082.failed` 则提供反例：末端最终误差只有 12.0 mm，并且确实抬起圆柱，
说明至少部分 HUG 位姿能够由 Isaac 中的机器人到达。该次因圆柱线速度和角速度过大，
被抓取稳定性条件拒收，而不是 IK 失败。

## 复现产物

- `outputs/ik_diagnostics/episode_000015/ik_reachability.png`
- `outputs/ik_diagnostics/episode_000015/report.json`
- `outputs/ik_diagnostics/episode_000082/ik_reachability.png`
- `outputs/ik_diagnostics/episode_000082/report.json`

诊断脚本为 `tools/analyze_ik_episode.py`。左图显示目标路径、Isaac 实测路径、阶段终点和
首次误差超限点；右图同时显示在线跟踪误差及消除初始固定变换后的 Pinocchio/Isaac
FK 分歧。
