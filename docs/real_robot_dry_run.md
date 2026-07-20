# G1 真机抓瓶 dry-run

这条链路的边界是“看相机并规划”，不会操控机器人：

1. 机器人计算机只运行 RealSense RGB-D 发布服务；
2. 本机接收一帧对齐后的 RGB、16 位深度和相机内参；
3. 用户在保存的 RGB 上指定瓶子像素；
4. HUG 生成相机系 Inspire 抓取位姿，官方 G1 腰部/相机 FK 将其变换到 pelvis；
5. 离线 IK 生成 `current → pregrasp → grasp → close → lift` 轨迹；
6. 结果只保存为 NPZ/JSON/PNG，不发送 DDS 或电机命令。

代码位置：

- 相机协议与只读客户端：`src/lab_g1_collect/real_camera.py`
- 机器人端无损 RGB-D 服务：`src/lab_g1_collect/real_camera_server.py`
- HUG、坐标变换、IK 和 dry-run 输出：`src/lab_g1_collect/real_grasp.py`
- 官方 G1 相机链 FK：`src/lab_g1_collect/g1_camera_geometry.py`
- LowState SSH 客户端：`src/lab_g1_collect/real_state.py`
- SONIC SDK 只读状态工具：`tools/g1_state_reader/`
- 安全与规划配置：`configs/real_robot_dry_run.yaml`

## 1. 环境准备

本机继续使用项目 README 中的 `unitree_sim_env`，依赖安装在该环境中：

```bash
conda activate unitree_sim_env
pip install -e '.[real]'
```

机器人端沿用 GR00T-WBC 官方安装脚本创建的 `.venv_camera`，不修改系统库、CUDA 或
驱动。将本仓库通过 Git 放到机器人后，用源码路径让相机 venv 找到模块；官方相机
环境已经包含 RealSense、OpenCV、ZMQ 和 msgpack：

```bash
cd /home/unitree/data_collection/GR00T-WholeBodyControl
PYTHONPATH=/机器人上的/sim_data_collect/src \
  .venv_camera/bin/python -m lab_g1_collect.real_camera_server --port 5555
```

这个服务只打开 RealSense 和 TCP 端口，不导入或访问机器人 DDS。

机器人端还需要用 SONIC 自带的官方 Unitree SDK 编译一次只读 LowState 工具：

```bash
cd /home/unitree/data_collection/sim_data_collect
cmake -S tools/g1_state_reader -B outputs/real_robot_tools/build \
  -DSONIC_ROOT=/home/unitree/data_collection/GR00T-WholeBodyControl \
  -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/real_robot_tools/build --target g1_read_lowstate -j2
cp outputs/real_robot_tools/build/g1_read_lowstate \
  outputs/real_robot_tools/g1_read_lowstate
```

该工具只包含 `ChannelSubscriber<LowState_>`，从 `rt/lowstate` 读取右臂硬件索引
`22~28` 和腰部索引 `12~14`，不包含 publisher。`lab-g1-real` 默认通过 SSH 别名
`g1_bjutech` 调用它；只有离线复现实验才需要用 `--right-arm-q` 覆盖自动读数。

如果只想先确认官方服务的 RGB 连通性，也可以运行：

```bash
cd /home/unitree/data_collection/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense --port 5555
```

官方服务的 RGB 可以被客户端读取，但其 RealSense 深度当前走普通 JPEG 编码，因此
客户端会明确标记为 `gear-sonic-rgb-only`，并拒绝拿它做 HUG/IK 规划。

## 2. 只读抓图

在本机执行；当前有线连接的机器人相机地址已写入配置：

```bash
conda activate unitree_sim_env
lab-g1-real --capture-only \
  --output outputs/real_robot_dry_run/check_camera
```

检查 `rgb.png` 中瓶子是否完整可见。输出 JSON 中应有：

```text
"control_output_enabled": false
"source_protocol": "lab-g1-rgbd-v1"
"has_metric_depth": true
```

## 3. 官方相机与腰部 FK

本项目不再要求另做 `T_base_camera` 标定。固定几何来自 NVIDIA 官方
GR00T-WholeBodyControl 的
`gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml`：

- `pelvis -> waist_yaw_link`：绕 Z 的 `waist_yaw_joint`；
- `waist_roll_link pos="-0.0039635 0 0.035"`，绕 X；
- `torso_link pos="0 0 0.019"`，绕 Y；
- `head_camera pos="0.06 0 0.45" euler="0 -0.8 -1.57"`。

RealSense 使用光学坐标 `+X right, +Y down, +Z forward`；MuJoCo camera 使用
`+X right, +Y up, -Z forward`，代码显式转换二者。最终定义为：

```text
p_pelvis = T_pelvis_camera_optical(waist_yaw, waist_roll, waist_pitch) @ p_camera
```

自动 LowState 会同时读取腰部硬件索引 `12~14`，因此新抓图默认使用同一时刻的腰姿态。
复现历史 `capture.npz` 时必须用 `--waist-q YAW ROLL PITCH` 提供捕获时状态；若未知，
只能像仿真评审一样明确记录姿态假设，不能将结果用于真机。

## 4. 生成抓瓶计划

先从 `rgb.png` 读取瓶身内部一个有有效深度的像素 `(u, v)`。当前右臂 7 个关节角会
自动从 SONIC 同款 `rt/lowstate` 订阅读取：

```bash
conda activate unitree_sim_env
lab-g1-real \
  --capture outputs/real_robot_dry_run/check_camera/capture.npz \
  --target-u <u> --target-v <v> \
  --output outputs/real_robot_dry_run/bottle_plan
```

计划输出包括：

- `dry_run_plan.npz`：13 维语义轨迹（右臂 7 个 URDF 弧度 + Inspire 6 个 URDF 弧度）；
- `plan.json`：目标深度、基座系目标、各阶段 IK 残差和最大关节速度；
- `target_preview.png`：瓶子条件点复核图；
- `hug_runtime/` 对应的 HUG 输入/输出诊断由现有 HUG 流程保存。

`dry_run_plan.npz` 不是 Unitree/DFX 的硬件命令，不能直接发布。尤其 Inspire 真机接口
使用的 0~1 开合语义与这里的 URDF 弧度不同；本 dry-run 路径刻意不包含转换器或发布器。

若独立 URDF IK 或真机速度门槛不通过，但需要把候选送入 Isaac 继续诊断，可添加
`--simulation-review`。输出会保留 `simulation_review_only=true`、失败阶段和超限值；
这种计划只允许传给 `sim_smoke --replay-real-plan`，不能用于真机。

## 5. 当前尚不能省略的实机信息

- 瓶身上有效深度像素。
- 与 RGB-D 同时刻的腰部与右臂 LowState。

缺少其中任一项时，不应把生成的位姿用于真机执行。

## 6. 真实计划只在 Isaac 中回放

仿真桌板默认是带碰撞的 `1.20 x 0.80 x 0.04 m` 水平板，上表面严格位于 `0.70 m`：

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" conda run --no-capture-output -n unitree_sim_env \
python -m lab_g1_collect.sim_smoke --device cpu --headless --auto-collect \
  --steps 900 --episodes 1 --object-shape cylinder --table-top-height-m 0.70 \
  --replay-real-plan outputs/real_robot_dry_run/bottle_plan/dry_run_plan.npz \
  --arm-ik xr_teleoperate --xr-ik-profile autonomous \
  --output outputs/real_bottle_sim_review
```

episode 元数据会额外保存桌高、非手指右臂最大接触力和右臂连杆原点最低桌面净空。
默认 40 mm 到点门槛不变；`--waypoint-tolerance-m` 只用于仿真敏感性复核。
