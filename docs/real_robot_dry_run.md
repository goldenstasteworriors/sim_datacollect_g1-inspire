# G1 真机抓瓶 dry-run

这条链路的边界是“看相机并规划”，不会操控机器人：

1. 机器人计算机只运行 RealSense RGB-D 发布服务；
2. 本机接收一帧对齐后的 RGB、16 位深度和相机内参；
3. 用户在保存的 RGB 上指定瓶子像素；
4. HUG 生成相机系 Inspire 抓取位姿，标定外参将其变换到 G1 base；
5. 离线 IK 生成 `current → pregrasp → grasp → close → lift` 轨迹；
6. 结果只保存为 NPZ/JSON/PNG，不发送 DDS 或电机命令。

代码位置：

- 相机协议与只读客户端：`src/lab_g1_collect/real_camera.py`
- 机器人端无损 RGB-D 服务：`src/lab_g1_collect/real_camera_server.py`
- HUG、外参、IK 和 dry-run 输出：`src/lab_g1_collect/real_grasp.py`
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

## 3. 标定前置条件

规划必须使用真实的 `T_base_camera`，定义为把相机光学坐标系中的点变换到 G1 base
坐标系：

```text
p_base = T_base_camera @ p_camera
```

完成手眼/外参标定并复核坐标轴、米制单位后，把矩阵填入
`configs/real_robot_dry_run.yaml`，再把 `calibration.valid` 改为 `true`。默认配置使用
占位单位阵并保持 `valid: false`；程序会拒绝规划，防止把占位外参误当成真机外参。

## 4. 生成抓瓶计划

先从 `rgb.png` 读取瓶身内部一个有有效深度的像素 `(u, v)`。再提供机器人当前右臂
7 个关节角，顺序与 `configs/pipeline.yaml` 的 `arm_joint_names` 相同：

```bash
conda activate unitree_sim_env
lab-g1-real \
  --capture outputs/real_robot_dry_run/check_camera/capture.npz \
  --target-u <u> --target-v <v> \
  --right-arm-q <q1> <q2> <q3> <q4> <q5> <q6> <q7> \
  --output outputs/real_robot_dry_run/bottle_plan
```

计划输出包括：

- `dry_run_plan.npz`：13 维语义轨迹（右臂 7 个 URDF 弧度 + Inspire 6 个 URDF 弧度）；
- `plan.json`：目标深度、基座系目标、各阶段 IK 残差和最大关节速度；
- `target_preview.png`：瓶子条件点复核图；
- `hug_runtime/` 对应的 HUG 输入/输出诊断由现有 HUG 流程保存。

`dry_run_plan.npz` 不是 Unitree/DFX 的硬件命令，不能直接发布。尤其 Inspire 真机接口
使用的 0~1 开合语义与这里的 URDF 弧度不同；本 dry-run 路径刻意不包含转换器或发布器。

## 5. 当前尚不能省略的实机信息

- 实际安装后的 `T_base_camera` 外参；
- 抓图时右臂 7 关节状态；
- 瓶身上有效深度像素。

缺少其中任一项时，只进行 `--capture-only` 检查，不应把生成的位姿用于真机执行。
