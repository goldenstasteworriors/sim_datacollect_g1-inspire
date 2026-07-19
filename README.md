# G1 右手实验器材自动数据采集 Pipeline

本项目复现并替换了参考方案的关键链路：HUG 从 RGB-D 生成 MANO 右手抓取，
将 21 点人手姿态重定向到 Unitree G1 上的 Inspire RH56DFTP 右手，再由右臂完成
接近、闭合和抬升。每帧同步保存双相机 RGB、深度、右臂 7 维状态和右手 6 维状态，
用于后续 LeRobot/SmolVLA 训练。

## 已固定的官方依赖

- `third_party/hug`：HUG 官方代码与模型接口。
- `third_party/unitree_sim_isaaclab`：Unitree 官方 G1-29DoF + Inspire 仿真本体、相机和 DDS。
- `third_party/xr_teleoperate`：Unitree 官方 Inspire URDF、DexPilot 配置和重定向实现。
- `third_party/LabUtopia`：LabUtopia 官方代码。
- `assets/labutopia/Beaker_01.usd`：LabUtopia-Dataset 的烧杯 OpenUSD 资产。

这些仓库均以 Git submodule 固定到可复现提交。仿真统一采用 Isaac Sim 5.1，避免修改
系统 CUDA 或显卡驱动。

Inspire 手指重定向复用 Unitree 官方 `xr_teleoperate` 中固定版本的
`dex-retargeting`、Inspire URDF 和 DexPilot 配置。按照上游方式安装到 IsaacLab
环境；使用 `--no-deps` 保留 IsaacLab 已验证的 Torch/CUDA 组合：

```bash
git submodule update --init --recursive
conda activate unitree_sim_env
pip install --no-deps -e third_party/xr_teleoperate/teleop/robot_control/dex-retargeting
```

官方配置的主动关节顺序为拇指旋转、拇指弯曲、食指、中指、无名指、小指；采集层
会重排成 Inspire API 的小指、无名指、中指、食指、拇指弯曲、拇指旋转。输出单位
为 URDF 弧度并按官方限位裁剪，不再把 `0~1` 归一化量直接当作关节角。

## 环境

轻量数据流和测试使用现有 `unitree_sim_env`：

```bash
conda activate unitree_sim_env
pip install -e .
lab-g1 --config configs/pipeline.yaml doctor
```

HUG 严格按官方 README 使用独立的 `hug` 环境。MANO 从
`/home/ykj/commonly_used/mano_v1_2/models` 读取，不复制或提交许可文件。HUG 权重放在
`checkpoints/hug/hug_full.safetensors`。

## 快速验证

不启动仿真即可跑通 120 帧、双 RGB、深度、状态和动作的完整 episode：

```bash
lab-g1 --config configs/pipeline.yaml mock-collect
lab-g1 validate outputs/lab_beaker_g1_inspire/episode_000000
```

数据维度固定为右臂 7 + Inspire 右手 6，共 13 维。右手顺序与 Unitree 官方
`DFX_inspire_service` 一致：`pinky, ring, middle, index, thumb_bend, thumb_rotation`。

在装有 LeRobot 0.4.x 的 `labvla` 环境中可直接转换为官方数据格式：

```bash
conda activate labvla
pip install -e .
lab-g1 export-lerobot \
  outputs/lab_beaker_g1_inspire/episode_000000 \
  outputs/lerobot_lab_g1_beaker
```

## HUG 推理

官方示例输入可用于检查权重和 MANO：

```bash
conda activate hug
python -m hug.prepare_inputs --dataset-path third_party/hug/data/custom
PYTHONPATH=src python -m lab_g1_collect.hug_condition \
  third_party/hug/data/custom/custom.pkl --u 122 --v 105
python -m hug.inference \
  --checkpoint-path checkpoints/hug/hug_full.safetensors \
  --dataset-path third_party/hug/data/custom \
  --num-samples 1 \
  --sampling-steps 2
```

`hug_condition` 将交互式 App 的目标点击写成 HUG 批量推理所需的
`condition_point` 和 PNG 掩码；坐标使用 HUG 的 224×224 模型输入尺度。
每次仿真调用 HUG 时，输入会保存到 `outputs/hug_runtime/episode_xxxxxx/`：
`capture.npz` 是传给独立 HUG 进程的完整数据，`rgb.png` 和 `depth_mm.png` 是原始
RGB/毫米深度，`input_metadata.json` 记录相机内参、中心裁剪和条件点，
为控制存储量，完整 PNG 可视化默认每 10 个 episode 保存一次。`hug_input_preview.png`
并排显示 HUG 的 224×224 RGB、3 m 内有效深度与烧杯条件点；
超出 HUG 点云范围的深度显示为黑色。推理完成后，`hug_output_wrist_pose.png` 会把
MANO 手骨架和相机系腕部 XYZ 三轴投影到输入图，原始矩阵保存在
`hug_output_metadata.json`。

HUG 保存的 `grasp_pred/*.pkl` 中包含 `landmarks_3d` 和 `T_camera_wrist`；
`lab_g1_collect.retarget.load_hug_prediction` 会校验并映射到 RH56DFTP。
其中 `T_camera_wrist` 先经过显式的 MANO 腕到 Inspire 安装基座 SE(3) 外参，
再从相机系变换到世界系。右臂差分 IK 使用完整的目标位置与四元数，而不是只做
位置 IK。GUI 中 18 cm 坐标轴表示 HUG 抓取位姿，10 cm 坐标轴表示预抓取位姿；
终端的 `[HUG target]` 同时输出世界坐标、`wxyz` 四元数、距烧杯距离和可达性判断。
运行时默认用 5 个 ODE steps 一次生成 8 个候选，并选择标定后 Inspire base 距条件
点最合理的候选。右手先到 10 cm 的小坐标轴，再沿直线到 18 cm 的大坐标轴并闭合
手指；实测末端没有进入 waypoint 容差时不会切换阶段，等待超时则按 IK 不可达失败
并自动 reset。相关 4096 次 HUG 重复性和 IK 消融见
`docs/hug_ik_experiment_2026-07-19.md`。

## Isaac Sim 场景

自定义任务 ID 为 `Isaac-PickPlace-LabBeaker-G129-Inspire-Right`，配置位于
`sim_tasks/lab_beaker_g1_inspire/env_cfg.py`。它继承 Unitree 官方
`PickPlaceG129InspireBaseFixEnvCfg`，保留 G1、Inspire 双手、前视和腕部相机，
但只生成机器人正前方的一张桌子、LabUtopia 烧杯、地面和灯光；仓库背景及其余
五张演示桌不会生成。策略/采集层只输出右臂和右手动作，左侧保持默认位姿。

启动前设置 Unitree 官方资源根目录：

```bash
conda activate unitree_sim_env
export PROJECT_ROOT="$PWD/third_party/unitree_sim_isaaclab"
export PYTHONPATH="$PWD:$PWD/third_party/unitree_sim_isaaclab:$PYTHONPATH"
python third_party/unitree_sim_isaaclab/sim_main.py \
  --device cpu --enable_cameras \
  --task Isaac-PickPlace-LabBeaker-G129-Inspire-Right \
  --enable_inspire_dds --robot_type g129
```

无 DDS 的场景加载和物理单步检查使用：

```bash
PROJECT_ROOT="$PWD/third_party/unitree_sim_isaaclab" \
PYTHONPATH="$PWD:$PWD/third_party/unitree_sim_isaaclab" \
python -m lab_g1_collect.sim_smoke --device cpu --steps 2
```

PC 桌面可视化可使用 `--no-headless`，并把 `--steps` 设置为期望保持窗口的物理步数。
加入 `--auto-collect --output outputs/gui_collect` 后，每轮正常动作包含 600 个逻辑步；
pre-grasp/grasp 到点门控可能增加物理等待步数。
无论成功或失败都会自动 reset 机器人和烧杯并开始下一轮。成功目录名为
`episode_NNNNNN`，失败目录名为 `episode_NNNNNN.failed`，两者的 `metadata.json`
都包含判定指标。

可视化运行期间，将焦点放在启动仿真的终端并按单键 `r`（无需回车），当前 episode
会以 `manual_reset=true` 保存为失败，然后立即 reset、刷新 RGB-D 并重新执行 HUG
规划。`Ctrl+C` 仍用于退出整个程序。

调试抓取几何时可用 `--object-shape box|cylinder|sphere` 替换烧杯；物体 reset 默认
在 XY 各 ±4 cm 内按 `--seed` 可复现随机。可用 `--hand-closure-scale` 调节闭合幅度，
`--hug-candidates` 和 `--hug-sampling-steps` 调节 HUG 推理。`--debug-ik` 会打印关键阶段的末端目标、
位置误差和雅可比预测位移。

已有 episode 可用下面的诊断工具区分目标轨迹跟踪误差与离线 Pinocchio/Isaac FK
不一致。图中的红点只表示在线跟踪误差首次超过门限，不能在 FK 模型一致性验证通过前
解释为该 Cartesian 点在物理上不可达：

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" conda run --no-capture-output -n unitree_sim_env \
python tools/analyze_ik_episode.py outputs/ik_filtered_100/episode_000015.failed \
  --output outputs/ik_diagnostics/episode_000015
```

成功要求最后 30 个物理步同时满足：物体相对初始高度始终不低于 3 cm、物体与
最近右手指节距离始终不超过 10 cm、物体线速度不超过 0.15 m/s、角速度不超过
1.0 rad/s；任意条件不满足即判失败。

reset 后先用零偏移关节指令推进 20 个控制步，让物体落稳并刷新 RGB-D，再执行
下一轮 HUG 规划。episode 会额外保存末端实际位姿 `observation.ee_pose_w` 和目标位姿
`target.ee_pose_w`，用于检查直线 approach、回退和横向偏差。
终端依次打印 `resetting environment`、`planning with fresh RGB-D` 和 `running`，
可用于区分 reset、推理等待和动作执行状态。

真实 G1 部署沿用 Unitree 官方 DDS：手命令发布到 `rt/inspire/cmd`，状态从
`rt/inspire/state` 读取。连接真机前应先在仿真检查关节方向、工作空间、碰撞与抓取力度。

## MANO wrist / Inspire base 交互标定

`tools/retarget_calibrator` 会将当前代码中的 `T_mano_wrist_inspire_base` 固定标定作为
初始值，叠加显示半透明 canonical MANO 开手网格和 Unitree 官方 Inspire URDF 零关节网格。可用
XYZ/RPY 输入框或三维移动、旋转手柄调节 Inspire base，随后复制 JSON 参数：

```bash
cd tools/retarget_calibrator
npm install
npm run dev -- --port 4173
```

浏览器打开 `http://127.0.0.1:4173`。快捷键 `W`/`E` 切换移动/旋转，`F` 恢复合适视角。

## 数据质量约束

- 每个物体位置/尺度变体至少 10 条成功轨迹，总计建议不少于 50 episodes。
- 只有烧杯离桌并稳定保持的轨迹才标记为成功；碰桌、滑落和超关节限位必须拒收。
- 保存源 RGB-D、HUG 抓取、机器人状态、动作和时间戳，便于重定向或重新渲染。
- 透明玻璃的仿真深度可能失真，应在材质、光照、摩擦和质量上做域随机化。
