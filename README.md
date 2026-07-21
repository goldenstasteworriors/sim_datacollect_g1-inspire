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

HUG 严格按官方 README 使用独立的 `hug` 环境。MANO 默认从
`~/commonly_used/mano_v1_2/models` 读取，不复制或提交许可文件；可用
`LAB_HUG_MANO_DIR` 覆盖（既可指向 `mano_v1_2`，也可指向其 `models`
子目录）。HUG 权重放在 `checkpoints/hug/hug_full.safetensors`，DINOv2
缓存默认放在 `checkpoints/hug/hf_cache`，可用 `LAB_HUG_HF_HOME` 覆盖。

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
点最合理的候选。右手先到较小坐标轴表示的 pre-grasp，再沿直线到较大坐标轴表示的
grasp 并闭合
手指；实测末端没有进入 waypoint 容差时不会切换阶段，等待超时则按 IK 不可达失败
并自动 reset。相关 4096 次 HUG 重复性和 IK 消融见
`docs/hug_ik_experiment_2026-07-19.md`。

HUG 流程的 pre-grasp 默认沿“物体中心→grasp腕位置”方向向外偏移30 mm，可用
`--pregrasp-offset-m` 调整。2026-07-19之前的实验使用固定100 mm，历史报告中的结果
仍按当时配置解释。固定 `(-0.21, 0.28, 0.86) m` 的20次实验及20次100 mm对照见
`docs/pregrasp_offset_3cm_experiment_2026-07-19.md`。

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

当前自动采集不再使用 Pinocchio/SciPy 离线 IK 过滤 HUG 候选，也不再把离线关节解
混入执行命令。8 个候选只按 Inspire base 到物体中心的几何距离选择；实际手臂控制
默认直接调用宇树 `xr_teleoperate` 的 `G1_29_ArmIK`（Pinocchio + CasADi/IPOPT）；
`--arm-ik dls` 才会切回本项目在 Isaac PhysX Jacobian 上实现的旧阻尼最小二乘对照。
桥接代码位于 `src/lab_g1_collect/xr_teleoperate_ik.py`，使用常驻 `tv` conda 子进程，
避免把官方 Pinocchio 3.1/CasADi 依赖混入 IsaacLab 环境。
在线到点门控仍会记录实际末端未能在规定时间进入 40 mm 容差的执行失败，但不再把它
表述为离线模型证明的运动学不可达。

`xr_teleoperate` 默认使用 `--xr-ik-profile autonomous`：将上游遥操作目标改为位置优先，
用 Isaac 实际软限位作为硬约束，以区间中心代替全零关节正则，并跳过遥操作用的4帧
移动平均。`--xr-ik-profile teleop` 可恢复上游原始权重做对照。每个 episode 在运动前
还会用独立求解器依次直接求 pre-grasp 和 grasp，`endpoint_ik_probes` 记录静态位置、
姿态残差、求解状态和最小关节余量。20次对照见
`docs/autonomous_ik_profile_experiment_2026-07-20.md`。

同一自主 IK 在 LabUtopia 烧杯上的20次测试见
`docs/autonomous_ik_beaker_experiment_2026-07-20.md`。烧杯动态到达12/20、实际抓起5/20；
其 HUG 腕姿态残差明显高于圆柱，是当前主要差异。

`--ik-rotation-weight 0` 可做仅位置 IK 对照。一次去除离线 IK 后的圆柱实测中，默认
姿态权重在 pre-grasp 超时，剩余位置误差 97 mm；仅位置控制能够越过 pre-grasp，
但在后续 100 mm 直线 approach 末端仍剩余 97 mm，且关节限位余量约 0.85 rad。
这表明当前瓶颈确实在自写在线 DLS/位置执行链路，而不是关节到达限位；后续超时日志
使用 `tracking timeout`，不再写成未经证明的 `unreachable`。

`xr_teleoperate` 按官方教程使用 Python 3.10、Pinocchio 3.1.0、NumPy 1.26.4，并安装
上游 `requirements.txt`。首次圆柱对照的 pre-grasp 最终误差由旧DLS的约97 mm降至
54.8 mm，但 Isaac 实测最小软限位余量为 -0.0072 rad，说明官方 URDF 优化边界与当前
Isaac USD 软限位并不完全相同；当前仍按 Isaac 限位裁剪命令，不能用越界解强行执行。

可用 `--nearby-ik-test` 绕过 HUG 做确定可达区域验证：圆柱固定在
`(-0.040, 0.380, 0.860) m`，腕部保持初始旋转并仅平移
`(-0.040, +0.050, +0.020) m`（总位移约67 mm）。XR IK 实测 pre-grasp 误差约
14.3 mm、grasp 误差约22.4 mm，均低于40 mm到点门限，并成功形成近距离夹持；物体
末段最低抬升27.3 mm，略低于30 mm成功阈值，且角速度过大，所以严格标签仍为失败。
完整双视角帧保存在 `outputs/xr_ik_nearby_visual/episode_000000.failed/images/`。

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" conda run --no-capture-output -n unitree_sim_env \
python -m lab_g1_collect.sim_smoke --device cpu --headless --steps 900 --episodes 1 \
  --auto-collect --object-shape cylinder --nearby-ik-test --arm-ik xr_teleoperate \
  --keep-failure-visuals 1 --output outputs/xr_ik_nearby_visual
```

`--object-fixed-xyz X Y Z` 可在完整 HUG 流程中固定物体而不启用近距离测试捷径。将圆柱
从上述位置向世界 X 负方向移动20 mm，即固定在 `(-0.060, 0.380, 0.860) m` 后，完整
HUG 生成的 grasp 腕位姿为约 `(-0.050, 0.299, 0.983) m`，径向外推100 mm得到的
pre-grasp 高度达到约1.067 m。XR IK 在 pre-grasp 最终仍差70.2 mm，并由
`right_wrist_yaw_joint` 的 Isaac 软限位截断（余量约 -0.0075 rad），因此尚未进入闭手
阶段。这说明圆柱左移20 mm并未解决问题，主要矛盾仍是HUG旋转加固定100 mm径向
pre-grasp与Isaac腕部限位的组合，而不是圆柱原位置本身。

批量比较摆放位置时，可用 `--object-grid-xy N --object-grid-step-m STEP` 以
`--object-fixed-xyz` 为中心逐 episode 扫描 N×N 个 XY 位置。程序会把实际物体初始
位置、HUG grasp/pre-grasp 目标和 IK 跟踪失败阶段写入各 episode 的 `metadata.json`：

```bash
PYTHONPATH="$PWD/src:$PYTHONPATH" conda run --no-capture-output -n unitree_sim_env \
python -m lab_g1_collect.sim_smoke --device cpu --headless --steps 7000 --episodes 9 \
  --auto-collect --object-shape cylinder --object-fixed-xyz -0.080 0.340 0.860 \
  --object-grid-xy 3 --object-grid-step-m 0.020 --arm-ik xr_teleoperate \
  --output outputs/xr_ik_position_grid_left_3x3
```

2026-07-19 的两轮3×3位置扫描中，只有 `(-0.080, 0.360, 0.860) m` 和
`(-0.060, 0.360, 0.860) m` 各出现一次完整 pre-grasp/grasp 到点并执行闭手与抬升，
但均未抬起圆柱。两点各追加5次后，完整执行分别为1/5和0/5，严格成功均为0。
详细记录见 `docs/ik_object_position_sweep_2026-07-19.md`。结果表明当前 HUG 腕旋转及
外推 pre-grasp 的采样波动大于20 mm物体平移的影响，不能把单次 tracking timeout
解释为该物体位置在G1右臂工作空间中没有运动学解。

每个 HUG episode 还会在 `metadata.json` 中保存 `hug_candidate_stats`、
`hug_sampling_stats` 和 `selected_hug_candidate`。前者包含每个候选腕位姿及其到物体
中心的距离；后者包含同批候选的距离分布、两两平移差和四元数最短角旋转差。新增的
4×4位置扫描和固定位置80候选实验见
`docs/hug_sampling_position_sweep_2026-07-19.md`。

大范围位置实验可用 `--object-grid-step-xy DX DY` 分别指定 X/Y 网格间距。IK 轨迹诊断
会在 episode 元数据中保存 `pregrasp_reached`、`grasp_reached`、`failure_category` 和
`approach_diagnostics`，从而区分 pre-grasp 目标未到达、直线接近偏离和 grasp 终点未到达。
双臂之间11×11共121个位置的实验与原因分析见
`docs/between_arms_121_position_sweep_2026-07-19.md`。

GUI会持续检查7个右臂关节到Isaac软限位的最小余量。默认余量不超过0.08 rad时，
对应关节/link中心会出现半径40 mm的红色警示球，终端同时打印
`[joint-limit-warning]`、关节名、当前角度和余量；离开警戒区后红球自动隐藏。可用
`--joint-limit-warning-rad` 调整阈值。红球表示Isaac实际关节状态接近限位，不是HUG
坐标轴的一部分。

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

## 真机相机与抓瓶 dry-run

真机入口 `lab-g1-real` 固定为 `dry_run`：只接收相机数据，生成 HUG/IK 语义轨迹并
保存到磁盘；代码不导入 Unitree SDK、不创建 DDS publisher，也没有启用控制的命令行
参数。具体准备、相机命令、外参要求和输出说明见
[`docs/real_robot_dry_run.md`](docs/real_robot_dry_run.md)。
右臂当前角度默认由机器人端 subscriber-only 工具按照 SONIC 的实现订阅
`rt/lowstate`，读取硬件索引 `22:29`；不再要求手工抄写关节角。

机器人上的 GR00T-WBC 官方相机命令可先用于 RGB 连通性检查：

```bash
cd /home/unitree/data_collection/GR00T-WholeBodyControl
.venv_camera/bin/python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense --port 5555
```

由于官方 composed camera 当前把 RealSense 深度也经过普通 JPEG 编码，完整 HUG
dry-run 应改用本项目的 `lab-g1-rgbd-server`，它将深度对齐到彩色图并以 16 位 PNG
传输，同时附带实际相机内参。相机到 pelvis 的位姿按 GR00T-WBC 官方 G1 MJCF
和实时腰部 LowState 做链式 FK，不再要求单独标定一个静态 `T_base_camera`。

真实图像生成的计划可用 `--replay-real-plan PLAN.npz` 仅在 Isaac 中回放；场景默认
生成上表面为 `0.76 m` 的碰撞桌板，并记录非手指右臂接触力与桌面净空。首次真实瓶子
评审结果见 `docs/real_bottle_sim_review_2026-07-20.md`。

机器人 reset 默认采用 SONICMJ/GEAR-SONIC 的 29DoF 初始姿态，而不是 Unitree
Isaac 资产的全零手臂姿态。真机侧提供同一目标的 LowState dry-run 规划器，以及基于
宇树官方 `rt/arm_sdk` 示例的双臂+腰初始化工具；腿部始终留给 SONIC/WBC 平衡控制。
完整安全流程见 `docs/real_robot_dry_run.md`。

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
