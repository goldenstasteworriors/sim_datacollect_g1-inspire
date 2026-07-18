# G1 右手实验器材自动数据采集 Pipeline

本项目复现并替换了参考方案的关键链路：HUG 从 RGB-D 生成 MANO 右手抓取，
将 21 点人手姿态重定向到 Unitree G1 上的 Inspire RH56DFTP 右手，再由右臂完成
接近、闭合和抬升。每帧同步保存双相机 RGB、深度、右臂 7 维状态和右手 6 维状态，
用于后续 LeRobot/SmolVLA 训练。

## 已固定的官方依赖

- `third_party/hug`：HUG 官方代码与模型接口。
- `third_party/unitree_sim_isaaclab`：Unitree 官方 G1-29DoF + Inspire 仿真本体、相机和 DDS。
- `third_party/LabUtopia`：LabUtopia 官方代码。
- `assets/labutopia/Beaker_01.usd`：LabUtopia-Dataset 的烧杯 OpenUSD 资产。

这些仓库均以 Git submodule 固定到可复现提交。仿真统一采用 Isaac Sim 5.1，避免修改
系统 CUDA 或显卡驱动。

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
python -m lab_g1_collect.hug_condition \
  third_party/hug/data/custom/custom.pkl --u 122 --v 105
python -m hug.inference \
  --checkpoint-path checkpoints/hug/hug_full.safetensors \
  --dataset-path third_party/hug/data/custom \
  --num-samples 1 \
  --sampling-steps 2
```

`hug_condition` 将交互式 App 的目标点击写成 HUG 批量推理所需的
`condition_point` 和 PNG 掩码；坐标使用 HUG 的 224×224 模型输入尺度。

HUG 保存的 `grasp_pred/*.pkl` 中包含 `landmarks_3d` 和 `T_camera_wrist`；
`lab_g1_collect.retarget.load_hug_prediction` 会校验并映射到 RH56DFTP。

## Isaac Sim 场景

自定义任务 ID 为 `Isaac-PickPlace-LabBeaker-G129-Inspire-Right`，配置位于
`sim_tasks/lab_beaker_g1_inspire/env_cfg.py`。它继承 Unitree 官方
`PickPlaceG129InspireBaseFixEnvCfg`，因此保留 G1、Inspire 双手、前视和腕部相机，
只把圆柱目标替换为 LabUtopia 烧杯。策略/采集层只输出右臂和右手动作，左侧保持默认位姿。

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

真实 G1 部署沿用 Unitree 官方 DDS：手命令发布到 `rt/inspire/cmd`，状态从
`rt/inspire/state` 读取。连接真机前应先在仿真检查关节方向、工作空间、碰撞与抓取力度。

## 数据质量约束

- 每个物体位置/尺度变体至少 10 条成功轨迹，总计建议不少于 50 episodes。
- 只有烧杯离桌并稳定保持的轨迹才标记为成功；碰桌、滑落和超关节限位必须拒收。
- 保存源 RGB-D、HUG 抓取、机器人状态、动作和时间戳，便于重定向或重新渲染。
- 透明玻璃的仿真深度可能失真，应在材质、光照、摩擦和质量上做域随机化。
