# Autonomous IK 烧杯测试（2026-07-20）

## 配置

- LabUtopia beaker 资产；
- 物体 rigid root 固定在 `(-0.21, 0.28, 0.86) m`，HUG 条件点按烧杯几何中心向上偏移36 mm；
- pre-grasp 径向偏移30 mm；
- autonomous XR IK；
- 每次重新采集 RGB-D 并生成8个 HUG 候选；
- 共20次。

## 结果

| 指标 | 结果 |
|---|---:|
| 静态 pre-grasp 求解器返回 | 20/20 |
| 静态 pre-grasp 满足10 mm/15° | 1/20 |
| 静态 pre-grasp 位置残差 | 平均5.3 mm，范围3.5–10.3 mm |
| 静态 pre-grasp 姿态残差 | 平均21.0°，范围15.0–41.1° |
| 静态 grasp 满足10 mm/15° | 15/20 |
| 静态 grasp 位置残差 | 平均2.2 mm，范围0.7–5.6 mm |
| 静态 grasp 姿态残差 | 平均10.0°，范围2.8–29.1° |
| 动态 pre-grasp 到达 | 12/20 |
| 动态 grasp 到达 | 12/20 |
| 实际抬升不少于30 mm | 5/20 |
| 严格数据质量成功 | 0/20 |

5个抓起样本的末段最低抬升分别为63.1、113.8、72.8、61.7和51.1 mm。其中4个只因
物体角速度超过1 rad/s失败；其余样本还受到未抬起、指尖距离或线速度影响。

## 与圆柱对比

同一位置的圆柱 autonomous 20次达到 pre-grasp/grasp 20/20、抓起14/20、严格成功
1/20。烧杯下降到12/20和5/20。烧杯静态 grasp 位置通常仍有高精度解，主要退化来自
HUG 腕姿态：pre-grasp 姿态残差平均21.0°、最大41.1°，动态执行中8次在 pre-grasp
阶段超时。

单次 smoke test曾抬升103.9 mm，证明烧杯的资产姿态、碰撞和夹持链路可以工作；批量
成功率下降不是因为烧杯始终横放或完全无法碰撞，而是 HUG 姿态波动与杯体接触稳定性。

原始数据：

- `outputs/autonomous_ik_beaker_smoke`
- `outputs/autonomous_ik_beaker_xm210_y280_repeat20`
