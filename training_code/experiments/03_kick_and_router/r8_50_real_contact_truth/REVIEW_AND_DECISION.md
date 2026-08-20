# 当前代码审阅与继续决策（2026-08-11）

## 结论

现有材料足以继续到 R8.50，不需要先补更多历史文件。R8.42 是当前可信的
ball-free 父节点；R8.43--R8.49 可以保留为探索记录，但不能作为真实触球或踢球成功
证据。当前最省总时间的路线，是先用一次短的真实接触扫描确定可训练区域，再训练
Kick residual，而不是继续扩大旧代理指标实验或直接投入 PPO。

## 已确认合理的设计

- Delta-Pose 作为高层意图、12 关节 residual 作为低层动作，层次划分合理。
- 高层严格使用进球 `+1`、乌龙 `-1`、其他 `0`，保持零和目标，合理。
- 高低层奖励与优化器分离，避免低层 shaping 改写比赛目标，合理。
- R8.42 的 Load → Lift → Swing → Recovery 冻结链路及权重哈希可复现，可作为
  Kick 的物理父节点。
- Teacher flag 为零、部署低层不依赖 teacher action，方向正确。

## 必须修正的问题

| 问题 | 证据 | 风险 | R8.50 处理 |
|---|---|---|---|
| 球初始高度错误 | 半径 `0.11 m`，旧球心约 `ground+0.055 m` | 球体穿地，球速和接触代理被污染 | 球心固定 `ground+0.115 m` 并等待稳定 |
| 触球是代理指标 | 旧代码主要依赖足球距离、球位移/速度 | 不能证明发生 FL-foot↔ball 碰撞 | 使用 filtered `force_matrix_w_history` |
| 过滤拓扑可能误用 | ContactSensor 过滤只支持单传感体到多目标 | 多足 pattern 会失去可靠归因 | 以 SoccerBall 为单传感体，逐刚体过滤 |
| 串行几何扫描 | 每个 case 单独占用全部并行环境 | 浪费向量化能力 | 25 个 case 按 env id 同时运行 |
| 过早使用 4096 episodes | 真值尚未建立就做大样本确认 | GPU 时间浪费 | 首筛 128/case，胜者再扩大 |
| Recovery 接触指标混杂 | 未过滤足部净力会包含球的作用力 | 球碰脚可伪装为地面支撑 | R8.50 不用旧 Recovery 作为触球真值；R8.51 再加足-地过滤 |
| 70-D 与 62-D 不兼容 | 隔离 R8 是 70-D，V4D 主干是 62-D | 直接装权重会维度/语义错配 | 暂不直连，物理验收后做显式 64-D 迁移 |

## R8.50 输出如何决定下一步

- 对照失败：先修传感器或接触采样周期，不训练。
- 25 个 case 全部没有 clean real strike：保持 R8.42 冻结，扩大一轮小范围几何或
  调整 Swing 动作幅度；不重复旧 proxy 验证。
- 存在 clean real strike：只取排名前 1--3 个邻域进入 R8.51。
- 接触存在但球目标方向增速不足：训练 strike residual/strength，而不是重训
  Load/Lift/Recovery。
- 非足或错误足先接触偏高：先收窄几何和动作 mask，不用奖励“补救”错误物理路径。

## R8.51 的高效训练边界

冻结 R8.42 基座与已验收 heads，只训练小型 bounded strike residual 和 strength
conditioning。先使用 3 个离散强度锚点验证单调性，再连续化；训练期间以真实接触、
目标方向球速度增量、非足/错足惩罚和恢复稳定为低层信号。最终 3×4096 只用于候选
定型验收，不用于广搜。

## 最终接回比赛主干

物理 Kick 验收前保持 V4D 17-D 高层观察、3-D Delta-Pose action 和 62-D 低层观察
不变。验收后建议高层 action 扩为 5-D：前三维 Delta-Pose，第四维 Kick gate，
第五维 strength；低层 context 将 3-skill one-hot 扩为 4-skill，再加 strength，
即 62→64-D。复制共享输入权重，新列零初始化，先保证旧三技能零回归，再做联合 PPO。

## 还需要用户提供什么

首次运行前无需继续补历史材料。运行 R8.50 后只需回传：

1. `r850_real_contact_summary.json`；
2. `r850_episode_rows.jsonl`；
3. 对应 Slurm log；
4. 若 discovery 或 ContactSensor 初始化失败，再补 Isaac Lab/Isaac Sim 版本与完整报错。

这些结果足以决定 R8.51 是直接训练、缩网格复扫，还是先修接触采样。
