GO2 SOCCER 项目本机 / CLUSTER / GITHUB 完整交接说明
版本：2026-08-23
范围：截至累计 V1.8 的源码、训练代码、运行配置、视频评估链路和文件传输流程

======================================================================
0. 阅读顺序与最重要结论
======================================================================

本文件应放在本机以下位置：

  /home/ubuntu/projects/go2-kick-progress-report/training_code/README_LOCAL_CLUSTER_HANDOFF.txt

后续让 Codex 接管项目时，应把以下目录作为工作目录：

  /home/ubuntu/projects/go2-kick-progress-report

Codex 开始修改前，必须先阅读：

  training_code/README_LOCAL_CLUSTER_HANDOFF.txt
  training_code/CODEX_START_HERE.md
  training_code/FILE_MANIFEST.tsv

最重要的版本边界：

1. current_v1_8_forensic/train_r898.py 是累计到 V1.8 的取证源码，不是可直接继续长训的父版本。
2. V1.8 的 reverse-turn / route-first / hold 状态机已被日志证明会进入静止或锁存死锁。
3. V1.4--V1.8 的本地 Chain V 视频主要用于诊断运行时 router；视频评估不会产生 PPO 学习。
4. 下一条正确训练分支是 staging-locomotion：先学习“到达球后方动态 staging pose”，再接回 push/kick。
5. GitHub 公开仓库不保存 checkpoint、视频、巨型 summary、完整日志、Conda 二进制环境或完整 Isaac Lab 安装。

======================================================================
1. 本机统一项目位置
======================================================================

1.1 Git 仓库 / Codex 工作目录

  /home/ubuntu/projects/go2-kick-progress-report

1.2 分类后的训练代码根目录

  /home/ubuntu/projects/go2-kick-progress-report/training_code

已观察到该目录约为 152 MB。大小可能随后续 commit 改变。

1.3 当前 GitHub 远端

  SSH：git@github.com:Chester-031/go2-kick-progress-report.git
  HTTPS：https://github.com/Chester-031/go2-kick-progress-report
  基础分支：main
  训练代码分支：training-code-v1.8-20260820-184110

分支页面：

  https://github.com/Chester-031/go2-kick-progress-report/tree/training-code-v1.8-20260820-184110

1.4 Cluster 源码归档在本机的位置

  /home/ubuntu/下载/go2_soccer_v18_handoff/
  go2_soccer_source_through_v1_8_20260820_184110.tgz
  go2_soccer_source_through_v1_8_20260820_184110.tgz.sha256

确认信息：

  归档字节数：23417614
  SHA256：116743bc7f7b790fd36fed55733ed653373f0dff794121aa6a21cf5ca283efcc
  Cluster 收集文件数：1055
  Cluster 可选文件缺失数：0

1.5 累积 V1.8 的本机原始位置

  源码：
  /home/ubuntu/go2_r898_video_16044032_gate045/source/train_r898.py

  与播放器配套的本机项目包：
  /home/ubuntu/go2_r898_video_16044032_gate045/project_source_current

  分类仓库中的副本：
  /home/ubuntu/projects/go2-kick-progress-report/training_code/current_v1_8_forensic/train_r898.py

  V1.x 历史备份目录：
  /home/ubuntu/projects/go2-kick-progress-report/training_code/current_v1_8_forensic/history

======================================================================
2. training_code 目录内容与用途
======================================================================

2.1 src/

位置：

  training_code/src/

内容：项目运行时 Python package，包含 task registration、soccer 环境配置、
manager-based task 代码以及训练入口 import 的本项目小包。

关键 Cluster 原始路径：

  /scratch/lj2634/Isaac-Go2-Low-Level-RL/source/isaac_go2_low_level_rl/isaac_go2_low_level_rl/tasks/manager_based/soccer/

关键环境配置：

  soccer_milestone2_env_cfg.py

任务 ID：

  Isaac-Go2-Soccer-Milestone2-v0

2.2 experiments/01_initial_v4d/

用途：最初始 V4D high/low joint 训练基线。

关键文件：

  v4d_joint/train_go2_soccer_v4d_joint.py
  v4d_joint/run_go2_soccer_v4d_joint.sbatch

Cluster 原始位置：

  /scratch/lj2634/Isaac-Go2-Low-Level-RL/experiments/v4d_joint

2.3 experiments/02_delta_pose_high_low/

用途：无 Kick 主干、Delta-Pose、高层 +1/-1/0 零和目标、低层 12 关节动作、
high/low 联合训练和早期 router/pose-push 代码。

主要目录族：

  v3_zero_sum
  v3a_*
  delta_pose_joint_v0
  delta_pose_joint_v4_formal
  v4a_*
  v4bc_*

接口约束：

  - 高层最终目标仍是严格 +1/-1/0 的比赛结果。
  - Kick 技能选择与强度属于 action/interface 扩展。
  - Kick dense reward 不应污染高层最终零和目标。

2.4 experiments/03_kick_and_router/

用途：Kick、支撑脚、真实接触、strength、Recovery、动作冻结和 router 接入。

包括：

  v4k1_* isolated/stable kick
  r8_support_skill
  r8_1--r8_59
  r860_router_smoke（若存在）

关键事实：

  - R8.42 是正式接受的无球完整 Kick 动作链父节点。
  - 接受的无球冻结目录：
    /scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244
  - R8.43--R8.49 的早期距离、球速、位移等代理不能单独证明真实接触。
  - 旧摆球曾把半径 0.11 m 的球心放在约 0.055 m，导致穿地；正确静止球心应约为
    ground_z + BALL_RADIUS + clearance，即约 0.112--0.115 m。
  - R8.50 改用 filtered ContactSensor 真实 FL-ball pair force。
  - R8.51 已建立约 96.9%--99.2% 的真实 FL 接触，但 strength 单调性和真实碰撞后
    Recovery 当时未通过。

R8.51 关键 run：

  /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/r851_kick64_15627683

2.5 experiments/04_curriculum_mixed_and_hard_only/

用途：R8.63 以后 V4D coverage、行为扫描、curriculum、mixed continuation、
fresh/hard-only、turn repair 和 R8.98 restore-native-command 训练线。

主要目录族：

  r8_63_full_support_v4d
  r8_64--r8_83
  r8_90_v4d_turn_phase_repair
  r8_95_v4d_receding_turn
  r8_96_v4d_turn_repair_oversample
  r8_97_turn_teacher_probe
  r8_98_restore_native_command_probe

必须区分两种 curriculum：

  A. 环境难度 curriculum
     easy -> intermediate -> hard_mix -> full_support
     由 R876_EASY_END、R876_INTERMEDIATE_END、R876_HARD_END 等控制。

  B. 优化阶段 schedule
     high_adapter -> joint_finetune -> stabilization
     由 V4D_STAGE1/2/3_ITERATIONS 和 high/low learning rate 控制。

两者互相独立。hard-only 仍可使用 staged optimization。

2.6 experiments/05_other_experiments/

用途：未落入上述四条主谱系、但为完整取证保留的实验源码。

2.7 current_v1_8_forensic/

用途：保留当前累计 V1.8 和补丁历史，供诊断、diff、抽取已验证实现。

禁止事项：

  - 不得把 V1.8 原样作为下一次 long training 的父代码。
  - 不得继续增加 reverse-turn/route-first/hold 状态机补丁。
  - 不得把 Chain V 视频成功等同于 PPO 学习成功。

V1.8 已知死锁机制：

  - route 指令同时包含后退、侧移和大角速度，超出已训练 low policy 的动作分布。
  - low policy 进入静态吸引子，route distance 不下降。
  - route=True 时 hold 被反复重置，失败状态永久锁存。
  - 到达 waypoint 后又可能切回纯旋转，引入第二个死锁。

2.8 local_runtime_package/

用途：与本机 V1.8 视频播放器配套的 project_source_current 副本。
它用于还原本机视频评估时的 import 路径，不保证与 Cluster 当前 checkout 完全相同。

2.9 submitted_jobs/

用途：保存曾位于 Git checkout 之外的实际提交脚本和补丁源码。

典型原始目录：

  /scratch/lj2634/incoming_r898_fix
  /scratch/lj2634/incoming_r898_fix_v2
  /scratch/lj2634/incoming_r898_fix_v3

其中应包含成功/失败实际使用过的 .sbatch，而不是根据聊天记录重新猜出的版本。

2.10 provenance/

用途：归档 manifest、Git 状态、选择的 run-side source snapshot、contract 和小型 summary。

重要文件：

  provenance/archive_meta/MANIFEST.tsv
  FILE_MANIFEST.tsv

FILE_MANIFEST.tsv 是分类后文件的 SHA256 / bytes / relative path 清单。

2.11 vendor_refs/isaaclab_api/

用途：仅保存与接触和环境更新相关的少量 Isaac Lab API 源码，例如：

  ContactSensor / ContactSensorCfg
  InteractiveScene
  ManagerBasedEnv

这不是完整 Isaac Lab 安装，不应直接作为 Python 环境使用。

======================================================================
3. 本机运行配置
======================================================================

3.1 操作系统用户与 Git 身份

  本机用户：ubuntu
  Git user.name：Lishengqi Jiang
  Git user.email：lj2634@nyu.edu

检查：

  git config --global --get user.name
  git config --global --get user.email

3.2 本机 Conda / Python 环境

  Conda 环境名：isaaclab232
  环境路径：/home/ubuntu/miniconda3/envs/isaaclab232
  Conda 初始化：/home/ubuntu/miniconda3/etc/profile.d/conda.sh

激活：

  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate isaaclab232

3.3 本机 Isaac Lab

  Isaac Lab 根目录：/home/ubuntu/IsaacLab_232
  启动器：/home/ubuntu/IsaacLab_232/isaaclab.sh

本机使用：

  /home/ubuntu/IsaacLab_232/isaaclab.sh -p TRAIN_SCRIPT [arguments]

本机视频渲染必须使用本机已经验证的 isaaclab.sh 链路，不要换成 Cluster direct-Python
或未经验证的新 evaluator。

3.4 本机常用运行文件

  V1.8/player：
  /home/ubuntu/go2_r898_video_16044032_gate045/source/train_r898.py

  project source：
  /home/ubuntu/go2_r898_video_16044032_gate045/project_source_current

  当前成功 hard-only repair 的本机 checkpoint 目录：
  /home/ubuntu/go2_r899_video_16063055_seed87701

  high checkpoint：
  /home/ubuntu/go2_r899_video_16063055_seed87701/logs/high_model_final.pt

  low checkpoint：
  /home/ubuntu/go2_r899_video_16063055_seed87701/logs/low_model_final.pt

  视频输出根目录：
  /home/ubuntu/下载/go2_soccer_phaseA_results/r898_video_bundle

3.5 本机 PYTHONPATH

标准形式：

  PLAYER=/home/ubuntu/go2_r898_video_16044032_gate045/source/train_r898.py
  PROJECT_SOURCE=/home/ubuntu/go2_r898_video_16044032_gate045/project_source_current

  export PYTHONPATH="$PROJECT_SOURCE/isaac_go2_low_level_rl:$PROJECT_SOURCE:$(dirname "$PLAYER"):${PYTHONPATH:-}"

若出现 ModuleNotFoundError，先检查上述两个目录，不要安装一个未知同名 pip package 来掩盖路径错误。

======================================================================
4. Cluster 连接、项目、环境和运行路径
======================================================================

4.1 登录节点

  lj2634@login.torch.hpc.nyu.edu

用途：交互检查、编辑、git、sbatch、squeue、sacct。

禁止：在登录节点直接运行 Isaac Sim、GPU 训练或长时间 Python 仿真。

4.2 数据传输节点（DTN）

  lj2634@dtn.torch.hpc.nyu.edu

用途：本机 <-> Cluster 的大文件传输；/scratch 在该节点可访问。

注意：材料中没有独立“DPU 训练路径”。如果“DPU”实际是指：

  - DTN：使用 dtn.torch.hpc.nyu.edu，仅做数据传输；
  - CPU：由 Slurm --cpus-per-task 分配；
  - 其他专用 DPU：当前交接没有配置或证据，使用前必须向集群管理员确认。

4.3 Cluster 用户主目录与 scratch

  用户：lj2634
  scratch 根：/scratch/lj2634

主要项目：

  /scratch/lj2634/Isaac-Go2-Low-Level-RL

训练输出：

  /scratch/lj2634/runs

临时 job runtime：

  /scratch/lj2634/job_runtime

外部提交源码：

  /scratch/lj2634/incoming_r898_fix*

源码交接归档：

  /scratch/lj2634/handoff_out/go2_soccer_source_through_v1_8_20260820_184110.tgz

4.4 Cluster Conda / Python

  环境根：/scratch/lj2634/envs/isaaclab232_v2
  Python：/scratch/lj2634/envs/isaaclab232_v2/bin/python

Cluster 已验证链路是直接调用该 Python，由 train_r898.py 内部的 AppLauncher 初始化 Isaac。
Cluster 不依赖 /home/ubuntu/IsaacLab_232/isaaclab.sh；本机和 Cluster 启动方式不能混用。

4.5 Cluster Isaac Lab / Isaac Sim 源码

  Isaac Lab 根：/scratch/lj2634/IsaacLab_232_src_37ddf62

  Headless experience：
  /scratch/lj2634/IsaacLab_232_src_37ddf62/apps/isaaclab.python.headless.kit

  Isaac Sim Python 包来自：
  /scratch/lj2634/envs/isaaclab232_v2/lib/python3.11/site-packages/isaacsim

已观察到节点系统为 RHEL 9.x、Kernel 5.14，并使用 NVIDIA L40S。历史日志中的
libusd_ts.so / GLIBC_2.35 warning 在部分成功 run 中并未阻止 AppLauncher 返回；必须以
是否进入 V4D/R8 iteration 和 PROCESS_RC 判断，不能仅凭该 warning 判任务失败。

4.6 Cluster PYTHONPATH

推荐 sbatch 内显式设置：

  PROJECT=/scratch/lj2634/Isaac-Go2-Low-Level-RL
  RUN_SOURCE=/实际父run/source

  export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:$PROJECT:$RUN_SOURCE:${PYTHONPATH:-}"

其中 task package 的实际嵌套结构是：

  $PROJECT/source/isaac_go2_low_level_rl/isaac_go2_low_level_rl/tasks

只写 $PROJECT/source 有时不足以解析 package。

======================================================================
5. CPU / GPU / Slurm 训练配置
======================================================================

5.1 已验证的资源形态

  Slurm account：torch_pr_1066_general
  GPU：NVIDIA L40S，1 张 GPU/job
  CPU：8 CPU/job（历史成功/诊断脚本常用值）
  内存：40G（部分任务曾因此不足；后续应按真实 MaxRSS 调整）
  GPU partition：历史输出显示 l40s_public（sacct 中可能被截断为 l40s_publ+）

以当前成功 .sbatch 中的 account/partition 为最终权威；README 不替代 sacctmgr 检查。

确认账户和 partition：

  sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS

5.2 标准 GPU sbatch 骨架

  #!/usr/bin/env bash
  #SBATCH --account=torch_pr_1066_general
  #SBATCH --partition=l40s_public
  #SBATCH --gres=gpu:1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=40G
  #SBATCH --time=01:00:00

  set -Eeuo pipefail

  PROJECT=/scratch/lj2634/Isaac-Go2-Low-Level-RL
  PY=/scratch/lj2634/envs/isaaclab232_v2/bin/python
  JOB_TMP=/scratch/lj2634/job_runtime/${SLURM_JOB_ID}

  mkdir -p "$JOB_TMP"/{tmp,xdg,matplotlib,omni}
  export TMPDIR="$JOB_TMP/tmp"
  export XDG_CACHE_HOME="$JOB_TMP/xdg"
  export MPLCONFIGDIR="$JOB_TMP/matplotlib"
  export OMNI_USER_HOME="$JOB_TMP/omni"
  export QT_QPA_PLATFORM=offscreen
  export PYTHONUNBUFFERED=1

  export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:$PROJECT:${PYTHONPATH:-}"

  "$PY" TRAIN_SCRIPT \
    --headless \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --num_envs NUM_ENVS \
    --seed SEED \
    --max_iterations ITERATIONS

不要把本机 /home/ubuntu 路径写进 Cluster sbatch。
不要在 Cluster 使用不存在的 /scratch/.../isaaclab.sh；Cluster 走 direct Python。

5.3 重要训练 run 路径

R8.98 总根：

  /scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe

成功 hard-fix long600：

  /scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_fix_long600_v2_16027021_20260819_125302

checkpoint：

  logs/high_model_final.pt
  logs/low_model_final.pt

none-curriculum/hard-only repair 成功任务：

  Job 16063055
  Job 16063056
  Job 16063042

对应 run 使用以下模式查找，不要猜目录尾号：

  find /scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe \
    -maxdepth 1 -type d \
    -name 'none_curriculum_fix_v1_1_long600_seed*160630*' \
    -print

5.4 训练成功判定

至少同时检查：

  sacct State=COMPLETED
  ExitCode=0:0
  PROCESS_RC=0
  出现 V4D_JOINT_TRAINING_COMPLETE
  出现 V4D_FINAL_GUARD
  stage2 low_lr 非零时 low_max_abs > 0
  high_model_final.pt / low_model_final.pt 存在且大小非零
  没有 V4D_FATAL / Traceback / RuntimeError / OOM / Killed

仅有 .pt 文件不等于训练有效；历史上 low_lr=0 时生成过最终文件但被 final guard 判定无效。

======================================================================
6. 本机视频链路 V / V4D（也称视频链路 V4）
======================================================================

6.1 链路性质

该链路是 checkpoint 推理、遥测和视频录制，不是训练。它输出：

  MP4 视频
  trajectory.csv
  summary.json
  console.log

只有看到 V4D_EVAL_VIDEO_OK 且上述文件存在，才表示视频链路完成。
是否进球、是否进入 push、是否静止需要结合 CSV 和 summary 判断；PLAYER_RC=0 仅表示程序结束。

6.2 固定本机路径

  ISAACLAB=/home/ubuntu/IsaacLab_232/isaaclab.sh
  PLAYER=/home/ubuntu/go2_r898_video_16044032_gate045/source/train_r898.py
  PROJECT_SOURCE=/home/ubuntu/go2_r898_video_16044032_gate045/project_source_current
  TRAINED_RUN=/home/ubuntu/go2_r899_video_16063055_seed87701
  HIGH=$TRAINED_RUN/logs/high_model_final.pt
  LOW=$TRAINED_RUN/logs/low_model_final.pt
  VIDEO_ROOT=/home/ubuntu/下载/go2_soccer_phaseA_results/r898_video_bundle

6.3 推荐基线评估命令

  source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
  conda activate isaaclab232

  ISAACLAB=/home/ubuntu/IsaacLab_232/isaaclab.sh
  PLAYER=/home/ubuntu/go2_r898_video_16044032_gate045/source/train_r898.py
  PROJECT_SOURCE=/home/ubuntu/go2_r898_video_16044032_gate045/project_source_current
  TRAINED_RUN=/home/ubuntu/go2_r899_video_16063055_seed87701
  HIGH=$TRAINED_RUN/logs/high_model_final.pt
  LOW=$TRAINED_RUN/logs/low_model_final.pt
  VIDEO_ROOT=/home/ubuntu/下载/go2_soccer_phaseA_results/r898_video_bundle
  PREFIX=R899_BASELINE_CHAIN_V_seed87701
  VIDEO=$VIDEO_ROOT/${PREFIX}.mp4
  CSV=$VIDEO_ROOT/${PREFIX}_trajectory.csv
  JSON=$VIDEO_ROOT/${PREFIX}_summary.json
  LOGDIR=$TRAINED_RUN/video_log_baseline

  mkdir -p "$VIDEO_ROOT" "$LOGDIR"

  export PYTHONPATH="$PROJECT_SOURCE/isaac_go2_low_level_rl:$PROJECT_SOURCE:$(dirname "$PLAYER"):${PYTHONPATH:-}"
  export V4D_HIGH_CHECKPOINT="$HIGH"

  export R876_HARD_GEOMETRY_V2=1
  export R876_HARD_REWARD_V2=1
  export R876_EASY_END=0
  export R876_INTERMEDIATE_END=0
  export R876_HARD_END=600

  export R898_STRATEGY_FIX=1
  export R898_ACTION_SCALE=0.75
  export R898_STAGING_PUSH_GATE=1
  export R898_STRICT_PUSH_GATE=1
  export R898_STAGING_MAX=0.45
  export R898_CONTACT_MIN=0.50
  export R898_CONTACT_MAX=0.80
  export R898_BODY_X_MIN=0.15
  export R898_GOAL_BODY_X_MIN=0.35
  export R898_CONTACT_LATERAL_MAX=0.30
  export R898_PUSH_HOLD_STEPS=8
  export R898_PUSH_HOLD_LATERAL_MAX=0.42
  export R898_PREPUSH_ALIGN=1
  export R898_PREPUSH_OFFSET=0.65
  export R898_PREPUSH_LIMIT=0.90
  export R898_PREPUSH_ROTATE_FIRST=0

  # 明确关闭已证伪的 V1.4--V1.8 reverse-route 状态机。
  export R898_REVERSE_TURN_FIRST=0
  export R898_ROUTE_DIAGNOSTIC=0

  export R873_HEADLESS_STRIP=1
  export QT_QPA_PLATFORM=offscreen
  export PYTHONUNBUFFERED=1

  "$ISAACLAB" -p "$PLAYER" \
    --headless \
    --enable_cameras \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --num_envs 1 \
    --seed 87701 \
    --max_iterations 1 \
    --log_dir "$LOGDIR" \
    --run_name "$PREFIX" \
    --eval_video "$VIDEO" \
    --eval_steps 1200 \
    --eval_fps 15 \
    --trajectory_csv "$CSV" \
    --eval_summary "$JSON" \
    --low_level_checkpoint "$LOW" \
    --trained_low_checkpoint "$LOW" \
    2>&1 | tee "$LOGDIR/console.log"

6.4 视频链路验收

  grep -E 'V4D_EVAL_VIDEO_OK|V4D_FATAL|Traceback' "$LOGDIR/console.log"
  ls -lh "$VIDEO" "$CSV" "$JSON"

若 console 出现 V4D_FATAL，视频/CSV/JSON 不齐，或 summary 没有对应字段，则评估无效。

======================================================================
7. 本机与 Cluster 文件互传
======================================================================

7.1 原则

  - 交互登录：login.torch.hpc.nyu.edu
  - 数据传输：dtn.torch.hpc.nyu.edu
  - 每次同时传文件和 .sha256。
  - 不使用 StrictHostKeyChecking=no。
  - login 主机可能负载均衡到不同后端；多次独立 scp 曾触发 host-key changed。
  - 大文件优先使用 DTN 或单一 SSH 会话。

7.2 Cluster -> 本机：单文件 + SHA

Cluster 生成 SHA：

  FILE=/scratch/lj2634/path/to/file.tgz
  sha256sum "$FILE" > "${FILE}.sha256"

本机下载：

  REMOTE=lj2634@dtn.torch.hpc.nyu.edu
  REMOTE_FILE=/scratch/lj2634/path/to/file.tgz
  LOCAL_DIR=/home/ubuntu/下载/target_dir

  mkdir -p "$LOCAL_DIR"
  scp "$REMOTE:$REMOTE_FILE" "$REMOTE:${REMOTE_FILE}.sha256" "$LOCAL_DIR/"

本机校验（只比较第一列，避免远端绝对路径影响）：

  LOCAL_FILE=$LOCAL_DIR/$(basename "$REMOTE_FILE")
  EXPECTED=$(awk 'NR==1{print $1}' "${LOCAL_FILE}.sha256")
  ACTUAL=$(sha256sum "$LOCAL_FILE" | awk '{print $1}')
  test "$EXPECTED" = "$ACTUAL" && echo SHA256_OK

7.3 Cluster -> 本机：单 SSH 会话传两个文件

用于避免重复 MFA：

  REMOTE=lj2634@dtn.torch.hpc.nyu.edu
  REMOTE_FILE=/scratch/lj2634/path/to/file.tgz
  REMOTE_DIR=$(dirname "$REMOTE_FILE")
  BASE=$(basename "$REMOTE_FILE")
  LOCAL_DIR=/home/ubuntu/下载/target_dir

  mkdir -p "$LOCAL_DIR"
  ssh "$REMOTE" \
    "cd '$REMOTE_DIR' && tar -cf - '$BASE' '${BASE}.sha256'" | \
    tar -xf - -C "$LOCAL_DIR"

7.4 本机 -> Cluster

  REMOTE=lj2634@dtn.torch.hpc.nyu.edu
  LOCAL_FILE=/home/ubuntu/下载/file.tgz
  REMOTE_DIR=/scratch/lj2634/incoming

先在登录节点创建目标目录：

  ssh lj2634@login.torch.hpc.nyu.edu "mkdir -p '$REMOTE_DIR'"

再通过 DTN 上传：

  sha256sum "$LOCAL_FILE" > "${LOCAL_FILE}.sha256"
  scp "$LOCAL_FILE" "${LOCAL_FILE}.sha256" "$REMOTE:$REMOTE_DIR/"

Cluster 校验：

  cd "$REMOTE_DIR"
  sha256sum -c "$(basename "$LOCAL_FILE").sha256"

7.5 同步代码目录

  rsync -av --progress \
    --exclude '.git/' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.mp4' \
    /home/ubuntu/local_code/ \
    lj2634@dtn.torch.hpc.nyu.edu:/scratch/lj2634/remote_code/

======================================================================
8. GitHub 内容与操作规则
======================================================================

8.1 仓库既有内容

main 分支原有报告仓库包括：

  assets/
  tools/reporting/
  README.md
  Report.md
  progress717.md

训练代码分支在此基础上新增：

  training_code/

8.2 training_code 中公开的内容

  本项目 source package
  初始 V4D
  Delta-Pose / high-low
  Kick / contact / recovery / router
  curriculum / mixed / hard-only / R8.98
  当前 V1.8 取证源码与历史备份
  实际提交过的 .sbatch
  run-side source snapshot / contract / 小型 summary
  少量 Isaac Lab API 参考源码
  SHA256 manifest 和本交接文档

8.3 不允许进入 GitHub 的内容

  *.pt
  *.pth
  *.ckpt
  *.onnx
  *.mp4 / *.avi
  巨型 v4d_joint_summary.json
  完整 Slurm 日志
  wandb 目录
  Conda 环境
  完整 Isaac Lab / Isaac Sim 安装
  SSH 私钥、GitHub token、设备认证 PIN

8.4 GitHub 状态核验

  REPO=/home/ubuntu/projects/go2-kick-progress-report
  BRANCH=training-code-v1.8-20260820-184110

  git -C "$REPO" branch --show-current
  git -C "$REPO" status --short
  git -C "$REPO" log -1 --oneline --decorate
  git -C "$REPO" remote -v
  git -C "$REPO" ls-remote --heads origin "$BRANCH"

ls-remote 有 commit hash 和 refs/heads/$BRANCH 才证明远端 push 完成。

8.5 后续提交

  cd /home/ubuntu/projects/go2-kick-progress-report
  git switch training-code-v1.8-20260820-184110
  git status --short
  git add -- training_code
  git commit -m "Describe change"
  git push

不要直接向 main force-push。需要合并时创建 Pull Request。

======================================================================
9. 后续 Codex 工作协议
======================================================================

每次让 Codex 接管本目录时，第一条指令建议为：

  “先阅读 training_code/README_LOCAL_CLUSTER_HANDOFF.txt、
   training_code/CODEX_START_HERE.md 和 training_code/FILE_MANIFEST.tsv。
   当前 V1.8 只用于取证；不要继续 reverse-turn/route-first 状态机。
   下一任务是建立干净的 staging-locomotion 训练分支，先审计再修改。”

新分支目标：

  任意初始机器人位置和朝向
  -> 计算动态球后方 staging pose
  -> locomotion 到达
  -> 面向球门
  -> 稳定保持
  -> 禁止提前碰球
  -> 验收后接回 push/kick

动态 staging 定义：

  p_stage = p_ball - d * normalize(p_goal - p_ball)

建议 d 约 0.60--0.70 m，但应作为配置和 curriculum 参数，而不是硬编码球的生成位置。

第一阶段建议：

  - 冻结已通过 locomotion regression 的 low policy。
  - 只训练 high navigation policy。
  - front/back/left/right x near/far 均衡 reset。
  - 奖励 staging distance progress、yaw progress、稳定保持。
  - 惩罚 premature ball contact、stall、fall 和动作抖动。
  - 先单 seed 短 canary，确认参数更新和分桶统计，再开长训。

验收目标：

  behind-near staging_success >= 90%
  behind-far staging_success >= 85%
  premature_ball_contact <= 5%
  stall <= 2%
  fall <= 1%

======================================================================
10. 常见错误与禁止重复验证的链路
======================================================================

1. 不要用距离壳、最近脚、球速变化替代真实 foot-ball contact。
2. 不要把球心放到低于 BALL_RADIUS 的高度。
3. 不要在真实碰撞成立前训练 Strike PPO；该问题已通过 R8.50/51 跨过。
4. 不要把 FL-ball force 当作 FL-ground touchdown/recovery 支撑力。
5. 不要继续生成 V1.4--V1.8 同类 router 视频来期待“学会”路径；视频不更新参数。
6. 不要把 Cluster 的 direct-Python 命令搬到本机视频链路。
7. 不要把本机 isaaclab.sh 路径搬到 Cluster。
8. 不要仅凭 checkpoint 文件存在判训练成功。
9. 不要在交互终端启用 set -euo pipefail 后粘贴诊断命令；一个非零 grep/test 会关闭终端。
10. 不要因历史源码尾随空格而机械重写全部文件；取证 snapshot 应保持源字节。
11. 不要在公开 GitHub 提交 checkpoint、视频、私钥、token 或设备 PIN。
12. 不要直接删除 known_hosts 或关闭 host-key checking 来绕过主机密钥变化；应使用 DTN 或联系 HPC 支持核验。

======================================================================
11. 快速健康检查
======================================================================

本机：

  test -d /home/ubuntu/projects/go2-kick-progress-report/.git
  test -d /home/ubuntu/projects/go2-kick-progress-report/training_code/src
  test -s /home/ubuntu/projects/go2-kick-progress-report/training_code/current_v1_8_forensic/train_r898.py
  test -x /home/ubuntu/IsaacLab_232/isaaclab.sh
  test -x /home/ubuntu/miniconda3/envs/isaaclab232/bin/python

Cluster：

  test -d /scratch/lj2634/Isaac-Go2-Low-Level-RL
  test -x /scratch/lj2634/envs/isaaclab232_v2/bin/python
  test -d /scratch/lj2634/IsaacLab_232_src_37ddf62
  test -d /scratch/lj2634/runs
  test -s /scratch/lj2634/handoff_out/go2_soccer_source_through_v1_8_20260820_184110.tgz

GitHub：

  git -C /home/ubuntu/projects/go2-kick-progress-report \
    ls-remote --heads origin training-code-v1.8-20260820-184110

======================================================================
12. 文档维护规则
======================================================================

以下变化必须同步更新本文件：

  - 本机/Cluster Isaac Lab 或 Python 环境改变；
  - task ID、project path、run root 改变；
  - 新 accepted checkpoint 或 parent freeze；
  - staging-locomotion 进入 canary、long training 或正式验收；
  - GitHub 分支重命名或合并；
  - 视频链路的 player、checkpoint 或 evaluator 改变；
  - 文件传输端点改变。

更新后运行：

  sha256sum training_code/README_LOCAL_CLUSTER_HANDOFF.txt
  git add training_code/README_LOCAL_CLUSTER_HANDOFF.txt
  git commit -m "Update local and cluster handoff documentation"
  git push

文档结束。
