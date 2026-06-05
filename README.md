# Dist-ERL: Distributed Evolutionary Reinforcement Learning

## 简介

本项目以 **Dist-ERL (`dist_erl`)** 作为论文主方法：在 ERL 框架中保留 EA 种群搜索与 RL actor-critic 学习的互补性，并把 EA 个体评估分发到多个 Ray worker 上，从而提升连续控制任务中的评估吞吐、样本采集效率和多种子实验可扩展性。

项目不再以 `dist_erl_re2` 为主线，也不再运行分布式 Re2 作为论文方法。`erl_re2` 仅保留为单 worker baseline，用于和已有 ERL-Re2 思路做对照。

**论文主旨：**

1. **分布式种群评估**：EA population 在多个 worker 上并行评估，主方法为 `dist_erl`。
2. **ERL 混合优化**：EA 负责全局搜索和精英保留，RL 负责梯度学习与 replay buffer 更新。
3. **低通信量日志化**：记录 `comm_upload_bytes` 与 `comm_full_traj_bytes`，突出只回传 seed/fitness 相比上传完整轨迹的通信优势。
4. **稳定性诊断**：记录 `weight_diversity`、`fitness_std`、`eval_ea_mean`、`eval_reward_mean`，用于分析种群多样性和 RL/EA 互补关系。

---

## 训练模式

| 模式 | EA | RL | Re2 | 分布式评估 | 论文定位 |
|------|:--:|:--:|:---:|:----------:|----------|
| `pure_rl` | - | yes | - | - | RL baseline |
| `pure_ea` | yes | - | - | optional | EA baseline |
| `standard_erl` | yes | yes | - | single worker | ERL baseline |
| `erl_re2` | yes | yes | yes | single worker | ERL-Re2 baseline |
| `dist_erl` | yes | yes | - | yes | **主方法** |

`dist_erl_re2` 已废弃，不再作为新实验模式、论文方法或默认脚本目标。

---

## 安装

```bash
conda activate dist-erl-re2
pip install -r requirements.txt
pip install swig "gymnasium[box2d,mujoco]"
```

说明：当前服务器已有 conda 环境仍叫 `dist-erl-re2`。代码和项目名已切到 Dist-ERL；环境名可后续另行克隆/重命名，不影响运行。

---

## 快速运行

```bash
cd ~/code/Dist-ERL
./run_dist_erl.sh --env Hopper-v2 --mode dist_erl --max-generations 50
```

默认入口即为主方法：

```bash
python -m src.main --env Hopper-v2
```

---

## 论文级实验

### 1. 多种子主实验

```bash
chmod +x run_dist_erl.sh run_seeds.sh run_scaling.sh
./run_seeds.sh
```

默认比较 5 个模式：

```text
pure_rl pure_ea standard_erl erl_re2 dist_erl
```

### 2. 单环境快速基准

```bash
ENV_NAME=Ant-v2 ./run_benchmarks.sh
```

### 3. Dist-ERL 扩展性与带宽

```bash
./run_scaling.sh
python3 scripts/plot_scaling_bandwidth.py --log-dir logs
```

### 4. 出图

```bash
python3 generate_plots.py --log-dir logs --require-real
```

禁止用无日志合成曲线作为实验结果。

---

## MuJoCo 主实验环境

与常见 ERL / ERL-Re2 文献一致，论文任务名使用 Gym `*-v2`：

| 任务 | `ENV_NAME` |
|------|------------|
| HalfCheetah | `HalfCheetah-v2` |
| Swimmer | `Swimmer-v2` |
| Hopper | `Hopper-v2` |
| Ant | `Ant-v2` |
| Walker2d | `Walker2d-v2` |
| Humanoid | `Humanoid-v2` |

配置见 `src/config.py` 中 `MUJOCO_V2_ENVS` 与 `env_run_preset()`。运行时通过 `src/utils/environment.py` 映射到本机 Gymnasium 可用版本。

---

## 指标含义

| 指标 | 含义 |
|------|------|
| `eval_reward_mean` | RL actor 的确定性评估；`pure_ea` 下为最佳 EA 个体评估 |
| `eval_ea_mean` | EA 精英多回合评估 |
| `best_fitness` / `mean_fitness` | EA 种群在 worker 上评估得到的最佳/平均适应度 |
| `weight_diversity` | 种群 actor 权重余弦多样性 |
| `fitness_std` | 种群适应度标准差 |
| `comm_upload_bytes` | 每代回传 seed/fitness 的近似通信量 |
| `comm_full_traj_bytes` | 假想上传完整轨迹的通信量 |

---

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `dist_erl` | 训练模式，默认即主方法 |
| `--algorithm` | `DDPG` | RL 算法（`DDPG` / `TD3` / `PPO`） |
| `--num-workers` | `4` | Ray rollout workers 数量 |
| `--population-size` | `50` | EA 种群大小 |
| `--rl-rollouts` | `2` | 每代 RL rollout 数 |
| `--rl-updates` | `10` | 每代/同步点梯度更新步数 |
| `--elite-fraction` | `0.2` | 未显式设 `--num-elitists` 时的精英比例 |
| `--num-elitists` | `1` | EA 精英数量 |
| `--policy-exploration-noise` | `0.1` | DDPG/TD3 采样时动作噪声 |
| `--stagnation-patience` | `12` | 停滞后触发 EA 移民和探索增强 |

Re2 相关参数仍保留给 `erl_re2` baseline，但不构成 Dist-ERL 主方法。

---

## 项目结构

```text
Dist-ERL/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── training.py
│   ├── manager.py
│   ├── learner.py
│   ├── worker.py
│   └── utils/
├── run_dist_erl.sh
├── run_seeds.sh
├── run_scaling.sh
├── run_benchmarks.sh
├── generate_plots.py
└── scripts/plot_scaling_bandwidth.py
```

---

## 引用

- Khadka & Tumer (2018). Evolutionary Reinforcement Learning. NeurIPS.
- Wan et al. (2022). ERL-Re2: Efficient Evolutionary Reinforcement Learning with Reproducible and Reusable Experience.
