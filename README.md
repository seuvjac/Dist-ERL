# FedEvoRL: Evolutionary Federated Reinforcement Learning

## 1. 项目定位

FedEvoRL 是一个面向异质连续控制任务的 **EA-guided Federated Reinforcement Learning** 框架。项目主线已经从普通分布式 ERL 切换为：

```text
Evolutionary Algorithm + Federated Reinforcement Learning
```

核心问题是：多个客户端处在不同环境动力学、奖励尺度、噪声水平或数据分布下，不能共享本地 trajectory，只能上传模型参数和标量摘要。传统 Federated RL 直接做 FedAvg 时，容易把多个局部有效策略平均成一个在所有客户端上都一般甚至失效的策略。FedEvoRL 用 EA policy population 保持策略多样性，用 federated RL 聚合吸收客户端本地梯度改进，从而缓解异质 MDP 下的平均化退化。

`fed_evo_rl` 是当前论文主方法。`dist_erl`、`standard_erl`、`erl_re2`、`pure_rl`、`pure_ea` 均作为 baseline。

---

## 2. 研究动机

### 2.1 当前方向为什么要从 Dist-ERL 改为 FedEvoRL

单纯的“分布式 EA + RL”与已有学界工作重合较高：ERL 已经提出 EA population 与 actor-critic RL 的混合训练；ES/EA 天然适合分布式评估；很多分布式 RL 系统也已经研究大规模采样和训练。因此，如果论文只强调 Ray 并行评估 EA population，创新性容易被认为不足。

FedEvoRL 改变问题设定：

- 从“单机/集群加速训练”转向“多客户端隐私约束下的联邦强化学习”；
- 从“worker 只是并行采样器”转向“client 拥有私有环境和本地 replay buffer”；
- 从“一个全局平均策略”转向“服务器维护策略种群，兼顾全局泛化与个性化适配”；
- 从“上传轨迹或集中训练”转向“只上传模型权重、fitness 和统计摘要”。

### 2.2 FedAvg 在 Federated RL 中的问题

Federated RL 比监督学习下的 FL 更难，因为每个客户端的数据分布由本地 policy 与本地 MDP 共同决定。客户端之间可能存在：

- 不同动力学：mass、friction、gravity、actuator strength 不同；
- 不同奖励尺度或奖励偏好；
- 不同探索噪声；
- 不同初始状态分布；
- 不同 episode horizon。

直接平均 actor/critic 参数有两个风险：

1. **Policy averaging collapse**：两个局部好策略的参数均值不一定对应一个好策略。
2. **Non-IID critic bias**：不同客户端 critic 估计的 value landscape 不一致，简单聚合会互相干扰。

FedEvoRL 的出发点是用 EA population 承接这种异质性：让服务器保留多个 candidate policies，而不是强迫所有客户端收敛到唯一平均策略。

---

## 3. 方法概述

FedEvoRL 包含三个角色：

| 角色 | 文件 | 职责 |
|------|------|------|
| Federated Evolution Server | `src/main.py` + `src/manager.py` | 维护 EA population，聚合 client 上传模型，执行选择/交叉/变异 |
| Federated RL Client | `src/federated.py` | 持有私有环境与 replay buffer，本地评估和训练策略 |
| Baseline Learner/Worker | `src/learner.py` / `src/worker.py` | 支撑 `dist_erl`、`standard_erl`、`erl_re2` 等 baseline |

主模式：

```bash
python -m src.main --mode fed_evo_rl
```

---

## 4. FedEvoRL 算法流程

### 4.1 高层伪代码

```text
Server initializes an EA policy population P = {theta_1, ..., theta_N}
Server starts M federated clients C = {c_1, ..., c_M}

for each communication round t:
    # Client-side private evaluation
    for each policy theta_i in P:
        for each client c_j:
            c_j evaluates theta_i in its private MDP
            c_j returns scalar fitness only

    # Server-side evolutionary search
    Server aggregates client fitness for each theta_i
    Server ranks population by cross-client fitness
    Server applies elitism, tournament selection, crossover, mutation

    # Client-side local RL
    Server selects a global/elite policy theta_star
    Server samples a subset of clients
    Each selected client:
        loads theta_star
        collects private trajectories locally
        updates actor-critic with local replay buffer
        uploads model weights and scalar summaries

    # Federated aggregation
    Server aggregates uploaded client models
    Aggregation can be uniform or fitness-weighted
    Aggregated policy is injected back into EA population

    # Logging
    Record client reward, EA fitness, diversity, communication, wall-clock
```

### 4.2 当前实现

当前 `fed_evo_rl` 的一轮训练做以下事情：

1. Server 初始化 EA population。
2. 每个 federated client 在自己的私有环境流上评估所有 EA candidate。
3. Server 用跨客户端平均 fitness 更新 population。
4. Server 对 population 做 EA evolution。
5. Server 取当前 best individual 下发给部分 clients。
6. Clients 本地 rollout、本地 actor-critic 更新。
7. Clients 上传模型权重与本地 reward 摘要。
8. Server 用 `fitness` 或 `uniform` aggregation 聚合模型。
9. Server 将聚合模型软注入 EA population。

代码入口：

- `src/federated.py`
  - `FederatedClient`
  - `aggregate_weight_dicts`
  - `weight_entropy`
- `src/main.py`
  - `_run_fed_evo_rl`

---

## 5. 联邦异质性建模

当前代码用轻量、稳定的方式模拟 client heterogeneity：

| 异质性来源 | 实现 |
|------------|------|
| reward scale | 每个 client 使用不同 reward scale |
| action noise | 每个 client 使用不同 action noise |
| rollout seed stream | 每个 client 使用不同 seed offset |
| private replay buffer | 每个 client 独立保存本地 buffer |

参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `--num-clients` | `4` | 联邦客户端数量 |
| `--client-fraction` | `1.0` | 每轮参与本地训练的客户端比例 |
| `--client-rollouts` | `2` | 每个选中客户端本地 rollout 数 |
| `--client-updates` | `10` | 每个选中客户端本地梯度更新步数 |
| `--client-heterogeneity` | `0.2` | 客户端异质性强度 |
| `--fed-aggregation` | `fitness` | 聚合方式：`fitness` 或 `uniform` |

后续论文实验可以进一步把异质性升级为真实 MuJoCo 参数扰动，例如 gravity、body mass、friction、actuator gear 等。

---

## 6. EA 组件

EA population 中每个 individual 包含：

| 字段 | 含义 |
|------|------|
| `id` | 个体编号 |
| `weights` | actor/critic 参数 |
| `seed` | 评估种子 |
| `fitness` | 跨客户端聚合后的适应度 |

EA evolution 使用：

1. fitness 排序；
2. 精英保留；
3. 锦标赛选择；
4. actor 权重 row-wise crossover；
5. mutation / super mutation / reset mutation；
6. 多样性统计 `weight_diversity`。

核心作用：

- 避免所有客户端策略被 FedAvg 拉向单一平均点；
- 在策略参数空间保留多个 candidate；
- 为异质客户端提供可选择、可演化的策略族；
- 将 federated RL 聚合结果作为高质量 genetic material 注入 population。

---

## 7. Federated RL 组件

每个 `FederatedClient` 持有：

- private environment stream；
- private replay buffer；
- local actor-critic policy；
- local optimizer；
- local reward/noise profile。

Client 不上传 trajectory。上传内容包括：

| 上传内容 | 用途 |
|----------|------|
| local model weights | server 聚合 |
| local avg reward | fitness-aware aggregation |
| buffer size / training steps | 日志诊断 |
| reward scale / action noise | 异质性记录 |

当前聚合方式：

```text
uniform:
    w_j = 1 / M

fitness:
    w_j = (score_j - min(score)) / sum(score - min(score))
```

当所有 client 得分相同或异常时，自动退化为 uniform aggregation。

---

## 8. 训练模式

| 模式 | EA | RL | Federated | Re2 | 定位 |
|------|:--:|:--:|:---------:|:---:|------|
| `pure_rl` | - | yes | - | - | RL baseline |
| `pure_ea` | yes | - | - | - | EA baseline |
| `standard_erl` | yes | yes | - | - | ERL baseline |
| `dist_erl` | yes | yes | - | - | distributed ERL baseline |
| `erl_re2` | yes | yes | - | yes | ERL-Re2 baseline |
| `fed_evo_rl` | yes | yes | yes | - | **主方法** |

---

## 9. 快速运行

```bash
cd ~/code/Dist-ERL
./run_fed_evo_rl.sh --env Hopper-v2 --max-generations 50
```

等价命令：

```bash
python -m src.main \
  --mode fed_evo_rl \
  --env Hopper-v2 \
  --num-clients 4 \
  --population-size 40 \
  --max-generations 50
```

轻量 smoke test：

```bash
./run_fed_evo_rl.sh \
  --env Pendulum-v1 \
  --population-size 6 \
  --num-clients 2 \
  --max-generations 3 \
  --max-episode-steps 100 \
  --batch-size 32 \
  --client-rollouts 1 \
  --client-updates 2
```

---

## 10. 论文级实验设计

### 10.1 主实验

```bash
./run_seeds.sh
```

默认比较：

```text
pure_rl pure_ea standard_erl erl_re2 dist_erl fed_evo_rl
```

### 10.2 单环境快速 benchmark

```bash
ENV_NAME=Ant-v2 ./run_benchmarks.sh
```

### 10.3 Client scaling

```bash
./run_scaling.sh
python3 scripts/plot_scaling_bandwidth.py --log-dir logs
```

### 10.4 出图

```bash
python3 generate_plots.py --log-dir logs --require-real
```

---

## 11. 推荐消融实验

| 消融 | 目的 |
|------|------|
| `fed_evo_rl` vs `dist_erl` | 证明不是普通分布式 ERL 即可解决异质联邦问题 |
| `--fed-aggregation uniform` | 比较 fitness-aware aggregation 的作用 |
| `--client-heterogeneity 0.0/0.2/0.5` | 分析非 IID 强度 |
| `--client-fraction 0.25/0.5/1.0` | 分析部分客户端参与 |
| no EA injection | 验证 EA population 对抗 policy averaging collapse 的作用 |
| no local RL | 验证客户端本地梯度学习的作用 |

---

## 12. 日志指标

每次实验会在 `logs/<exp-name>/` 下生成：

```text
metadata.json
metrics.csv
```

关键指标：

| 指标 | 含义 |
|------|------|
| `eval_reward_mean` | federated clients 本地训练后的平均 reward |
| `client_reward_mean` | 选中 clients 的本地 reward 均值 |
| `client_reward_std` | 选中 clients 的本地 reward 标准差 |
| `client_fitness_mean` | EA candidates 跨客户端评估 fitness 均值 |
| `client_fitness_std` | EA candidates 跨客户端评估 fitness 标准差 |
| `best_fitness` | EA population 当前最佳跨客户端 fitness |
| `mean_fitness` | EA population 平均 fitness |
| `weight_diversity` | actor 权重多样性 |
| `selected_clients` | 当前 federated round 参与本地训练的客户端数量 |
| `aggregation_entropy` | 聚合权重熵，反映是否由少数 client 主导 |
| `comm_upload_bytes` | 模型上传 + seed/fitness 摘要的估算通信量 |
| `comm_full_traj_bytes` | 假想上传完整轨迹的通信量 |

---

## 13. 与已有工作的差异

FedEvoRL 与传统 ERL 的区别：

- ERL 假设集中式训练或共享交互数据；
- FedEvoRL 假设多个私有客户端，trajectory 不出本地。

FedEvoRL 与普通 Distributed ERL 的区别：

- Distributed ERL 的 worker 通常只是并行 evaluator；
- FedEvoRL 的 client 是拥有私有 MDP、私有 buffer、本地 learner 的学习主体。

FedEvoRL 与普通 Federated RL 的区别：

- 普通 Federated RL 多数维护单一 global policy；
- FedEvoRL 维护 EA policy population，以多样性缓解异质 MDP 下的平均化退化。

FedEvoRL 与 ERL-Re2 的区别：

- ERL-Re2 强调经验重现与共享表示；
- FedEvoRL 强调隐私约束、客户端异质性、fitness-aware aggregation 和 policy population。

---

## 14. 当前实现状态

已实现：

- `fed_evo_rl` 主模式；
- federated client remote actor；
- client 本地 replay buffer 和 actor-critic 更新；
- cross-client EA fitness evaluation；
- fitness-aware / uniform model aggregation；
- aggregated policy 注入 EA population；
- client reward、fitness、aggregation entropy、通信指标日志；
- baseline 模式保留。

后续可增强：

- 真实 MuJoCo 物理参数异质性；
- personalized policy assignment；
- clustered federated aggregation；
- actor/critic 分离聚合；
- privacy/noise-aware aggregation；
- FedProx / SCAFFOLD 风格校正项；
- no-EA / no-Fed / no-local-RL 系统消融。

---

## 15. 安装

```bash
conda activate dist-erl-re2
pip install -r requirements.txt
pip install swig "gymnasium[box2d,mujoco]"
```

说明：当前服务器已有 conda 环境仍叫 `dist-erl-re2`，暂未重命名；这不影响 FedEvoRL 代码运行。

---

## 16. 项目结构

```text
Dist-ERL/
├── src/
│   ├── main.py                  # FedEvoRL 主循环与 baseline 训练入口
│   ├── federated.py             # federated clients and aggregation
│   ├── manager.py               # EA population manager
│   ├── learner.py               # baseline RL learner
│   ├── worker.py                # baseline distributed rollout worker
│   ├── config.py                # modes, env presets, plot styles
│   └── utils/
├── run_fed_evo_rl.sh            # FedEvoRL 主启动脚本
├── run_dist_erl.sh              # 兼容旧名的启动脚本
├── run_seeds.sh                 # 多环境多种子实验
├── run_scaling.sh               # client scaling
├── run_benchmarks.sh            # 快速 benchmark
└── generate_plots.py
```

---

## 17. 引用方向

- Khadka & Tumer (2018). Evolution-Guided Policy Gradient in Reinforcement Learning.
- Pourchot & Sigaud (2018). CEM-RL: Combining Evolutionary and Gradient-Based Methods for Policy Search.
- Bodnar et al. (2020). Proximal Distilled Evolutionary Reinforcement Learning.
- Wan et al. (2022). ERL-Re2.
- Jin et al. (2022). Federated Reinforcement Learning.
- Recent work on federated actor-critic and heterogeneous federated RL.
