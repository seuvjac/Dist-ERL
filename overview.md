# FedEvoRL / Dist-ERL

## 1. 总体框架 (Framework Overview)

### 1.1 问题定义：异质联邦强化学习下的控制任务

本项目面向 **heterogeneous Federated Reinforcement Learning** 场景，核心问题是：

多个客户端分别处在不同的控制环境中，每个客户端拥有自己的私有交互数据、私有 replay buffer 和局部环境扰动，但不能上传 trajectory。服务器只能接收模型参数、fitness 和少量标量统计信息。

传统 Federated RL 往往维护单一全局策略，并通过 FedAvg 聚合客户端模型。然而在 RL 中，客户端数据分布由本地 MDP 与当前 policy 共同决定，不同客户端之间可能存在明显的 non-IID 差异：

*   **动力学差异**：mass、friction、gravity、actuator strength 等物理参数不同。
*   **奖励尺度差异**：不同客户端 reward scale 不一致。
*   **探索差异**：客户端 action noise、seed stream 和状态分布不同。
*   **局部经验差异**：每个客户端只维护自己的 replay buffer。

因此，直接平均参数容易产生两个问题：

1.  **Policy averaging collapse**：多个局部有效策略的参数均值不一定仍是有效策略。
2.  **Non-IID critic bias**：不同客户端 critic 学到的 value landscape 不一致，简单聚合会互相干扰。

FedEvoRL 将问题建模为：

```text
Evolutionary Policy Population
+ Federated Local RL
+ Global Elite Archive
+ Heterogeneous Client MDPs
```

服务器不强迫所有客户端收敛到唯一策略，而是维护一个 EA policy population，用种群多样性承接异质 MDP 下的多模态最优策略。

---

## 2. 方法主线：EA-guided Federated Reinforcement Learning

### 2.1 三类角色

FedEvoRL 包含三个核心角色：

| 角色 | 代码位置 | 职责 |
|------|----------|------|
| Federated Evolution Server | `src/main.py`, `src/manager.py` | 维护 EA 种群，执行评估、选择、交叉、变异和 RL 注入 |
| Federated RL Client | `src/federated.py` | 持有私有环境与 replay buffer，执行本地评估和本地 actor-critic 更新 |
| Baseline Learner / Worker | `src/learner.py`, `src/worker.py` | 支撑 pure RL、pure EA、standard ERL、ERL-Re2、Dist-ERL 等对比算法 |

主方法入口：

```bash
python -m src.main --mode fed_evo_rl
```

### 2.2 算法流程

每一轮 communication / evolution round 中，FedEvoRL 执行：

1.  **客户端私有评估**
    *   服务器将 EA population 中的 candidate policies 下发到各客户端。
    *   每个客户端在自己的私有 MDP 上评估每个 policy。
    *   客户端只返回 scalar fitness，不上传 trajectory。

2.  **服务器端进化**
    *   服务器聚合跨客户端 fitness。
    *   根据平均 fitness 排序 population。
    *   执行 elitism、tournament selection、crossover、mutation。

3.  **客户端本地 RL**
    *   服务器选择当前 best / elite policy 下发给一部分客户端。
    *   客户端在本地 rollout，写入私有 replay buffer。
    *   客户端执行若干步 actor-critic 更新。

4.  **联邦聚合**
    *   客户端上传本地更新后的模型权重和 reward 摘要。
    *   服务器默认每 `K=5` 代执行一次聚合，降低通信和 wall-clock 成本。
    *   默认使用 softmax fitness aggregation，并过滤低分客户端上传。
    *   聚合后的模型只有在分数接近当前 EA best 时才作为 genetic material 注入 EA population。

5.  **全局 elite archive**
    *   服务器维护全局 top-k elite policies。
    *   每代进化和 RL 注入后，将 archive 中的高分策略重新 pin 回 active population。
    *   防止高分策略在后续 mutation、aggregation 或 RL 注入中丢失。

6.  **日志记录**
    *   记录 reward、fitness、diversity、通信量、客户端参与数和 aggregation entropy。

---

## 3. 核心机制

### 3.1 EA Policy Population：缓解策略平均化坍塌

FedEvoRL 的核心不是维护单个 global policy，而是维护一组 candidate policies：

$$
P = \{\theta_1, \theta_2, \dots, \theta_N\}
$$

每个 individual 包含：

| 字段 | 含义 |
|------|------|
| `id` | 个体编号 |
| `weights` | actor / critic 参数 |
| `seed` | 评估 seed |
| `fitness` | 跨客户端聚合后的适应度 |

种群的作用是：

*   保留多个异质客户端下可能有效的策略模式。
*   避免 FedAvg 将多个局部策略平均成低质量中间策略。
*   为不同客户端提供可演化的策略材料。
*   将本地 RL 学到的梯度信息转化为可被 EA 利用的 genetic material。

### 3.2 Federated Client：私有环境与私有 replay buffer

每个客户端是一个独立学习主体，而不是普通分布式采样器。

客户端拥有：

*   private environment stream；
*   private replay buffer；
*   local actor-critic policy；
*   local optimizer；
*   reward scale / action noise / seed offset。

客户端上传内容仅包括：

| 上传内容 | 用途 |
|----------|------|
| local model weights | 用于服务器聚合 |
| local avg reward | 用于 fitness-aware aggregation |
| buffer size / training steps | 诊断本地学习状态 |
| reward scale / action noise | 记录异质性配置 |

这使得 FedEvoRL 更接近 Federated RL 问题设定，而不是普通 distributed ERL。

### 3.3 Fitness-aware Aggregation

FedEvoRL 支持三种聚合方式：

```text
uniform:
    w_j = 1 / M

fitness:
    w_j = (score_j - min(score)) / sum(score - min(score))

softmax:
    w_j = exp(score_j / T) / sum_k exp(score_k / T)
```

当客户端分数完全相同或出现异常时，自动退化为 uniform aggregation。

默认配置使用 `softmax`，并增加以下稳定化策略：

| 机制 | 参数 | 目的 |
|------|------|------|
| 间隔聚合 | `--fed-aggregation-interval 5` | 降低通信与本地训练成本 |
| 温度控制 | `--fed-aggregation-temperature 75` | 防止最高分客户端过度支配 |
| 低分过滤 | `--fed-min-client-score-quantile 0.25` | 过滤低质量本地更新 |
| 注入门控 | `--fed-inject-margin -0.05` | 避免明显劣质聚合 actor 污染 EA |

fitness-aware / softmax aggregation 的直觉是：

*   表现更好的客户端更新权重更大；
*   低质量本地更新不会过度影响全局模型；
*   在 non-IID 环境中，让聚合更接近跨客户端有效策略；
*   通过温度和间隔控制降低 policy averaging collapse 风险。

### 3.4 Global Elite Archive

当前数据表明，算法有时能找到高分策略，但后续训练不能稳定保留。为此新增 global elite archive：

```text
archive = top-k policies across historical generations
```

每代执行：

1.  在当前 population 评估后更新 archive。
2.  进化后将 archive 中最优个体恢复到 active population。
3.  RL 聚合注入后再次恢复 archive elite，避免高分策略被覆盖。

关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--elite-archive-size` | `5` | 保留历史 top-k 策略 |
| `--elite-archive-restore-copies` | `1` | 每代 pin 回 population 的 archive elite 数 |

### 3.5 Soft EA Injection

本地 RL 聚合得到的模型不会直接替换 elite，而是以 soft injection 的方式注入弱 non-elite individuals：

$$
\theta_{new} = (1 - \alpha)\theta_{old} + \alpha\theta_{RL} + \epsilon
$$

其中：

*   $\alpha$ 对应 `migration_blend`；
*   $\epsilon$ 是 actor 权重上的注入噪声；
*   注入位置避开 elite，优先更新弱个体。

该机制的作用是：

1.  利用 RL 梯度更新带来的局部改进。
2.  避免 hard replacement 导致种群多样性快速坍塌。
3.  将客户端学习结果作为 EA 的搜索材料，而不是直接支配整个 population。

### 3.6 异质客户端环境

除 MuJoCo locomotion 外，当前实现新增了 FedRL 文献中常见的轻量异质环境：

| 环境 | 动作类型 | 异质性来源 |
|------|----------|------------|
| `CartPole-v1` | discrete | gravity、cart mass、pole mass、pole length |
| `Acrobot-v1` | discrete | link length、link mass、center of mass、gravity |
| `LunarLander-v3` | discrete | gravity、wind、turbulence |
| `LunarLanderContinuous-v3` | continuous | gravity、wind、turbulence |
| `BipedalWalkerHardcore-v3` | continuous | terrain seed stream、observation/action/reward perturbation |

所有 client 仍然只上传模型权重和标量摘要，不上传 trajectory。

---

## 4. Baseline 与相似算法比较

为了证明 FedEvoRL 的贡献不是来自单一组件，项目保留了多种相似算法作为 baseline：

| 模式 | EA | RL | Federated | Re2 | 定位 |
|------|:--:|:--:|:---------:|:---:|------|
| `pure_rl` | - | yes | - | - | 纯 actor-critic RL baseline |
| `pure_ea` | yes | - | - | - | 纯进化搜索 baseline |
| `standard_erl` | yes | yes | - | - | 单机 ERL baseline |
| `dist_erl` | yes | yes | - | - | 分布式 ERL baseline |
| `erl_re2` | yes | yes | - | yes | ERL-Re2 baseline |
| `fed_evo_rl` | yes | yes | yes | - | 本项目主方法 |

主对比图要求至少 4 条曲线，本项目默认使用 6 条：

```text
FedEvoRL, Dist-ERL, ERL-Re2, Standard ERL, Pure EA, Pure RL
```

对于 FedRL 异质环境实验，主图优先比较 FedEvoRL 家族内部曲线，并叠加 Stable-Baselines3 强 RL baseline：

```text
FedEvoRL-full
FedEvoRL-uniform_aggregation
FedEvoRL-no_local_rl
FedEvoRL-no_ea_injection
FedEvoRL-no_heterogeneity
SB3-PPO
SB3-SAC
SB3-TD3
```

说明：Stable-Baselines3 不是联邦强化学习算法库，这里将其作为标准单智能体 RL baseline。离散动作环境只使用 `SB3-PPO`；连续动作环境使用 `SB3-PPO/SAC/TD3`。

---

## 5. 消融实验设计

### 5.1 FedEvoRL 组件消融

项目新增了 FedEvoRL 专用消融参数：

```bash
--fed-ablation full
--fed-ablation uniform_aggregation
--fed-ablation no_local_rl
--fed-ablation no_ea_injection
--fed-ablation no_heterogeneity
```

各消融含义如下：

| 消融项 | 实际改动 | 目的 |
|--------|----------|------|
| `full` | 完整 FedEvoRL | 主方法 |
| `uniform_aggregation` | 将 fitness aggregation 改为 uniform | 验证 fitness-aware aggregation 的作用 |
| `no_local_rl` | `client_updates = 0` | 验证客户端本地梯度学习的作用 |
| `no_ea_injection` | `migration_copies = 0` | 验证聚合模型注入 EA population 的作用 |
| `no_heterogeneity` | `client_heterogeneity = 0.0` | 分析 IID / non-IID 差异 |

消融图默认包含 5 条曲线，满足至少 4 条曲线的论文绘图要求。

### 5.2 Re2 Baseline 消融

项目也保留 ERL-Re2 baseline 的机制消融：

| 消融项 | 含义 |
|--------|------|
| `no_re2` | 移除 Re2 机制 |
| `no_reproduction` | 移除 elite trajectory reproduction |
| `no_migration` | 移除 RL-to-EA migration |
| `full` | 完整 ERL-Re2 baseline |

这部分用于解释 baseline 行为，而不是 FedEvoRL 主贡献。

---

## 6. MuJoCo 长线实验

### 6.1 主对比实验

长线 MuJoCo 对比脚本：

```bash
./run_mujoco_long_compare.sh
```

默认配置：

```text
env: Hopper-v2
seeds: 0 1 2
modes: fed_evo_rl dist_erl erl_re2 standard_erl pure_ea pure_rl
generations: 使用 src/config.py 中的 MuJoCo preset
```

输出日志：

```text
logs/<exp-name>/metadata.json
logs/<exp-name>/metrics.csv
```

### 6.2 FedEvoRL 消融实验

消融脚本：

```bash
./run_fed_evo_ablation_mujoco.sh
```

默认配置：

```text
env: Hopper-v2
seeds: 0 1 2
fed_ablations: full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity
```

### 6.3 后台运行

由于实验时间较长，推荐使用 tmux：

```bash
tmux new-session -d -s dist_erl_compare "cd ~/code/Dist-ERL && ./run_mujoco_long_compare.sh | tee logs/background/tmux_compare.out"
tmux new-session -d -s dist_erl_ablation "cd ~/code/Dist-ERL && ./run_fed_evo_ablation_mujoco.sh | tee logs/background/tmux_ablation.out"
```

查看运行状态：

```bash
tmux list-sessions
tail -f logs/background/tmux_compare.out
tail -f logs/background/tmux_ablation.out
```

---

## 7. FedRL 异质环境实验

### 7.1 环境矩阵

FedRL 异质环境套件：

```text
CartPole-v1
Acrobot-v1
LunarLander-v3
LunarLanderContinuous-v3
BipedalWalkerHardcore-v3
```

这些环境用于模拟 FedRL 文献中的 client heterogeneity，例如 gravity、mass、pole length、link length、wind、terrain distribution 等差异。

### 7.2 运行脚本

```bash
./run_fedrl_heterogeneous_suite.sh
```

默认配置：

```text
seeds: 0 1 2
fed_ablations: full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity
sb3_algos: PPO SAC TD3
```

输出：

```text
logs_fedrl_hetero/<exp-name>/metrics.csv
logs_sb3/<exp-name>/metrics.csv
plots/fedrl_heterogeneous/<env>_comparison.png
```

### 7.3 SB3 Baseline

Stable-Baselines3 baseline 入口：

```bash
python3 scripts/train_sb3_baseline.py \
  --env Pendulum-v1 \
  --algo SAC \
  --seed 0 \
  --total-timesteps 50000
```

绘图入口：

```bash
python3 scripts/plot_fedrl_heterogeneous.py \
  --fed-log-dir logs_fedrl_hetero \
  --sb3-log-dir logs_sb3
```

---

## 8. 指标与可视化

### 8.1 主要训练指标

`metrics.csv` 中的关键字段：

| 指标 | 含义 |
|------|------|
| `eval_reward_mean` | 当前策略的评估 reward |
| `best_fitness` | EA population 当前最佳 fitness |
| `mean_fitness` | EA population 平均 fitness |
| `weight_diversity` | actor 权重多样性 |
| `client_reward_mean` | 被选中客户端的本地 reward 均值 |
| `client_reward_std` | 被选中客户端的本地 reward 方差 |
| `client_fitness_mean` | 跨客户端 EA fitness 均值 |
| `client_fitness_std` | 跨客户端 EA fitness 方差 |
| `selected_clients` | 当前 round 参与本地训练的客户端数量 |
| `aggregation_entropy` | 聚合权重熵 |
| `fed_round_applied` | 当前 generation 是否执行 federated aggregation |
| `archive_best` | 全局 elite archive 中的最佳历史 fitness |
| `archive_size` | archive 当前保留的 elite 数量 |
| `aggregation_temperature` | softmax 聚合温度 |
| `comm_upload_bytes` | 实际上传量估计 |
| `comm_full_traj_bytes` | 假设上传完整 trajectory 的通信量估计 |

### 8.2 绘图脚本

统一绘图入口：

```bash
python3 generate_plots.py --log-dir logs --require-real
```

主要输出：

| 图片 | 内容 |
|------|------|
| `plots/sample_efficiency_comparison_steps.png` | 按环境步数比较主方法与 baseline |
| `plots/sample_efficiency_comparison_wallclock.png` | 按 wall-clock time 比较 |
| `plots/ablation_fed_evo_rl.png` | FedEvoRL 组件消融 |
| `plots/ablation_re2_impact.png` | ERL-Re2 baseline 消融 |
| `plots/multi_env_comparison.png` | 多 MuJoCo 环境综合比较 |
| `plots/fedrl_heterogeneous/<env>_comparison.png` | FedRL 异质环境与 SB3 baseline 对比 |

---

## 9. 项目创新点总结

### 9.1 与普通 Distributed ERL 的区别

Distributed ERL 中 worker 通常只是并行 evaluator，用于加速 population evaluation。

FedEvoRL 中 client 是拥有私有环境、私有 buffer 和本地 learner 的学习主体。客户端之间不共享 trajectory，服务器只接收模型权重和标量摘要。

因此，FedEvoRL 不是简单的并行加速，而是面向隐私约束和 non-IID MDP 的 federated learning 框架。

### 9.2 与普通 Federated RL 的区别

普通 Federated RL 多维护单个 global policy。

FedEvoRL 维护 EA policy population：

*   用 population diversity 缓解 policy averaging collapse。
*   用 EA search 保留异质客户端下的多个策略模式。
*   用 local RL 提供梯度改进。
*   用 softmax / interval aggregation 控制联邦更新成本和稳定性。
*   用 global elite archive 保留历史高分策略。
*   用 soft injection 将梯度学习结果转化为进化材料。

### 9.3 与 ERL-Re2 的区别

ERL-Re2 主要强调 elite trajectory reproduction 和 EA/RL 经验复用。

FedEvoRL 主要强调：

*   隐私约束下的客户端本地学习；
*   跨客户端异质性；
*   softmax fitness-aware federated aggregation；
*   global elite archive；
*   EA population 对抗 policy averaging collapse。

---

## 10. 当前实现状态

已实现：

*   `fed_evo_rl` 主训练模式；
*   federated clients 与本地 replay buffer；
*   跨客户端 EA fitness evaluation；
*   uniform / fitness / softmax model aggregation；
*   interval aggregation 与低分客户端过滤；
*   global elite archive；
*   soft EA injection；
*   FedEvoRL 专用消融参数；
*   CartPole / Acrobot / Pendulum / LunarLander / BipedalWalkerHardcore 异质客户端环境；
*   Stable-Baselines3 PPO/SAC/TD3 baseline 接入；
*   MuJoCo 长线对比脚本；
*   FedRL 异质环境对比脚本；
*   FedEvoRL 消融脚本；
*   主对比与消融可视化。

后续可增强：

*   真实 MuJoCo 物理参数异质性，如 gravity、body mass、friction、gear；
*   personalized policy assignment；
*   clustered federated aggregation；
*   FedProx / SCAFFOLD 风格校正项；
*   离散动作环境的专门 DQN / discrete actor-critic FedRL 实现；
*   privacy-preserving aggregation；
*   多环境、多 seed 的完整论文级统计显著性分析。
