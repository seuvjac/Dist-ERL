# FedEvoFSAC / Dist-ERL

## 1. Federated RL 问题设定

本项目当前主线只研究三个离散动作控制环境：

```text
CartPole-v1
Acrobot-v1
LunarLander-v3
```

每个 client 拥有自己的私有环境、私有 replay buffer 和本地 learner。client 之间不共享 trajectory，服务器只能接收 actor 参数、reward / fitness 等标量统计信息。

异质性来自不同 client 的局部 MDP 扰动：

| 环境 | 动作类型 | client heterogeneity |
|------|----------|----------------------|
| `CartPole-v1` | discrete | gravity、cart mass、pole mass、pole length |
| `Acrobot-v1` | discrete | link length、link mass、center of mass、gravity |
| `LunarLander-v3` | discrete | gravity、wind、turbulence |

因此，本项目不再把 MuJoCo / Pendulum / 连续动作环境作为当前主实验对象。

当前在每个原始环境上定义三种异质联邦场景：

| 场景 | `client_heterogeneity_mode` | 强度 | 含义 |
|------|-----------------------------|------|------|
| `dynamics_mild` | `env_params_only` | `0.25` | 只改变客户端物理参数，如 gravity、mass、length、wind |
| `sensor_reward` | `reward_action_noise` | `0.35` | 原始动力学不变，只改变观测噪声、reward scale/bias 和 seed stream |
| `mixed_hard` | `mixed` | `0.50` | 同时改变动力学参数与观测/奖励扰动，作为最难异质场景 |

这样可以区分三类问题：动力学 non-IID、感知/奖励 non-IID，以及混合强异质 non-IID。

## 2. 当前算法：FedEvoFSAC

当前主方法是：

```text
FedEvoFSAC
= Federated discrete SAC
+ ERL-Re2-style actor neuroevolution
+ reward-aware federated actor aggregation
+ global elite archive
```

它对应论文 `Federated Reinforcement Learning for Sharing Experiences Between Multiple Workers` 中的 FSAC 思路，但额外加入 EA population，用演化搜索维护多个候选 actor。

## 3. 三类角色

| 角色 | 代码位置 | 职责 |
|------|----------|------|
| Federated Evolution Server | `src/main.py`, `src/manager.py` | 维护 actor population，执行选择、交叉、变异、archive 和 RL 注入 |
| Federated FSAC Client | `src/federated.py` | 持有私有环境和 replay buffer，本地训练 discrete SAC actor / critics |
| Evaluator / Baseline Worker | `src/worker.py`, `src/learner.py` | 支撑 pure RL、pure EA、standard ERL、Dist-ERL、ERL-Re2 等对比 |

主入口：

```bash
python -m src.main --mode fed_evo_rl --algorithm FSAC --env CartPole-v1
```

## 4. 算法流程

每一轮 generation 中执行：

1. **跨 client 私有评估**
   - server 下发 EA population 中的 actor 参数；
   - 每个 client 在自己的异质 MDP 中评估 actor；
   - client 只返回 scalar fitness，不上传 trajectory。

2. **服务器端演化**
   - server 按跨 client 平均 fitness 排序；
   - 执行 elitism、tournament selection、row-wise crossover、bounded mutation；
   - 维护 global elite archive，防止历史最优 actor 被后续扰动覆盖。

3. **本地 FSAC 更新**
   - 每隔 `K` 代，server 将当前 best actor 下发给部分 clients；
   - client 用该 actor rollout，transition 写入私有 replay buffer；
   - client 本地训练 discrete SAC actor、critic1、critic2 和 temperature；
   - 上传更新后的 actor 参数和本地 reward 摘要。

4. **联邦 actor 聚合**
   - server 对 client actor update 做 reward-aware aggregation；
   - 默认使用 softmax 权重、低分 client 过滤和 delta norm clipping；
   - 聚合 actor 通过 soft injection 注入 EA population 的弱 non-elite 个体。

## 5. FSAC：Federated Discrete SAC

这里的 FSAC 是 **Federated Soft Actor-Critic**。因为三个环境都是离散动作，所以本项目实现的是 discrete SAC：

```text
actor(s) -> logits over discrete actions
pi(a|s) = softmax(actor(s))
critic1(s), critic2(s) -> Q values for all actions
```

本地更新目标：

```text
V(s') = sum_a pi(a|s') * (min(Q1_t, Q2_t)(s',a) - alpha * log pi(a|s'))
y = r + gamma * (1 - done) * V(s')
critic_loss = Huber(Q1(s,a), y) + Huber(Q2(s,a), y)
actor_loss = sum_a pi(a|s) * (alpha * log pi(a|s) - min(Q1,Q2)(s,a))
```

temperature `alpha` 使用可学习参数 `log_alpha`，target entropy 取 `0.98 * log(|A|)`。

## 6. EA 进化什么，不进化什么

FedEvoFSAC 的 EA genotype 只包含：

```text
actor.*
```

不把 critic 放进 EA population：

```text
critic1.*
critic2.*
target_critic1.*
target_critic2.*
log_alpha
```

原因：

- actor 是可直接执行的策略，适合被 EA 评估、交叉和变异；
- critic 是 client 本地 value estimator，强烈依赖本地 replay buffer；
- 聚合或变异 critic 容易引入 Q 值尺度漂移；
- 只共享 actor 更接近 FSAC 论文中的 parameter sharing 思路。

## 7. 演化算法

服务器端使用 ERL-Re2-style steady-state neuroevolution：

| 步骤 | 作用 |
|------|------|
| Elitism | 保留 top actors |
| Tournament selection | 从 population 中选择 winners / parents |
| Crossover | 对 actor 的二维权重矩阵做 row-wise 重组 |
| Bounded mutation | 对 actor 权重加入有界扰动 |
| Super / reset mutation | 给低质量个体提供更强探索 |
| Global elite archive | 保存历史 top-k actor，并每代恢复 |

当前实现将 GA 的作用前缀配置为：

```text
actor_prefix = "actor."
```

并通过 `--ea-weight-clip` 限制 actor 参数范围，避免 mutation 过强导致策略崩溃。

## 8. 联邦聚合

FedEvoFSAC 聚合的是 client 上传的 actor 参数，不聚合 trajectory，也不聚合 critic。

默认机制：

| 机制 | 参数 | 作用 |
|------|------|------|
| 间隔聚合 | `--fed-aggregation-interval 5` | 降低通信和训练开销 |
| softmax reward 权重 | `--fed-aggregation softmax` | 高 reward client 权重更大 |
| 温度控制 | `--fed-aggregation-temperature 75` | 防止单个 client 过度支配 |
| 低分过滤 | `--fed-min-client-score-quantile 0.25` | 丢弃低质量 actor update |
| delta clipping | `--fed-delta-clip-norm 5` | 限制 client update 的参数步长 |
| soft injection | `--migration-blend 0.35` | 将聚合 actor 软注入 EA 弱个体 |

聚合形式是以当前 best actor 为中心的 delta aggregation：

```text
delta_i = clip(theta_i - theta_best)
theta_fed = theta_best + sum_i w_i * delta_i
```

这比直接平均完整 actor 更稳。

## 9. Baseline 和对比曲线

主实验只比较三个离散环境。

FedEvoFSAC 家族内部消融：

```text
FedEvoFSAC-full
FedEvoFSAC-uniform_aggregation
FedEvoFSAC-no_local_rl
FedEvoFSAC-no_ea_injection
FedEvoFSAC-no_heterogeneity
```

当前对照组只保留 SAC / FSAC：

```text
Paper-SAC
Paper-FSAC
```

说明：当前实验不再把 SB3 放入主对照组。主算法维持使用 `FedEvoFSAC`；对照只保留同一算法族内的 `Paper-SAC` 和 `Paper-FSAC`，这样能更直接地检验“联邦共享”和“EA actor 进化”分别带来的影响。论文里的 FSAC 复现为本项目的 `Paper-FSAC`：每个 worker 本地训练 discrete SAC，critic、target critic 和温度参数留在本地；服务器根据 worker 的 performance index 选择当前最优 worker，只共享最优 actor，并用 reward/PI 诱导的 Boltzmann 权重把本地 actor 与最优 actor 混合。`Paper-SAC` 是去掉联邦共享后的独立多 worker SAC。

## 10. 实验脚本

完整三环境套件：

```bash
./run_fedrl_heterogeneous_suite.sh
```

三种异质联邦场景：

```bash
./run_fedrl_three_scenarios.sh
```

默认只跑 `FedEvoFSAC-full`，用于比较不同异质场景的影响；如果要同时跑消融，可以覆盖：

```bash
FED_VARIANTS="full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity" \
  ./run_fedrl_three_scenarios.sh
```

只补跑 FedEvoFSAC：

```bash
./run_fedevofsac_for_baselines.sh
```

只跑 SAC / FSAC baseline：

```bash
./run_fsac_paper_baseline.sh
```

快速 smoke：

```bash
python -m src.main \
  --env CartPole-v1 \
  --mode fed_evo_rl \
  --algorithm FSAC \
  --population-size 4 \
  --num-clients 2 \
  --max-generations 2 \
  --max-episode-steps 100 \
  --client-rollouts 1 \
  --client-updates 1 \
  --batch-size 16 \
  --eval-episodes 1
```

## 11. 指标

| 指标 | 含义 |
|------|------|
| `best_fitness` | EA population 当前最佳跨 client fitness |
| `mean_fitness` | EA population 平均 fitness |
| `weight_diversity` | actor population 多样性 |
| `client_reward_mean` | 被选中 clients 本地 rollout reward 均值 |
| `client_reward_std` | 被选中 clients 本地 reward 方差 |
| `client_fitness_mean` | 跨 client 评估均值 |
| `client_fitness_std` | 跨 client 评估方差 |
| `aggregation_entropy` | 联邦聚合权重熵 |
| `archive_best` | global elite archive 历史最佳 fitness |
| `comm_upload_bytes` | 实际上传 actor 参数量估计 |
| `comm_full_traj_bytes` | 假设上传完整 trajectory 的通信量估计 |

## 12. 当前实现状态

已完成：

- 三个离散环境主线：`CartPole-v1`、`Acrobot-v1`、`LunarLander-v3`；
- `FSACPolicy`：discrete SAC actor、twin critics、target critics、learnable alpha；
- EA genotype actor-only；
- GA actor 前缀可配置且当前固定为 `actor.`；
- reward-aware federated actor aggregation；
- delta clipping 与 bounded EA mutation；
- global elite archive；
- FedEvoFSAC 消融脚本；
- Paper-SAC / Paper-FSAC baseline 和对比绘图。

主要风险：

- LunarLander 比 CartPole / Acrobot 更难，FSAC 可能需要更长训练和更细调参；
- actor-only 共享更稳，但 client critic 完全本地化，早期本地更新可能噪声较大；
- 异质 client reward scale 会影响 softmax 聚合权重，需要关注 `aggregation_entropy`；
- 当前 privacy 是 trajectory-private，不是 differential privacy 或 secure aggregation。
