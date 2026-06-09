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
| `CartPole-v1` | discrete | gravity、cart mass、pole mass、pole length、force magnitude、integration timestep、observation/reward/action perturbation |
| `Acrobot-v1` | discrete | link length、link mass、center of mass、available torque、joint velocity limits、integration timestep、observation/reward/action perturbation |
| `LunarLander-v3` | discrete | gravity、wind、turbulence |

因此，本项目不再把 MuJoCo / Pendulum / 连续动作环境作为当前主实验对象。

当前在每个原始环境上定义三种异质联邦场景：

| 场景 | `client_heterogeneity_mode` | 强度 | 含义 |
|------|-----------------------------|------|------|
| `dynamics_mild` | `env_params_only` | `0.25` | 只改变客户端物理参数，如 gravity、mass、length、wind |
| `sensor_reward` | `reward_action_noise` | `0.35` | 原始动力学不变，只改变观测噪声、reward scale/bias 和 seed stream |
| `mixed_hard` | `mixed` | `0.50` | 同时改变动力学参数与观测/奖励/动作扰动，作为最难异质场景 |

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

当前同协议对照组只保留 SAC / FSAC / DQN：

```text
Paper-SAC
Paper-FSAC
FedAvg-SAC
FedSoftmax-SAC-noEA
FedBest-SAC
RobustFed-SAC-Median
RobustFed-SAC-TrimmedMean
ContextFed-SAC-lite
FedAvg-DQN
```

说明：当前实验不再把 SB3 和 `EvoSAC-noFed` 放入主横向对照组。主算法维持使用 `FedEvoFSAC`；横向对照只保留同协议 SAC / FSAC / DQN 系联邦方法。FedEvoFSAC 的模块消融单独出图，不和其他算法混在同一张横向对比图里。

这里的 `FedAvg-SAC`、`RobustFed-SAC-Median`、`RobustFed-SAC-TrimmedMean`、`ContextFed-SAC-lite` 是**同协议内部基线**：它们统一使用本项目的三环境、异质 client 设置、日志格式和评估协议，用来拆解聚合规则本身的影响。它们不是外部论文的严格原代码复现，不能在论文图注里写成 external baseline。若需要和论文方法比较，必须使用外部仓库的原代码单独跑 external-original comparison。

各对照组含义：

| 曲线 | 含义 | 主要回答的问题 |
|------|------|----------------|
| `Paper-SAC` | 多 worker 本地 discrete SAC，不共享 actor | 没有联邦共享时 SAC 的表现 |
| `Paper-FSAC` | 论文式 FSAC：最优 worker actor 与本地 actor 做 Boltzmann blending | 论文式 best-worker 共享是否有效 |
| `FedAvg-SAC` | client actor 做均匀平均，critic 保持本地 | 普通 FedAvg actor 聚合是否足够 |
| `FedSoftmax-SAC-noEA` | client actor 按 performance index softmax 加权聚合，无 EA | reward-aware federation 在没有 EA 时的贡献 |
| `FedBest-SAC` | 每轮将最优 worker actor 广播给所有 worker | 贪心 best-worker 共享是否稳定 |
| `RobustFed-SAC-Median` | client actor 做逐参数 median 聚合，critic 保持本地 | 鲁棒聚合思想在异质 client 下是否改善稳定性 |
| `RobustFed-SAC-TrimmedMean` | client actor 做逐参数 trimmed mean 聚合，critic 保持本地 | 比 median 更平滑的鲁棒聚合是否更稳 |
| `ContextFed-SAC-lite` | 用 performance index 和 actor 距离构造 context-aware 权重聚合 actor | 轻量上下文聚合是否优于简单平均 |
| `FedAvg-DQN` | 多 worker 本地 DQN，周期性 FedAvg 聚合 Q-network | 参考 Federated-DRL，DQN 系联邦方法和 SAC 系方法的差异 |

论文里的 FSAC 复现为本项目的 `Paper-FSAC`：每个 worker 本地训练 discrete SAC，critic、target critic 和温度参数留在本地；服务器根据 worker 的 performance index 选择当前最优 worker，只共享最优 actor，并用 reward/PI 诱导的 Boltzmann 权重把本地 actor 与最优 actor 混合。

外部论文原代码复现作为单独的 external-original comparison 管理，不直接和同协议内部基线混名。原因是这些仓库支持的环境、依赖和训练协议不同；能在相同环境上运行的才放入外部复现图。外部复现结果不参与 FedEvoFSAC 消融图，也不和内部同协议曲线混称为同一类 baseline。

| 外部代码 | 实际 RL 主体 | 本地路径 | 可比环境 | 使用方式 |
|----------|--------------|----------|----------|----------|
| FedFormer | SAC | `/home/ywj/code/FedFormer` | MetaWorld MT10，不是 CartPole/Acrobot/LunarLander | related work 或单独 MetaWorld 复现，不放三环境主图 |
| Byzantine-Federated-RL / FedPG-BR | policy gradient | `/home/ywj/code/Byzantine-Federated-RL` | CartPole-v1、LunarLander-v2、HalfCheetah-v2 | 可作为 CartPole/LunarLander 外部原代码复现 |
| Federated-DRL | DQN / DDQN | `/home/ywj/code/Federated-DRL` | CartPole-v1、LunarLander-v2、Mario | 可作为 FedAvg-DQN 外部原代码复现 |
| FederatedRL | PPO | `/home/ywj/code/FederatedRL` | CartPole-v1、若干 MuJoCo/IoT 任务 | 可作为 PPO-FedRL related work，默认不进三环境主图 |

当前可严格复现的外部对照边界：

- `CartPole-v1`：可跑 `Federated-DRL` 的 FedAvg-DQN/DDQN 原代码，也可跑 `Byzantine-Federated-RL` 的 FedPG-BR 原代码。
- `LunarLander`：外部仓库多使用 `LunarLander-v2`，本项目主环境是 Gymnasium 的 `LunarLander-v3`；可做外部复现图，但图注必须说明环境版本不同。
- `Acrobot-v1`：当前已下载外部仓库没有直接支持 Acrobot 的原代码复现，不强行改源码充当 strict reproduction。
- `FedFormer`：原论文是 MetaWorld 连续控制 SAC，不适合直接放入 CartPole/Acrobot/LunarLander 三环境主图；若要比较，应另开 MetaWorld 复现实验。

## 10. 实验脚本

完整三环境套件：

```bash
./run_fedrl_heterogeneous_suite.sh
```

该脚本默认使用更强的异质设定：

```text
CLIENT_HETEROGENEITY=0.60
CLIENT_HETEROGENEITY_MODE=mixed
NUM_WORKERS=4
```

输出两类图：

```text
plots/fedrl_comparison_mixed   # FedEvoFSAC-full vs SAC/FSAC/DQN 横向算法对比
plots/fedrl_ablations_mixed    # FedEvoFSAC 内部消融
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

只跑外部论文原代码复现：

```bash
./run_external_original_baselines.sh
```

该脚本只调用 `/home/ywj/code` 下的外部仓库，并把输出整理到 `external_original_logs/`。它依赖外部仓库自己的 Python/Gym/PyTorch 版本；如果当前 conda 环境不兼容，应单独建对应环境后用 `EXTERNAL_PYTHON=/path/to/python ./run_external_original_baselines.sh` 运行。

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

主曲线默认使用 `eval_reward_mean`，即当前策略评估回报；`best_fitness` / `archive_best` 只作为辅助表格或附图指标，避免把历史最优和当前性能混在同一主图里。FedRL 主文建议优先使用 communication rounds 作为横坐标，因为它直接对应联邦通信效率；raw environment steps 和 normalized progress 作为补充视角：

- reward vs communication round / generation：主图，说明联邦通信效率。
- reward vs raw environment steps：补充图，说明样本效率；所有算法应跑到同一个 step budget，提前收敛时曲线保持最后当前评估值。
- reward vs normalized progress：只作为可视化辅助，不作为主定量结论。

最终表格至少报告 `Final return mean +/- std`、`Best return mean +/- std`、`max_steps`、`max_round` 和 `wall_time_sec`。CartPole 只作为 sanity check；核心证据优先放在 Acrobot 和 LunarLander。LunarLander 结论应写成强异质下相对改善，而不是声称完全解决异质性退化。

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
- SAC / FSAC / EvoSAC baseline 和对比绘图。

主要风险：

- LunarLander 比 CartPole / Acrobot 更难，FSAC 可能需要更长训练和更细调参；
- actor-only 共享更稳，但 client critic 完全本地化，早期本地更新可能噪声较大；
- 异质 client reward scale 会影响 softmax 聚合权重，需要关注 `aggregation_entropy`；
- 当前 privacy 是 trajectory-private，不是 differential privacy 或 secure aggregation。
