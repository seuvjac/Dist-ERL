# FedEvoSAC / Dist-ERL

## 1. Federated RL 问题设定

本项目当前主线研究连续动作异质控制环境：

```text
Swimmer-v5
Reacher-v5
HalfCheetah-v5
```

每个 client 拥有自己的私有环境、私有 replay buffer 和本地 SAC learner。client 之间不共享 trajectory，服务器只能接收 actor 参数、reward / fitness 等标量统计信息；critic、target critic、temperature 和 replay buffer 全部留在本地。

异质性来自不同 client 的局部 MDP 扰动：

| 环境 | 动作类型 | client heterogeneity |
|------|----------|----------------------|
| `Swimmer-v5` | continuous | gravity、body mass、joint damping、geom friction、observation/reward/action perturbation |
| `Reacher-v5` | continuous | body mass、joint damping、geom friction、observation/reward/action perturbation |
| `HalfCheetah-v5` | continuous | gravity、body mass、joint damping、geom friction、observation/reward/action perturbation |

当前在每个原始环境上定义三种异质联邦场景：

| 场景 | `client_heterogeneity_mode` | 强度 | 含义 |
|------|-----------------------------|------|------|
| `dynamics_mild` | `env_params_only` | `0.25` | 只改变客户端物理参数，如 gravity、mass、length、wind |
| `sensor_reward` | `reward_action_noise` | `0.35` | 原始动力学不变，只改变观测噪声、reward scale/bias 和 seed stream |
| `mixed_hard` | `mixed` | `0.50`-`0.60` | 同时改变动力学参数与观测/奖励/动作扰动，作为最难异质场景；主实验先使用 no-heterogeneity / mild dynamics，hard mixed 作为压力测试 |

这样可以区分三类问题：动力学 non-IID、感知/奖励 non-IID，以及混合强异质 non-IID。

## 2. 当前主线算法：FedEvoSAC

当前主方法切换为连续控制：

```text
FedEvoSAC
= Federated continuous SAC
+ ERL-Re2-style actor neuroevolution
+ reward-aware federated actor aggregation
+ global elite archive
```

连续 SAC 是 SAC 更原生的使用场景，也更适合 EA：EA 直接扰动连续 actor 参数时行为变化更平滑，client heterogeneity 可以通过 gravity、mass、friction、motor strength、wind、terrain 等动力学差异表达。

当前连续主实验环境：

```text
Swimmer-v5
Reacher-v5
HalfCheetah-v5
```

连续主对照组：

```text
FedAvg-SAC
FedBest-SAC
FedSoftmax-SAC-noEA
RobustFed-SAC-Median
FedEvoSAC
```

## 3. 三类角色

| 角色 | 代码位置 | 职责 |
|------|----------|------|
| Federated Evolution Server | `src/main.py`, `src/manager.py` | 维护 actor population，执行选择、交叉、变异、archive 和 RL 注入 |
| Federated SAC Client | `src/federated.py` | 持有私有环境和 replay buffer；本地训练 continuous SAC actor / critics / temperature |
| Evaluator / Baseline Worker | `src/worker.py`, `src/learner.py` | 支撑 pure RL、pure EA、standard ERL、Dist-ERL、ERL-Re2 等对比 |

主入口：

```bash
python -m src.main --mode fed_evo_rl --algorithm SAC --env Reacher-v5
```

## 4. 算法流程

每一轮 generation 中执行：

1. **EA population 跨 client 评估**
   - server 维护一组 actor population，每个个体只包含 `actor.*`；
   - server 将每个 actor 下发给所有 federated clients；
   - 每个 client 在自己的本地 MDP 中 rollout 评估该 actor；
   - client 只返回 scalar fitness，不上传 trajectory；
   - server 对同一 actor 的多 client fitness 求均值，得到该个体的 federated fitness。

2. **Global elite archive 更新**
   - server 按 federated fitness 对 population 排序；
   - 将当前高分 actor 与历史 archive 合并；
   - 保留 top-k actor 作为 global elite archive；
   - archive actor 是当前可部署策略候选，防止历史好策略被后续 mutation 或 RL migration 覆盖。

3. **EA selection / crossover / mutation**
   - 执行 elitism，保留高 fitness actor；
   - 用 tournament selection 选择 parent / winner actor；
   - 对 actor 的二维权重矩阵做 row-wise crossover；
   - 对 non-elite actor 执行 bounded mutation、super mutation 和 reset mutation；
   - EA 只作用于 `actor.*`，不进化 critic、target critic、temperature 或 replay buffer。

4. **Elite restore**
   - GA 结束后，将 global elite archive 中的 top actor hard restore 回 population 前部；
   - 这样 archive elite 会继续参与下一代评估，也能作为本轮 SAC 本地更新的 warm-start actor。

5. **本地 SAC refinement**
   - 每隔 `K` 代，server 将当前 best / archive elite actor 下发给部分 clients；
   - client 用该 actor 初始化本地 SAC actor；
   - client rollout 产生 transition，写入私有 replay buffer；
   - client 本地训练 continuous SAC actor、critic1、critic2 和 temperature；
   - critic、target critic、temperature 和 replay buffer 始终留在本地；
   - client 只上传更新后的 actor 参数和本地 reward 摘要。

6. **联邦 actor 聚合**
   - server 对 client 上传的 actor update 做 reward-aware aggregation；
   - 默认使用 softmax 权重、低分 client 过滤和 delta norm clipping；
   - 聚合时以当前 best actor 为中心聚合 delta，避免直接平均完整 actor 造成大幅漂移。

7. **RL-to-EA soft injection 和 deployable policy**
   - 若聚合 actor 的本地表现达到门控条件，server 将其 soft inject 到 EA population 的弱 non-elite 个体；
   - injection 后再次 restore archive elite，防止历史最优 actor 被覆盖；
   - `eval_reward_mean` 记录 global elite archive 中 best actor 的 deployable evaluation；
   - `client_reward_mean` / `client_reward_std` 单独记录本地 SAC rollout 表现。

## 5. Continuous SAC

连续主线使用 `SACPolicy`，actor 是 tanh-squashed Gaussian policy：

```text
actor(s) -> mean(s), log_std(s)
a = tanh(mean + std * epsilon)
critic1(s,a), critic2(s,a) -> Q values
```

本地更新目标：

```text
y = r + gamma * (1 - done) * (min(Q1_t,Q2_t)(s',a') - alpha * log pi(a'|s'))
critic_loss = Huber(Q1(s,a), y) + Huber(Q2(s,a), y)
actor_loss = E[alpha * log pi(a|s) - min(Q1,Q2)(s,a)]
alpha_loss = -log_alpha * (log pi(a|s) + target_entropy)
```

temperature `alpha` 使用可学习参数 `log_alpha`。连续 SAC 的 target entropy 默认取 `-|A|`。

## 6. EA 进化什么，不进化什么

FedEvoSAC 的 EA genotype 只包含：

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
- 只共享 actor 可以降低通信量，也避免跨异质 client 直接平均 critic 带来的 value-scale mismatch。

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

FedEvoSAC 聚合的是 client 上传的 actor 参数，不聚合 trajectory，也不聚合 critic。

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

连续主实验只比较 continuous SAC 系方法：

```text
FedAvg-SAC
FedBest-SAC
FedSoftmax-SAC-noEA
RobustFed-SAC-Median
FedEvoSAC
```

FedEvoSAC 家族内部消融可以沿用：

```text
FedEvoSAC-full
FedEvoSAC-uniform_aggregation
FedEvoSAC-no_local_rl
FedEvoSAC-no_ea_injection
FedEvoSAC-no_heterogeneity
```

离散附线仍可使用原来的 FSAC / DQN 对照：

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

离散附线只用于 sanity / preliminary，不作为主论文核心证据。`FSACPolicy` 是 discrete SAC：actor 输出离散动作 logits，critic 输出每个动作的 Q 值，temperature target entropy 取 `0.98 * log(|A|)`。离散 `FedEvoFSAC` 对应论文 `Federated Reinforcement Learning for Sharing Experiences Between Multiple Workers` 中的 FSAC 思路，但额外加入 EA population；这条线和连续 `FedEvoSAC` 主实验分开报告。

说明：连续主图不再放入 SB3、DQN、离散 FSAC、`EvoSAC-noFed` 或 `Independent-SAC`。`Independent-SAC` 没有联邦通信，只作为可选诊断曲线，不进入主横向对比。主算法为 `FedEvoSAC`；横向对照只保留同协议 continuous SAC 系联邦方法。FedEvoSAC 的模块消融单独出图，不和其他算法混在同一张横向对比图里。

这里的 `FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median` 是**同协议内部基线**：它们统一使用本项目的连续环境、异质 client 设置、日志格式和评估协议，用来拆解聚合规则本身的影响。它们不是外部论文的严格原代码复现，不能在论文图注里写成 external baseline。若需要和论文方法比较，必须使用外部仓库的原代码单独跑 external-original comparison。

各对照组含义：

| 曲线 | 含义 | 主要回答的问题 |
|------|------|----------------|
| `FedAvg-SAC` | client actor 做均匀平均，critic 保持本地 | 普通 FedAvg actor 聚合是否足够 |
| `FedSoftmax-SAC-noEA` | client actor 按 performance index softmax 加权聚合，无 EA | reward-aware federation 在没有 EA 时的贡献 |
| `FedBest-SAC` | 每轮将最优 worker actor 广播给所有 worker | 贪心 best-worker 共享是否稳定 |
| `RobustFed-SAC-Median` | client actor 做逐参数 median 聚合，critic 保持本地 | 鲁棒聚合思想在异质 client 下是否改善稳定性 |
| `FedEvoSAC` | continuous SAC 本地学习 + federated actor aggregation + EA actor population | EA 是否能在连续异质控制中提供更稳探索和更好 actor 多样性 |

四个联邦 SAC baseline 使用相同的训练/部署分离协议。client 的训练 actor 始终继续执行本地 SAC 更新和各自的联邦聚合，不因一次评估下降而回滚；server 另外保存固定评估协议下的历史最佳 actor，作为 deployable checkpoint。主图的 `eval_reward_mean` 画可部署 checkpoint，候选策略辅助图画 `candidate_eval_mean`，因此既能避免已验证策略被覆盖，也能真实展示当前训练策略的上升、波动或失败。`deployment_rollback` 字段现在表示本轮继续保留旧 checkpoint，而不是把训练 actor 回滚。

为避免旧实现中“每收集 1000 个 transition 只更新 4 次”的欠训练问题，baseline 按新采样量计算有限的本地 SAC 更新数：`updates = min(max(base_updates, ceil(new_steps * update_to_data_ratio)), max_updates_per_round)`。默认 `update_to_data_ratio=0.02`、上限 `20`，在不改变网络、损失函数和聚合规则的前提下，使 critic/actor 获得足够更新。联邦聚合替换 actor 参数后清空 actor optimizer 的陈旧 Adam 动量；本地 critic optimizer、temperature optimizer 和 replay buffer 全部保留。FedAvg、FedBest、FedSoftmax 和 Median 的原聚合思想保持不变。

外部论文原代码复现作为单独的 external-original comparison 管理，不直接和同协议内部基线混名。原因是这些仓库支持的环境、依赖和训练协议不同；能在相同环境上运行的才放入外部复现图。外部复现结果不参与 FedEvoSAC 消融图，也不和内部同协议曲线混称为同一类 baseline。

| 外部代码 | 实际 RL 主体 | 本地路径 | 可比环境 | 使用方式 |
|----------|--------------|----------|----------|----------|
| FedFormer | SAC | `/home/ywj/code/FedFormer` | MetaWorld MT10，不是当前三个 Gymnasium 连续主环境 | related work 或单独 MetaWorld 复现，不混入主图 |
| Byzantine-Federated-RL / FedPG-BR | policy gradient | `/home/ywj/code/Byzantine-Federated-RL` | CartPole-v1、LunarLander-v2、HalfCheetah-v2 | 可作为 CartPole/LunarLander/HalfCheetah 外部原代码复现 |
| Federated-DRL | DQN / DDQN | `/home/ywj/code/Federated-DRL` | CartPole-v1、LunarLander-v2、Mario | 可作为 FedAvg-DQN 外部原代码复现 |
| FederatedRL | PPO | `/home/ywj/code/FederatedRL` | CartPole-v1、若干 MuJoCo/IoT 任务 | 可作为 PPO-FedRL related work，默认不进连续主图 |

当前可严格复现的外部对照边界：

- `HalfCheetah`：`Byzantine-Federated-RL` 支持旧版 `HalfCheetah-v2`，本项目主线使用 `HalfCheetah-v5`；可做 external-original 辅助图，但不能和内部同协议主图混称。
- `Reacher`：当前外部仓库没有同版本、同协议的严格复现结果，因此只进入本项目内部同协议主图。
- `CartPole / Acrobot / MountainCar`：保留为离散附线；外部 DQN / FedPG 代码可用于 related-work 复现，不进入连续 FedEvoSAC 主图。
- `FedFormer`：原论文是 MetaWorld 连续控制 SAC；若要比较，应另开 MetaWorld 复现实验。

## 10. 实验脚本

连续主线 FedEvoSAC 套件：

```bash
./run_continuous_fedevosac_suite.sh
```

该脚本默认使用更强的异质设定：

```text
CLIENT_HETEROGENEITY=0.60
CLIENT_HETEROGENEITY_MODE=mixed
NUM_WORKERS=3
```

默认连续环境：

```text
Swimmer-v5
Reacher-v5
HalfCheetah-v5
```

输出三类结果：

```text
plots/fedevosac_continuous_comparison_round   # FedEvoSAC vs continuous SAC 横向对比
plots/fedevosac_continuous_candidate_round    # 当前训练 candidate，展示真实学习波动
plots/fedevosac_continuous_ablations_round    # FedEvoSAC 内部消融
plots/fedevosac_continuous_tables             # final / best return 表格
```

默认连续对照组为 `FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median` 和 `FedEvoSAC`；这些 baseline 基于本项目已有联邦 RL 复现框架迁移到 continuous SAC，共享同一环境异质性、评估、日志和 actor 聚合协议。

三种异质联邦场景脚本仍可用于离散附线分析：

```bash
./run_fedrl_three_scenarios.sh
```

默认只跑离散 `FedEvoFSAC-full`，用于比较不同异质场景的影响；如果要同时跑消融，可以覆盖：

```bash
FED_VARIANTS="full uniform_aggregation no_local_rl no_ea_injection no_heterogeneity" \
  ./run_fedrl_three_scenarios.sh
```

离散附线完整套件：

```bash
./run_fedrl_heterogeneous_suite.sh
```

只补跑离散 FedEvoFSAC：

```bash
./run_fedevofsac_for_baselines.sh
```

默认脚本使用 `BUDGET_PRESET=reduced`，会缩小 population / generation / evaluation 数量来控制 equal-step 预算，适合日常对比和调参。若要跑最终 full budget，可显式设置：

```bash
BUDGET_PRESET=full ./run_continuous_fedevosac_suite.sh
```

只跑离散 SAC / FSAC baseline：

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
  --env Reacher-v5 \
  --mode fed_evo_rl \
  --algorithm SAC \
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
| `candidate_eval_mean` | 当前聚合训练 actor 的固定协议评估回报 |
| `deployable_eval_mean` | server 历史最佳可部署 actor 的评估回报 |
| `local_updates_per_worker` | 每轮每个 baseline client 实际执行的本地 SAC 更新数 |
| `comm_upload_bytes` | 实际上传 actor 参数量估计 |
| `comm_full_traj_bytes` | 假设上传完整 trajectory 的通信量估计 |

主曲线默认使用 `eval_reward_mean`，表示 server 当前可部署策略。baseline 的当前训练策略记录在 `candidate_eval_mean`，并单独生成 candidate learning curve；FedEvoSAC 的本地 SAC rollout 分数保存在 `client_reward_mean` / `client_reward_std`。`best_fitness` / `archive_best` 仍作为辅助表格或附图指标，用于解释 EA 搜索过程。FedRL 主文建议优先使用 communication rounds 作为横坐标，因为它直接对应联邦通信效率；raw environment steps 和 normalized progress 作为补充视角：

- reward vs communication round / generation：主图，说明联邦通信效率。
- reward vs raw environment steps：补充图，说明样本效率；所有算法应跑到同一个 step budget，提前收敛时曲线保持最后当前评估值。
- reward vs normalized progress：只作为可视化辅助，不作为主定量结论。

最终表格至少报告 `Final return mean +/- std`、`Best return mean +/- std`、`max_steps`、`max_round` 和 `wall_time_sec`。离散 CartPole 只作为 sanity check；核心证据优先放在连续 `Swimmer-v5`、`Reacher-v5` 和 `HalfCheetah-v5`。强异质结论应写成相对改善，而不是声称完全解决异质性退化。

## 12. 当前实现状态

已完成：

- 连续环境主线：`Swimmer-v5`、`Reacher-v5`、`HalfCheetah-v5`；
- `SACPolicy`：tanh Gaussian actor、twin critics、target critics、learnable alpha；
- continuous SAC federated baselines：`FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median`；
- EA genotype actor-only；
- GA actor 前缀可配置且当前固定为 `actor.`；
- reward-aware federated actor aggregation；
- delta clipping 与 bounded EA mutation；
- global elite archive；
- FedEvoSAC 连续对比和消融脚本；
- 离散 `FedEvoFSAC`、`FSACPolicy`、DQN/SAC/FSAC 附线 baseline 保留为 sanity / preliminary。

主要风险：

- 连续控制环境训练方差更大，`HalfCheetah-v5` 需要更长预算和多 seed 统计；
- actor-only 共享更稳，但 client critic 完全本地化，早期本地更新可能噪声较大；
- 异质 client reward scale 会影响 softmax 聚合权重，需要关注 `aggregation_entropy`；
- MuJoCo 异质性过强时会改变最优动作尺度，EA mutation、action noise 和 SAC temperature 需要联动调参；
- 当前 privacy 是 trajectory-private，不是 differential privacy 或 secure aggregation。
