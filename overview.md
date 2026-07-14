# FedEvoSAC / Dist-ERL

## 1. Federated RL 问题设定

本项目当前主线研究连续动作异质控制环境：

```text
Swimmer-v5
Walker2d-v5
Hopper-v5
```

每个 client 拥有自己的私有环境、私有 replay buffer 和本地 SAC learner。client 之间不共享 trajectory，服务器只能接收 actor 参数、reward / fitness 等标量统计信息；critic、target critic、temperature 和 replay buffer 全部留在本地。

异质性来自不同 client 的局部 MDP 扰动：

| 环境 | 动作类型 | client heterogeneity |
|------|----------|----------------------|
| `Swimmer-v5` | continuous | gravity、body mass、joint damping、geom friction、reward scale、observation/action perturbation |
| `Walker2d-v5` | continuous | gravity、body mass、joint damping、geom friction、reward scale、observation/action perturbation |
| `Hopper-v5` | continuous | gravity、body mass、joint damping、geom friction、reward scale、observation/action perturbation |

当前在每个原始环境上定义三种异质联邦场景：

| 场景 | `client_heterogeneity_mode` | 强度 | 含义 |
|------|-----------------------------|------|------|
| `dynamics_mild` | `env_params_only` | `0.25` | 只改变客户端物理参数，如 gravity、mass、length、wind |
| `sensor_reward` | `reward_action_noise` | `0.35` | 原始动力学不变，只改变观测噪声、reward scale/bias 和 seed stream |
| `reward_scale_ablation` | `reward_scale_only` | `1.00` | 原始动力学不变，3 个 worker 的 reward scale 约为 `1/3, 1, 3`，专门检验 raw-softmax 是否会偏向高 reward-scale client |
| `dynamics_reward_scale` | `env_params_reward_scale` | `1.00` | 同时改变动力学参数和 reward scale；训练 client non-IID，但最终仍在统一标准环境上评估 |
| `mixed_hard` | `mixed` | `0.50`-`0.60` | 同时改变动力学参数与观测/奖励/动作扰动，作为最难异质场景；主实验先使用 no-heterogeneity / mild dynamics，hard mixed 作为压力测试 |

这样可以区分四类问题：动力学 non-IID、感知/奖励 non-IID、reward-scale non-IID，以及混合强异质 non-IID。`reward_scale_ablation` / `dynamics_reward_scale` 主要用于 FedEvoSAC 内部消融，不作为唯一主结论环境。

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
Walker2d-v5
Hopper-v5
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
python -m src.main --mode fed_evo_rl --algorithm SAC --env Walker2d-v5
```

## 4. 算法流程

每一轮 generation 中执行：

1. **EA population 跨 client 评估**
   - server 维护一组 actor population，每个个体只包含 `actor.*`；
   - server 将每个 actor 下发给所有 federated clients；
   - 同一 generation 的所有个体使用相同的一组 common random seeds，避免 Reacher 随机目标或 MuJoCo 初始状态差异造成幸运个体；
   - 每个 client 在自己的本地 MDP 中 rollout 评估该 actor；
   - client 只返回 scalar fitness，不上传 trajectory；
   - server 对同一 actor 的多 client fitness 求均值，得到该个体的 federated fitness。

2. **Global elite archive 更新**
   - server 按 generation fitness 对 population 排序；
   - top candidates 在独立固定 validation seeds 上跨 client 复评；
   - 只有独立复评结果可以进入历史 archive；
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
   - client 加载新的 server actor 后先执行少量 critic-only warm-up，再更新 actor 和 temperature，避免未校准 critic 立即破坏 elite actor；
   - client 本地训练 continuous SAC actor、critic1、critic2 和 temperature；
   - critic、target critic、temperature 和 replay buffer 始终留在本地；
   - client 只上传更新后的 actor 参数和本地 reward 摘要。

6. **联邦 actor 聚合**
   - server 对 client 上传的 actor update 做 reward-aware aggregation；
   - 默认使用 normalized relative softmax 权重、低分 client 过滤和 delta norm clipping；
   - 每个 client 维护本地 reward EMA / std，聚合分数使用相对提升 `(reward_i - EMA_i) / std_i`，避免 reward scale 大的 client 天然支配聚合；
   - 聚合时以当前 best actor 为中心聚合 delta，避免直接平均完整 actor 造成大幅漂移。

7. **RL-to-EA soft injection 和 deployable policy**
   - 若聚合 actor 的本地表现达到门控条件，server 将其 soft inject 到 EA population 的弱 non-elite 个体；
   - 通过固定 validation 的聚合 actor 会立即写入 global elite archive，成为可部署候选；
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
| 每代聚合 | `--fed-aggregation-interval 1` | 短预算下保证 SAC 持续参与；长预算可调大 |
| normalized relative softmax | `--fed-score-normalization relative_gain` | 用 client 相对提升而不是 raw reward 计算聚合权重 |
| 温度控制 | `--fed-aggregation-temperature 75` | 防止单个 client 过度支配 |
| reward scale EMA | `--fed-score-ema-beta 0.90` | 维护每个 client 的 reward baseline / scale |
| raw softmax 消融 | `--fed-ablation raw_softmax` | 保留原始 reward softmax，用于验证归一化聚合贡献 |
| 低分过滤 | `--fed-min-client-score-quantile 0.25` | 丢弃低质量 actor update |
| delta clipping | `--fed-delta-clip-norm 5` | 限制 client update 的参数步长 |
| soft injection | `--migration-blend 0.35` | 将聚合 actor 软注入 EA 弱个体 |

聚合形式是以当前 best actor 为中心的 delta aggregation：

```text
delta_i = clip(theta_i - theta_best)
theta_fed = theta_best + sum_i w_i * delta_i
```

这比直接平均完整 actor 更稳。

默认 `FedEvoSAC-full` 使用 normalized relative softmax。`FedEvoSAC-raw_softmax` 作为消融保留旧版 raw reward softmax，用来回答：在 client reward scale 异质时，按相对提升归一化是否比直接用 episodic return 更稳定。

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
FedEvoSAC-raw_softmax
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

四个联邦 SAC baseline 使用相同的稳定聚合协议。client 先连续执行多个本地 rollout/update round，再按各自的 FedAvg、FedBest、Softmax 或 Median 规则产生 actor candidate。server 使用 `server_learning_rate` 将 candidate 与上一全局 actor 小步融合，并在固定 validation seeds 上验收；只有不降低当前全局评估回报的 candidate 才广播。critic、temperature 和 replay buffer 始终保留在 client。主图画已验收的全局 actor，candidate 辅助图画聚合候选，因此既能展示学习过程，也能避免一次不稳定聚合永久破坏策略。

为避免旧实现中“每收集 1000 个 transition 只更新 4 次”的欠训练问题，baseline 按新采样量计算有限的本地 SAC 更新数：`updates = min(max(base_updates, ceil(new_steps * update_to_data_ratio)), max_updates_per_round)`。当前统一协议使用 `update_to_data_ratio=0.05`、上限 `20`。联邦聚合替换 actor 参数后清空 actor optimizer 的陈旧 Adam 动量，并执行少量 critic-only warm-up；本地 critic optimizer、temperature optimizer 和 replay buffer 全部保留。FedAvg、FedBest、FedSoftmax 和 Median 的原聚合思想保持不变。

外部论文原代码复现作为单独的 external-original comparison 管理，不直接和同协议内部基线混名。原因是这些仓库支持的环境、依赖和训练协议不同；能在相同环境上运行的才放入外部复现图。外部复现结果不参与 FedEvoSAC 消融图，也不和内部同协议曲线混称为同一类 baseline。

| 外部代码 | 实际 RL 主体 | 本地路径 | 可比环境 | 使用方式 |
|----------|--------------|----------|----------|----------|
| FedFormer | SAC | `/home/ywj/code/FedFormer` | MetaWorld MT10，不是当前三个 Gymnasium 连续主环境 | related work 或单独 MetaWorld 复现，不混入主图 |
| Byzantine-Federated-RL / FedPG-BR | policy gradient | `/home/ywj/code/Byzantine-Federated-RL` | CartPole-v1、LunarLander-v2 | 可作为离散 external-original 辅助复现 |
| Federated-DRL | DQN / DDQN | `/home/ywj/code/Federated-DRL` | CartPole-v1、LunarLander-v2、Mario | 可作为 FedAvg-DQN 外部原代码复现 |
| FederatedRL | PPO | `/home/ywj/code/FederatedRL` | CartPole-v1、若干 MuJoCo/IoT 任务 | 可作为 PPO-FedRL related work，默认不进连续主图 |

当前可严格复现的外部对照边界：

- `Reacher`：当前外部仓库没有同版本、同协议的严格复现结果，因此只进入本项目内部同协议主图。
- `CartPole / Acrobot / MountainCar`：保留为离散附线；外部 DQN / FedPG 代码可用于 related-work 复现，不进入连续 FedEvoSAC 主图。
- `FedFormer`：原论文是 MetaWorld 连续控制 SAC；若要比较，应另开 MetaWorld 复现实验。

## 10. 实验脚本

连续主线 FedEvoSAC 套件：

```bash
./run_continuous_fedevosac_suite.sh
```

当前三环境主实验使用按环境设置的异质性，并以两个独立 seed 为一个最小统计单元：

```text
CLIENT_HETEROGENEITY=0.0
CLIENT_HETEROGENEITY_MODE=none
NUM_WORKERS=3
SEEDS="0 1"
BUDGET_PRESET=converged
```

默认连续环境：

```text
Swimmer-v5
Walker2d-v5
Hopper-v5
```

HalfCheetah 已从主实验中移除。旧 `perenv_tuned_s0` 中 Swimmer 达到 `242.70`，但三 seed 结果为 `111.91 +/- 92.52`；逐行检查日志后发现，seed 0 找到高回报 actor，而 seed 1/2 停在约 `50/44`，且三个 run 的 `migrated` 都为 `0`。因此旧好结果主要来自一次成功的 EA 搜索，不足以证明稳定性。新协议不把方差仅当成画图问题，而是同时修正远程 actor 随机种子、停滞恢复、SAC refinement 和训练预算。

当前三环境协议因此改为：

| 环境 | horizon | FedEvoSAC population | 联邦频率 | client SAC updates | archive validation |
|------|---------|----------------------|----------|--------------------|-------------------|
| `Swimmer-v5` | `1000` | `10` | 每 2 代 | `64`，前 48 次 critic-only，actor lr `3e-5` | top-2 candidates x 1 episode |
| `Walker2d-v5` | `1000` | `12` | 每 5 代 | `6` | top-2 candidates x 1 episode |
| `Hopper-v5` | `1000` | `12` | 每 4 代 | `8` | top-3 candidates x 2 episodes |

同一环境内所有算法使用相同真实交互预算：三个环境均为 `1,200,000` steps。训练 rollout、EA evaluation、archive validation 和聚合 candidate validation 全部计入预算；提前达到稳定回报的算法仍跑满预算，画图时保持最后一个当前策略回报。所有 EA 个体使用同代 common seeds，archive 和聚合 actor 使用独立固定 validation seeds。FedEvoSAC 的基本原则是：长 horizon locomotion 任务中以 EA actor population 负责全局探索，SAC/federation 低频辅助 refinement。

为降低训练方差而不改变 FedEvoSAC 的核心结构，新版增加以下约束：

- `EAManager` 和每个 Ray `FederatedClient` 显式接收实验 seed；manager 持有私有 Python/NumPy RNG，并将其传入每一代交叉、变异、immigrant 和 injection，避免 `erl_re2_epoch()` 每代临时创建未受控随机源；
- archive 连续若干代未达到最小增益时，保留 archive elite，只重置底部 `25%` 个体并增强中部个体变异；Swimmer patience 为 `4` 代，Walker2d/Hopper 为 `8` 代；
- Swimmer 每 2 代进行一次联邦 refinement，使用更小的 actor learning rate；每次先训练 critic，再做少量 actor 更新，第一次聚合只 warm-up，从第二次开始允许通过独立验证的 actor 注入；
- Swimmer archive 与 injection 采用跨 client 的 `mean - 0.25 * std` 验收，降低单条幸运轨迹进入 archive 的概率；archive 每代只复评 top-2、每 client 1 episode，把更多固定交互预算用于实际 EA/SAC 学习；
- `env_params_only` 的扰动只在环境 wrapper 中执行，client 不再重复施加 reward scale/action noise。

Walker2d 的 `relative_gain` 聚合曾经过于接近 uniform averaging，导致 full 消融不能稳定优于 raw-softmax。当前 Walker2d 默认保留 `temperature=60`，并将归一化聚合分数乘以 `fed_score_scale=8.0`，提高 softmax 的有效选择压力；`FedEvoSAC-raw_softmax` 不使用该缩放，仍作为原始 reward-scale 敏感性的消融对照。

为了降低无意义的 evaluation 方差，同时保留可检验的 FedEvoSAC 消融差异，当前调参版允许每个环境单独设置 client heterogeneity。推荐设置为：

| 环境 | `client_heterogeneity` | `client_heterogeneity_mode` | 目的 |
|------|------------------------|-----------------------------|------|
| `Swimmer-v5` | `0.0` | `none` | Swimmer 对 reward-scale / dynamics perturbation 极敏感；先作为稳定 locomotion sanity 环境 |
| `Walker2d-v5` | `0.0` | `none` | 不额外加强 dynamics heterogeneity；主要通过 `fed_score_scale=8.0` 修正 near-uniform 聚合 |
| `Hopper-v5` | `0.25` | `env_params_only` | 使用 mild dynamics heterogeneity，避免 reward-scale 引入过大的 eval 方差 |

Swimmer 对早期 federation 较敏感。当前只在 `Swimmer-v5` 上启用 warm-up：第一次 federated aggregation 使用 `batch_zscore` 并跳过 injection；第二次起恢复 `relative_gain`、`fed_score_scale=4` 和正常 injection。`FedEvoSAC-raw_softmax` 消融不使用该 warm-up 和 score scaling，保持原始 raw reward softmax 路径。

Reacher 已从主环境中移出。它的短 horizon 和 dense distance reward 更适合作调试 SAC 稳定性，不适合作为 EA+FedSAC 的核心证据：FedEvoSAC 的 population search 优势容易被短任务的快速局部优化掩盖，且 evaluation variance 会显著影响结论。当前改用 `Walker2d-v5`，它同样是 MuJoCo 连续控制，但 horizon 更长、动作维度更高、步态探索更依赖 actor 多样性，更适合检验 EA + federated SAC。Hopper 的 `1000+` 回报在 MuJoCo Hopper 中并非异常上界，但仍偏中等，因此 Hopper 保留为可继续提分的 locomotion 任务。

正式重复实验使用：

```bash
EXPERIMENT_ID=fedevosac_20x2_converged_20260714 \
  ./scripts/run_fedrl_20x2_experiment.sh
```

`20x2` 表示 20 个 outer repeat，每个 repeat 恰好两个 seed。不同 repeat 使用不同 seed pair：`(0,1), (2,3), ..., (38,39)`，所以每个 repeat 的图是 `n=2`，最终 aggregate 是 40 个独立 seed；不会把完全相同的 seed 重跑 20 次后伪装成更大的独立样本量。脚本支持断点重启，已达到目标 steps 的 run 会自动跳过。

调度器默认每批并行运行 4 个 repeat；每个 repeat 的 stdout 独立写入 `logs/experiments/<experiment_id>/repeat_XX.log`，批内任务全部结束后才重画 aggregate，避免并发覆盖结果。可通过 `PARALLEL_REPEATS` 调整并发度，通过 `START_REPEAT` / `END_REPEAT` 做分片或断点续跑。

每个 repeat 的横向比较运行 `FedEvoSAC-full` 和四个联邦 SAC baseline；独立消融图复用 full，并额外运行 `uniform_aggregation`、`no_local_rl`、`no_ea_injection`、`raw_softmax`。`no_heterogeneity` 不进入本轮统一三环境消融，因为 Swimmer/Walker2d 的主设置本身就是无额外异质扰动，在这两个环境上该曲线会与 full 重复；它只适合另行用于 Hopper 异质性实验。

新结果只写入 `plots_new`，目录结构为：

```text
plots_new/<experiment_id>/reference_single_seed/  # 旧 perenv_tuned_s0 图，仅供视觉参考
plots_new/<experiment_id>/repeats/repeat_XX/      # 每个 n=2 repeat 的图、表与收敛检查
plots_new/<experiment_id>/aggregate/main/         # round 主对比图
plots_new/<experiment_id>/aggregate/supplement/   # steps 样本效率图
plots_new/<experiment_id>/aggregate/diagnostics/  # progress 形状诊断图
plots_new/<experiment_id>/aggregate/ablation/     # 独立消融图
plots_new/<experiment_id>/aggregate/paper_figures/# 三环境横向拼图 PNG/PDF
plots_new/<experiment_id>/aggregate/tables/       # final、best 与 convergence CSV
```

旧 `fedevosac_perenv_tuned_s0_comparison` 会原样复制到 `reference_single_seed`，但明确标记为 single-seed reference，不参加新均值、置信区间或显著性统计。

日志与图表目录约定：

```text
logs/                 # 所有 metrics、metadata、shell run log
logs/run/             # 后台/脚本 stdout 日志
plots/                # 所有对比图、消融图、表格
plots_new/            # 新 20x2 实验，和历史 plots 完全分开
plots/training/       # src.main 生成的单次训练过程图
plots/tables/         # 零散 summary / significance CSV
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

该脚本只调用 `/home/ywj/code` 下的外部仓库，并把输出整理到 `logs/external_original_logs/`。它依赖外部仓库自己的 Python/Gym/PyTorch 版本；如果当前 conda 环境不兼容，应单独建对应环境后用 `EXTERNAL_PYTHON=/path/to/python ./run_external_original_baselines.sh` 运行。

快速 smoke：

```bash
python -m src.main \
  --env Walker2d-v5 \
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
| `aggregation_score_mean` | 进入聚合 softmax 的归一化 client score 均值 |
| `aggregation_score_std` | 进入聚合 softmax 的归一化 client score 方差 |
| `archive_best` | 独立固定 validation seeds 上的 global elite archive 最佳回报 |
| `candidate_eval_mean` | 当前聚合训练 actor 的固定协议评估回报 |
| `deployable_eval_mean` | server 历史最佳可部署 actor 的评估回报 |
| `local_updates_per_worker` | 每轮每个 baseline client 实际执行的本地 SAC 更新数 |
| `communication_round` | 实际 server-client 协调轮数；FedEvoSAC 每代 population 分发/fitness 回传计 1，baseline 仅 actor 聚合时计 1 |
| `comm_upload_bytes` | 实际上传 actor 参数量估计 |
| `comm_full_traj_bytes` | 假设上传完整 trajectory 的通信量估计 |

主曲线默认使用 `eval_reward_mean`，表示 server 当前可部署策略。baseline 的当前训练策略记录在 `candidate_eval_mean`，并单独生成 candidate learning curve；FedEvoSAC 的本地 SAC rollout 分数保存在 `client_reward_mean` / `client_reward_std`。`best_fitness` / `archive_best` 只作为辅助表格或附图指标，用于解释 EA 搜索过程。

三种横坐标回答的是不同问题，不能互相替代：

| 横坐标 | 回答的问题 | 论文位置 | 限制 |
|--------|------------|----------|------|
| communication round | 达到某个 return 需要多少次真实 server-client 协调，即通信效率 | 主图 | FedEvoSAC 与 baseline 每轮的 payload 和环境交互量不同，不能单独证明样本或字节效率 |
| counted environment interactions | 在相同真实采样预算下谁学得更快，即样本效率 | 补充主证据 | 必须统计 population、local rollout、archive/candidate validation 的全部交互 |
| normalized progress (%) | 从该 run 第一次评估的 `0%` 到最后评估的 `100%`，比较学习形状、停滞点和末段稳定性 | 诊断/附录 | 会抹去绝对预算差异，不能据此宣称更高效率 |

这种主图/补充图分层与 FRL 文献的常见结构一致。[Federated Reinforcement Learning with Environment Heterogeneity](https://proceedings.mlr.press/v151/jin22a.html) 在复杂任务中画 averaged return vs episodes/frames，并以均值和 `1.65 x standard error` 阴影展示不确定性，同时单独研究 local-update interval 对通信频率的影响；[Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling](https://proceedings.mlr.press/v162/khodadadian22a.html) 则把 environment iterations/sample complexity 与 communication cost 分开分析。本项目因此输出三环境横向 panel：round 是 main figure，steps 是 supplementary evidence，progress 是 diagnostics。旧日志没有 `communication_round` 时绘图器会回退到 `generation`，仅用于历史图兼容；新实验不使用该回退口径。

新图报告 current policy 的跨 seed 均值和 90% normal-approximation CI（样本标准差 `ddof=1`，阴影为 `1.645 x standard error`），不混入 archive-best 曲线；原始每 seed CSV、seed standard deviation 和无方差图仍保留，可复核置信带。单个 repeat 只有两个 seed，其区间只作运行诊断；正式 aggregate 使用 40 个独立 seed。平滑只作用于显示曲线，不改 summary CSV。每个 run 额外生成 `convergence_report.csv`：最后 6 个评估点的增益和范围必须落在绝对/相对容差内，未通过者不能在论文中标为 converged。

最终表格至少报告 `Final return mean +/- std`、`Best return mean +/- std`、`max_steps`、`max_round`、`wall_time_sec` 和 convergence 状态。当前连续证据来自 `Swimmer-v5`、`Walker2d-v5` 和 `Hopper-v5`。

## 12. 当前实现状态

已完成：

- 连续环境主线：`Swimmer-v5`、`Walker2d-v5`、`Hopper-v5`；
- `SACPolicy`：tanh Gaussian actor、twin critics、target critics、learnable alpha；
- continuous SAC federated baselines：`FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median`；
- EA genotype actor-only；
- GA actor 前缀可配置且当前固定为 `actor.`；
- normalized relative reward-aware federated actor aggregation；
- delta clipping 与 bounded EA mutation；
- global elite archive；
- FedEvoSAC 连续对比和消融脚本；
- 离散 `FedEvoFSAC`、`FSACPolicy`、DQN/SAC/FSAC 附线 baseline 保留为 sanity / preliminary。

主要风险：

- 连续控制环境训练方差更大，Hopper 仍需要更长预算和多 seed 统计；
- actor-only 共享更稳，但 client critic 完全本地化，早期本地更新可能噪声较大；
- raw reward softmax 容易受异质 client reward scale 影响；当前 full 已改用 normalized relative softmax，但仍需要通过 `raw_softmax` 消融、`aggregation_entropy` 和 `aggregation_score_std` 验证稳定性；
- MuJoCo 异质性过强时会改变最优动作尺度，EA mutation、action noise 和 SAC temperature 需要联动调参；
- 当前 privacy 是 trajectory-private，不是 differential privacy 或 secure aggregation。
