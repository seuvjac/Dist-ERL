# FedEvoSAC / Dist-ERL

> 最后核对：2026-09-03。当前主线是连续动作 `FedEvoSAC`。新版正式协议使用
> Walker2d-Locomotion、Hopper-Locomotion、Ant-v5、HalfCheetah-v5 和 Swimmer-v5，
> 输出 current return 对 communication rounds / counted interactions 的均值曲线与双侧
> 95% Student-t CI，不再生成 normalized training-progress 图。正式统计使用运行前固定的
> 30 个 held-out seed；禁止按 Full 得分事后筛选 seed。5 个运行前固定 seed 只允许用于
> 个体轨迹示例和敏感度分析，不能替代 n=30 汇总与显著性检验。

## 1. Federated RL 问题设定

本项目研究异质 client 下的连续动作 Federated RL。当前正式实验使用五个 MuJoCo 环境：

```text
Walker2d-Locomotion (基于 Walker2d-v5)
Hopper-Locomotion (基于 Hopper-v5)
Ant-v5
HalfCheetah-v5
Swimmer-v5
```

每个 client 拥有自己的私有环境、私有 replay buffer 和本地 SAC learner。client 之间不共享 trajectory，服务器只能接收 actor 参数、reward / fitness 等标量统计信息；critic、target critic、temperature 和 replay buffer 全部留在本地。

2026-09-03 协议重新纳入 Ant、HalfCheetah 和 Swimmer，用于扩大任务覆盖；这不会抹去旧实验发现的风险。Swimmer 的历史 seed 方差较大，Ant 的默认 healthy reward 可能形成生存捷径，HalfCheetah 的高维 actor 搜索可能收敛较慢，因此三者必须完整报告全部预注册 seed、95% CI、收敛率和 locomotion diagnostics，不能只展示有利运行。Walker2d 和 Hopper 启用 `env_params_only` dynamics heterogeneity，强度分别为 `0.30` 和 `0.25`；Ant、HalfCheetah 和 Swimmer 分别使用 `0.15`、`0.15` 和 `0.12`。`Walker2d-Locomotion` 与 `Hopper-Locomotion` 是显式 reward 变体，不与 Gymnasium 默认成绩直接混合比较。

异质性来自不同 client 的局部 MDP 扰动：

| 环境 | 动作类型 | client heterogeneity |
|------|----------|----------------------|
| `Walker2d-Locomotion` | continuous | 基于 `Walker2d-v5`；gravity、左右腿质量/惯量、左右关节阻尼、脚底摩擦、左右执行器 gear；`healthy_reward=0.05`、`forward_reward_weight=1.0` |
| `Hopper-Locomotion` | continuous | 基于 `Hopper-v5`；gravity、body mass、joint damping、geom friction；`healthy_reward=0.05`、`forward_reward_weight=1.0` |
| `Ant-v5` | continuous | gravity、body mass、joint damping、geom friction；`0.15 / env_params_only` |
| `HalfCheetah-v5` | continuous | gravity、body mass、joint damping、geom friction；`0.15 / env_params_only` |
| `Swimmer-v5` | continuous | body mass、joint damping、geom friction；`0.12 / env_params_only` |

框架支持以下五种异质联邦场景：

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

当前连续正式主实验环境：

```text
Walker2d-Locomotion (基于 Walker2d-v5)
Hopper-Locomotion (基于 Hopper-v5)
Ant-v5
HalfCheetah-v5
Swimmer-v5
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
   - server 维护一组 actor population；连续 SAC 个体保存完整 `actor.*` 以便加载，但 EA 的有效 genotype 只包含 `actor.net.*` 与 `actor.mean.*`；
   - server 将每个 actor 下发给所有 federated clients；
   - 同一 generation 的所有个体使用相同的一组 common random seeds，降低 MuJoCo 初始状态差异造成的幸运个体偏差；
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
   - 对 deterministic mean policy 的二维权重矩阵做 row-wise crossover；
   - 对 non-elite actor 执行 layer-RMS normalized bounded mutation、super mutation 和 reset mutation，weight row 与对应 bias 一起变化，并用非零 scale floor 防止近零输出层无法移动；
   - SAC 的 `actor.log_std.*` 不参与 EA fitness 对应的交叉/变异；critic、target critic、temperature 和 replay buffer 同样不进化。

4. **Elite restore**
   - GA 结束后，将 global elite archive 中的 top actor hard restore 回 population 前部；
   - 这样 archive elite 会继续参与下一代评估，也能作为本轮 SAC 本地更新的 warm-start actor。

5. **本地 SAC refinement**
   - 每隔 `K` 代，server 将当前 best / archive elite actor 下发给部分 clients；
   - client 用该 actor 初始化本地 SAC actor；
   - client rollout 产生 transition，写入私有 replay buffer；
   - client 加载新的 server actor 后先执行少量 critic-only warm-up，再更新 actor 和 temperature，避免未校准 critic 立即破坏 elite actor；
   - client 本地训练 continuous SAC actor、critic1、critic2 和 temperature，其中 `log_std` 与 temperature 保持 client-local；
   - 本地更新后，client 用相同 held-out seed 比较 server actor 与 local candidate；candidate 退化时回退到 server actor；
   - critic、target critic、`log_std`、temperature 和 replay buffer 始终留在本地；
   - client 只上传通过验收的 `actor.net.*` / `actor.mean.*` 和本地 reward 摘要。

6. **联邦 actor 聚合**
   - server 对 client 上传的 deterministic mean actor update 做 reward-aware aggregation；
   - Walker2d-Locomotion 和 Hopper 均使用 normalized relative-gain softmax，并保留低分 client 过滤和 delta norm clipping；
   - 每个 client 维护 reward EMA / std，聚合分数使用 `(reward_i - EMA_i) / std_i`；batch-zscore 只作为聚合敏感性诊断，不进入正式 Full；
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

当前连续 SAC 实现将 GA 的包含前缀与排除项配置为：

```text
actor_prefix = "actor."
actor_exclude_substrings = ("actor.log_std.",)
```

正式环境统一使用 `layer_rms` mutation、非零 scale floor 和 bias mutation，并通过 `--ea-weight-clip` 限制 actor 参数范围。该设计让 EA 搜索 deterministic policy mean，让 SAC 的随机探索尺度由各 client 的局部 critic/temperature 学习信号负责。

## 8. 联邦聚合

FedEvoSAC 聚合的是 client 上传的 actor 参数，不聚合 trajectory，也不聚合 critic。

当前正式协议的聚合机制：

| 机制 | 当前参数 | 作用 |
|------|----------|------|
| 聚合周期 | Walker2d/Hopper/Ant/HalfCheetah/Swimmer 为 `5/4/4/4/2` 代 | 在长 horizon 任务中控制 SAC refinement 与 EA 评估的预算占比 |
| score normalization | 五个环境均为 `relative_gain` | 按各 client 相对自身历史的提升计算权重，降低 reward scale 差异的直接影响 |
| normalized temperature | Walker2d/Hopper/Ant/HalfCheetah/Swimmer 为 `8/1/60/4/4` | 环境维度和 reward range 不同，因此预注册环境级温度，不在 seed 结果出现后修改 |
| score scale | Walker2d/Hopper/Ant/HalfCheetah/Swimmer 为 `8/1/4/4/4` | 在保持归一化的同时调节 client 权重区分度 |
| reward scale EMA | `--fed-score-ema-beta 0.90` | 为每个正式环境维护每个 client 的 reward baseline / scale |
| federation warm-up | 前两次聚合使用 warm-up score，并跳过 injection | 避免本地 critic 尚未校准时破坏 archive elite |
| raw softmax 敏感性对照 | `--fed-ablation raw_softmax` | 保留原始 reward softmax，用于单独筛选聚合策略，不进入模块消融图 |
| 低分过滤 | `--fed-min-client-score-quantile 0.25` | 丢弃低质量 actor update；uniform 消融不执行该过滤 |
| delta clipping | Walker2d/Hopper/Ant/HalfCheetah/Swimmer 为 `4/0.5/4/1/3` | 限制 client update 的全局 L2 参数步长 |
| soft injection | 环境级低噪声、受限 blend，Ant/HalfCheetah/Swimmer 每次最多迁移 1 个个体 | 将通过独立验证的聚合 actor 注入 EA，同时避免复制过多导致种群塌缩 |

聚合形式是以当前 best actor 为中心的 delta aggregation：

```text
delta_i = clip(theta_i - theta_best)
theta_fed = theta_best + sum_i w_i * delta_i
```

这比直接平均完整 actor 更稳。聚合器直接消费上游明确生成的 raw / batch-zscore / relative-gain score，不再在 `aggregate_weight_dicts()` 内二次 z-score；因此 `fed_score_scale` 和 raw-softmax 的定义均与日志一致。

正式 `FedEvoSAC-full` 在五个环境都使用 `relative_gain`。`batch_zscore`、`raw_softmax` 与 `uniform_aggregation` 作为聚合敏感性对照保留，用来回答不同归一化方式是否比直接用 episodic return 或均匀平均更稳定。模块消融图仍只包含 Full、no-local-RL、no-EA-injection 和 no-heterogeneity；异质性强度敏感度分析单独出图，不与模块消融混线。

## 9. Baseline 和对比曲线

连续主实验只比较 continuous SAC 系方法：

```text
FedAvg-SAC
FedBest-SAC
FedSoftmax-SAC-noEA
RobustFed-SAC-Median
FedEvoSAC
```

FedEvoSAC 正式模块消融为：

```text
FedEvoSAC-full
FedEvoSAC-no_local_rl
FedEvoSAC-no_ea_injection
FedEvoSAC-no_heterogeneity
```

聚合策略敏感性实验单独比较：

```text
FedEvoSAC-relative_gain
FedEvoSAC-batch_zscore
FedEvoSAC-raw_softmax
FedEvoSAC-uniform_aggregation
```

仓库仍保留早期离散 `FSACPolicy`、`FedEvoFSAC` 和 DQN 代码用于历史复核，但它们已经退出当前实验协议，不参与当前主图、消融、统计结论或默认脚本。当前 overview 中的 `SAC` 均指 continuous SAC，除非明确标为 legacy discrete。

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

四个联邦 SAC baseline 使用相同的稳定聚合协议。client 先连续执行多个本地 rollout/update round，再按各自的 FedAvg、FedBest、Softmax 或 Median 规则产生 actor candidate。server 使用 `server_learning_rate` 将 candidate 与上一全局 actor 小步融合，并在固定 validation seeds 上验收；只有不降低当前全局评估回报的 candidate 才广播。训练和 validation 都使用与 FedEvoSAC 相同的异质 client suite，不再在同质环境中验收异质环境训练出的 actor。critic、temperature 和 replay buffer 始终保留在 client。主图画已验收的全局 actor，candidate 辅助图画聚合候选，因此既能展示学习过程，也能避免一次不稳定聚合永久破坏策略。

为避免旧实现中“每收集 1000 个 transition 只更新 4 次”的欠训练问题，baseline 按新采样量计算有限的本地 SAC 更新数：`updates = min(max(base_updates, ceil(new_steps * update_to_data_ratio)), max_updates_per_round)`。通用默认值仍为 `update_to_data_ratio=0.05`、上限 `20`；Walker locomotion pilot 使用 `0.25`、上限 `256`、基础更新 `64`、actor learning rate `1e-4`，并在 actor 同步后进行 `32` 次 critic-only warm-up。联邦聚合替换 actor 参数后清空 actor optimizer 的陈旧 Adam 动量；本地 critic optimizer、temperature optimizer 和 replay buffer 全部保留。FedAvg、FedBest、FedSoftmax 和 Median 的原聚合思想保持不变。

外部论文原代码复现作为单独的 external-original comparison 管理，不直接和同协议内部基线混名。原因是这些仓库支持的环境、依赖和训练协议不同；能在相同环境上运行的才放入外部复现图。外部复现结果不参与 FedEvoSAC 消融图，也不和内部同协议曲线混称为同一类 baseline。

| 外部代码 | 实际 RL 主体 | 本地路径 | 可比环境 | 使用方式 |
|----------|--------------|----------|----------|----------|
| FedFormer | SAC | `/home/ywj/code/FedFormer` | MetaWorld MT10，不是当前五个 Gymnasium 连续主环境 | related work 或单独 MetaWorld 复现，不混入主图 |
| Byzantine-Federated-RL / FedPG-BR | policy gradient | `/home/ywj/code/Byzantine-Federated-RL` | CartPole-v1、LunarLander-v2 | 可作为离散 external-original 辅助复现 |
| Federated-DRL | DQN / DDQN | `/home/ywj/code/Federated-DRL` | CartPole-v1、LunarLander-v2、Mario | 可作为 FedAvg-DQN 外部原代码复现 |
| FederatedRL | PPO | `/home/ywj/code/FederatedRL` | CartPole-v1、若干 MuJoCo/IoT 任务 | 可作为 PPO-FedRL related work，默认不进连续主图 |

当前可严格复现的外部对照边界：

- `Reacher`：当前外部仓库没有同版本、同协议的严格复现结果；它只保留为内部调试环境，已经退出当前五环境主图。
- `CartPole / Acrobot / MountainCar`：只保留为 legacy discrete / related-work 复核，不进入当前连续 FedEvoSAC 主图或正式统计。
- `FedFormer`：原论文是 MetaWorld 连续控制 SAC；若要比较，应另开 MetaWorld 复现实验。

## 10. 实验脚本

连续主线 FedEvoSAC 套件：

```bash
./run_continuous_fedevosac_suite.sh
```

新版正式实验以单个独立 seed 为最小审计单元，后台可并行调度 3 个 seed。主脚本预先固定 seed `100..129`，全部 30 个 seed 都进入汇总：

```text
NUM_WORKERS=3
SEED_BASE=100
NUM_SEEDS=30
BUDGET_PRESET=converged
```

默认连续环境：

```text
Walker2d-Locomotion (脚本参数仍为 Walker2d-v5)
Hopper-Locomotion (脚本参数仍为 Hopper-v5)
Ant-v5
HalfCheetah-v5
Swimmer-v5
```

旧 `perenv_tuned_s0` 中 Swimmer 达到 `242.70`，但三 seed 结果为 `111.91 +/- 92.52`，随后历史 40-seed 实验也只有 `26/40` 个 run 尾部稳定。新版按用户要求重新纳入 Swimmer 和 HalfCheetah，但必须把这一既有风险写入结果分析，并以完整 30-seed 结果决定其是否能支撑主结论。不得通过挑选 5 个高分 seed 掩盖失败运行。

正式 30-seed 启动入口：

```bash
./scripts/run_fedrl_30seed_experiment.sh
```

异质性强度敏感度分析使用运行前固定的 5 个独立 seed 和 5 个强度水平，只运行 Full：

```bash
./scripts/run_fedrl_sensitivity_analysis.sh
```

已完成的 `20260714` 历史三环境运行协议如下，数值已由最终 run 的 `metadata.json` 复核，不会回写或混入新版结果：

| 环境 | clients | horizon | population | active heterogeneity | 聚合周期 | temperature / score scale | SAC updates / critic warm-up | archive validation |
|------|---------|---------|------------|----------------------|----------|---------------------------|------------------------------|-------------------|
| `Swimmer-v5` | `3` | `1000` | `10` | `0.0 / none` | 每 2 代 | `75 / 4` | `64 / 48`，actor lr `3e-5` | top-2 x 1 episode，`mean - 0.25 std` |
| `Walker2d-v5` | `3` | `1000` | `12` | `0.0 / none` | 每 5 代 | `60 / 8` | `6 / 3` | top-2 x 1 episode |
| `Hopper-v5` | `3` | `1000` | `12` | `0.25 / env_params_only` | 每 4 代 | `50 / 1` | `8 / 4` | top-3 x 2 episodes |

同一环境内所有算法使用相同目标交互预算：三个环境均为 `1,200,000` counted environment interactions。训练 rollout、EA evaluation、archive validation 和聚合 candidate validation 全部计入预算；提前稳定的算法仍继续到目标预算，并保持当前 deployable policy。由于 rollout / validation 是不可拆分批次，实际最后记录会小幅越过目标：最终范围为 Swimmer `1.203M-1.212M`、Walker2d `1.200M-1.229M`、Hopper `1.200M-1.242M`，不能写成逐步完全相等。所有 EA 个体使用同代 common seeds，archive 和聚合 actor 使用独立固定 validation seeds。FedEvoSAC 的基本原则是：长 horizon locomotion 任务中以 EA actor population 负责全局探索，SAC/federation 低频辅助 refinement。

为降低训练方差而不改变 FedEvoSAC 的核心结构，新版 Walker2d/Hopper 正式协议增加以下约束：

- `EAManager` 和每个 Ray `FederatedClient` 显式接收实验 seed；manager 持有私有 Python/NumPy RNG，并将其传入每一代交叉、变异、immigrant 和 injection，避免 `erl_re2_epoch()` 每代临时创建未受控随机源；
- EA population 使用 `anchor_antithetic`：一个个体保持标准 SAC anchor，其余个体围绕 anchor 做成对的 `+delta/-delta` 层尺度扰动；`log_std` 不扰动。population 固定为 `13`，由一个 anchor 和六对 antithetic 个体组成；
- archive 连续若干代未达到最小增益时，保留 archive elite，并围绕 archive elite 对底部个体做受控增强变异，不再重新生成整网高斯随机 actor；最多 boost `2` 次，两次至少间隔 `12` 代；
- Hopper 每 client 执行 `96` 次本地 SAC 更新，其中前 `88` 次 critic-only warm-up，actor learning rate 为 `3e-5`；Walker locomotion pilot 使用 `256 / 192 / 3e-5`。两者都先校准 critic，再以较小 actor learning rate 更新；前两次聚合均跳过 injection；
- 每个 client 在上传前对 base/local actor 做 common-seed deterministic validation；local actor 低于 base 时上传 base actor，并记录 `local_candidate_accept_rate` 与 `local_candidate_gain_mean`；
- archive 与 injection 采用跨 client risk-adjusted score 验收，风险惩罚系数为 `0.25`，降低幸运轨迹进入 archive 的概率；
- `env_params_only` 的扰动只在环境 wrapper 中执行，client 不再重复施加 reward scale/action noise。

Walker2d 的 `relative_gain` 在早期聚合中出现过 `aggregation_score_std=367` 和 entropy `0`，说明短历史 EMA/std 可能让单 client 垄断权重。3-seed 筛选显示 batch-zscore 更高且更稳，但正式协议按当前算法设计决定继续保留 relative-gain；因此该风险必须通过新版 30-seed 正式结果和 entropy 诊断如实检验。Walker normalized/raw temperature 为 `8 / 60`，score scale 为 `8.0`；Hopper分别为 `1 / 50` 和 `1.0`。

为了降低无意义的 evaluation 方差，同时保留可检验的 FedEvoSAC 消融差异，当前调参版允许每个环境单独设置 client heterogeneity。推荐设置为：

| 环境 | `client_heterogeneity` | `client_heterogeneity_mode` | 目的 |
|------|------------------------|-----------------------------|------|
| `Walker2d-Locomotion` | `0.30` | `env_params_only` | 中间 client 保持原始动力学，两侧 client 使用镜像左右腿质量/惯量、阻尼、摩擦、motor gear 和小幅重力差异；所有方法统一使用 `healthy_reward=0.05`、`forward_reward_weight=1.0` |
| `Hopper-Locomotion` | `0.25` | `env_params_only` | 使用 mild dynamics heterogeneity；所有消融统一使用 `healthy_reward=0.05`、`forward_reward_weight=1.0`，避免站立策略依靠 healthy reward 获胜 |

旧 Walker `0.15` profile 只统一缩放全身质量、重力、阻尼和摩擦，对稳定站立策略影响很小。20260805 批次的 8 个完整 seed 中，各消融最终回报集中在约 `980-1007`；Walker2d 默认每个 healthy timestep 提供 `+1`，1000-step horizon 因此可能把“站满 1000 步但几乎不前进”误判为收敛。20260806 的第一版 gait-structured pilot 仍使用 Gymnasium 默认 reward，并在每次 archive evaluation 记录 `episode_length`、`forward_return`、`survive_return`、`ctrl_return`、`x_displacement` 和 `x_velocity`。

`fedevosac_walker_gait_hetero_pilot_3seed_20260806` 使用 `0.30` profile、默认 reward、seed `0/1/2` 和约 300k counted interactions。最终分别为：`1044.93 / 1000 steps / x_vel 0.049`、`333.16 / 220 steps / x_vel 0.526`、`896.45 / 813 steps / x_vel 0.147`。这说明结构化异质性能够区分“长时间站立”和“短暂前进”，但三 seed 回报为 `758.18 +/- 375.49`，方差高于旧 `0.15` profile 在相近预算下的 `807.49 +/- 185.33`。因此继续增大异质性不能替代对 healthy-reward 局部最优和 SAC 更新不足的修复。

第二版 pilot 将 `healthy_reward` 从 `1.0` 降为 `0.2`，保持 `forward_reward_weight=1.0`，同时增加 critic warm-up 和 SAC update-to-data ratio。`fedevosac_walker_locomotion_v2_pilot_3seed_20260806` 在约 300k counted interactions 下得到总回报 `172.52 +/- 22.98`，但速度为 `0.474 +/- 0.431`：seed 1/2 达到 `0.729 / 0.716`，seed 0 却站满 1000 步并以速度 `-0.024` 后退，其 `200` 分 healthy reward 仍足以覆盖前进不足。因此 `0.2` 降低了总分方差，却没有彻底消除 survival shortcut。

当前第三版 `Walker2d-Locomotion` 将 `healthy_reward` 进一步降为 `0.05`，使站满 1000 步最多只获得 `50` 分 healthy reward。对第二版轨迹的离线重算显示，seed 0 第 7 代的正向策略约为 `124.9` 分，而后来的站立策略约为 `24.7` 分，因此 archive 不应再用站立策略淘汰行走策略。它仍使用 Gymnasium 官方 Walker2d 参数接口，但属于新的任务定义；论文、图题、表格和 metadata 必须写明两个 reward 参数，不能简称为标准 `Walker2d-v5`，也不能与旧默认 reward 或 `0.2` 数据合并。所有 FedEvoSAC、baseline 和消融方法使用完全相同的任务参数与异质 client validation suite，因此该修改用于消除共同的生存奖励捷径，不给主算法单独加分。

Hopper 正式 40-seed 旧协议也出现同类问题：Full 的最终总回报为 `932.25 +/- 184.81`，其中平均 `906.25` 来自存活、只有 `26.65` 来自前进；平均速度为 `0.049`。三个消融的高分同样主要来自站立，而且 Full 反而显著低于 `no_ea_injection`、`no_local_rl` 和 `no_heterogeneity`。因此旧 Hopper 消融结论作废。新版 `Hopper-Locomotion` 与 Walker 一样使用 `healthy_reward=0.05`、`forward_reward_weight=1.0`，站满 1000 步最多获得约 50 分存活奖励；主表必须同时报告 total return、forward return、episode length、x displacement 和 x velocity。Full 与所有消融共享完全相同的任务参数，旧默认 reward 数据不得拼接进新曲线。

第一版 `fedevosac_hopper_locomotion_ablation_pilot_20260825` 使用 seed `0/1/2` 和 300k counted interactions。Full 为 `135.49 +/- 13.87`、平均速度 `1.120 +/- 0.190`，证明 survival shortcut 已消除；但 `no_ea_injection=151.02 +/- 18.49`、`no_local_rl=150.80 +/- 18.63`，Full 没有形成模块优势。诊断显示被本地 validation 拒绝的 client 会上传原 server actor，聚合后得到与 EA elite 相同的候选；旧逻辑仍可能把它计作 RL injection 并对弱个体加噪。注入门槛又把 archive-seed 上的 candidate risk score 与 generation-seed 上的 current fitness 比较，放行标准不一致。

第二版只修正这一数据流，不改变环境、EA 结构和交互预算：服务器仅聚合 `candidate_accepted=1` 的真实本地更新；没有通过验证的 client 时跳过聚合和注入；candidate 必须在相同 archive validation seeds 上比 risk-adjusted archive score 至少高 `1%`；Hopper 每次只迁移 `1` 个弱个体，blend/noise 降为 `0.25/0.002`；本地 SAC 使用 `96/88/3e-5` 和 2-episode common-seed validation。新增 `accepted_client_uploads` 与 `injection_reference_score` 用于审计，避免把原 actor 回传误记为 EA+RL 贡献。

第二版最终为 Full `151.22 +/- 18.27`、`no_ea_injection=151.02 +/- 18.49`、`no_local_rl=150.80 +/- 18.63`，三条曲线基本重合，且三个 seed 合计只有一次真实迁移。继续审计发现联邦本地训练从 generation-common-seed 的当前最高个体出发，而 deployable 主曲线和注入门槛来自 archive-validation-seed 的 archive 最优；两个 actor 可能不同。这会出现“本地 candidate 相对其起点改善，但仍远低于 deployable archive”的口径错位。

第三版将 FedSAC 起点固定为 independently validated archive actor，并把 client delta clipping 修正为整模型全局 L2 norm，而不是逐张量分别裁剪；Hopper 使用 `global_delta_clip_norm=0.5`。前 4 个联邦轮只更新 critic，之后 client upload 使用 `0.15` trust-region blend；client validation 允许相对 base 最多 `3%` 的候选进入服务器候选池，但最终注入仍必须在全客户端 archive validation 上不低于现有 risk-adjusted archive。候选池同时比较 relative-gain softmax actor 与各 accepted client actor，吸收 FedSoftmax 和 FedBest 的优点。无真实参数变化时禁止迁移，避免把 archive 原 actor 加噪后误记为 RL 贡献。新增 `accepted_delta_norm_mean/max`、`candidate_pool_size/source` 记录实际更新幅度和候选来源。

`fedevosac_walker_locomotion_v3_pilot_3seed_20260806` 使用 seed `0/1/2` 和约 300k counted interactions。最终回报为 `135.31 / 138.37 / 138.53`，即 `137.40 +/- 1.81`；`x_velocity` 为 `0.564 / 0.552 / 0.542`，即 `0.553 +/- 0.011`；`x_displacement` 为 `1.003 +/- 0.008`。相比第二版 `x_velocity=0.474 +/- 0.431`，第三版消除了本次三个 seed 中的站立策略，并显著降低步态方差。

同一 profile 的小规模横向与消融筛选已经完成。`FedEvoSAC-full` 使用三个 seed；其余方法当前都只有 seed 0，因此下面的单 seed 数值只能诊断实现和学习过程，不能用于显著性或方差结论：

| 类型 | 方法 | 300k final current return |
|------|------|---------------------------|
| proposed | `FedEvoSAC-full` | `137.40 +/- 1.81` (`n=3`) |
| baseline | `FedAvg-SAC` | `135.90` (`n=1`) |
| baseline | `FedBest-SAC` | `136.44` (`n=1`) |
| baseline | `FedSoftmax-SAC-noEA` | `135.13` (`n=1`) |
| baseline | `RobustFed-SAC-Median` | `136.89` (`n=1`) |
| aggregation diagnostic | `uniform_aggregation` | `147.90` (`n=1`) |
| aggregation diagnostic | `raw_softmax` | `177.22` (`n=1`) |
| ablation | `no_local_rl` | `130.50` (`n=1`) |
| ablation | `no_ea_injection` | `106.98` (`n=1`) |
| control | `no_heterogeneity` | `139.58` (`n=1`) |

这轮结果验证了 locomotion task 修复，却没有验证当时的 normalized-relative-gain 聚合：full 在最终回报上只与四个 baseline 持平，并且约到 220k interactions 才接近其最终水平；四个 baseline 在约 20k--70k 内就取得 130 以上的 deployable checkpoint。`raw_softmax` 和 uniform 单 seed 又明显高于 full，说明仅含 dynamics heterogeneity、没有 reward-scale 扰动时，relative-gain normalization/temperature 可能削弱了有效 client 排序。另一方面，`no_ea_injection` 明显下降，`no_local_rl` 也较低，提示 EA-to-RL injection 与本地 SAC refinement 值得保留并扩大 seed 验证。baseline 的 current 曲线主要由 rollback checkpoint 保持稳定，其 candidate 策略仍多次退化，因此后续必须同时报告 deployable current 和 candidate，而不能只凭平坦主曲线声称训练稳定。

随后使用开发 seed `0/1/2`、相同 300k interaction 预算完成聚合筛选：`batch_zscore = 145.34 +/- 1.91`，`relative_gain = 137.40 +/- 1.81`，`raw = 127.43 +/- 18.27`。尽管 batch-zscore 在该开发集更高，Walker Full 按算法设计继续保留 relative-gain。旧 20 x 2 协议使用 seed `3..42`；新版使用不重叠的 seed `100..129`，正式结论来自全部 30 个 seed，不能按 Full 是否领先筛除运行。

五环境新版统一采用 actor-mean EA、client-local `log_std`/temperature、local candidate rollback、risk-adjusted archive 验收和低噪声 injection。正式结论必须来自完整 30-seed aggregate；smoke test 和预注册 5-seed 敏感度只验证趋势与稳健性边界，不替代主统计。

Reacher 已从主环境中移出。它的短 horizon 和 dense distance reward 更适合作调试 SAC 稳定性，不适合作为 EA+FedSAC 的核心证据：FedEvoSAC 的 population search 优势容易被短任务的快速局部优化掩盖，且 evaluation variance 会显著影响结论。当前改用 `Walker2d-v5`，它同样是 MuJoCo 连续控制，但 horizon 更长、动作维度更高、步态探索更依赖 actor 多样性，更适合检验 EA + federated SAC。Hopper 的 `1000+` 回报在 MuJoCo Hopper 中并非异常上界，但仍偏中等，因此 Hopper 保留为可继续提分的 locomotion 任务。

已暂停的正式重复实验使用：

```bash
EXPERIMENT_ID=fedevosac_formal_20x2_walker_hopper_20260805 \
  ./scripts/run_fedrl_20x2_experiment.sh
```

`20x2` 表示 20 个 outer repeat，每个 repeat 恰好两个 seed。不同 repeat 使用不同 seed pair：`(0,1), (2,3), ..., (38,39)`，所以每个 repeat 的图是 `n=2`，目标 aggregate 是 40 个独立 seed；不会把完全相同的 seed 重跑 20 次后伪装成更大的独立样本量。该批次已于 2026-08-06 主动停止，旧 Walker profile 的完成结果不能用 `SKIP_EXISTING` 混入新 profile。

保留 Walker relative-gain 后，新版 held-out 正式实验使用：

```bash
EXPERIMENT_ID=fedevosac_formal_20x2_relative_20260809 \
SEED_BASE=3 \
PARALLEL_REPEATS=2 \
FED_WALKER2D_SCORE_NORMALIZATION=relative_gain \
FED_HOPPER_SCORE_NORMALIZATION=relative_gain \
PLOT_ROOT=plots/formal_candidates/fedevosac_formal_20x2_relative_20260809 \
  ./scripts/run_fedrl_20x2_experiment.sh
```

20 个 repeat 对应 seed pair `(3,4), (5,6), ..., (41,42)`，与开发 seed `0/1/2` 完全分离。正式目录保存全部结果，不执行 Full-win 过滤；任何 post-hoc 示例图只能放入 diagnostics 并显式标注，不能替代 aggregate。

该批次已于 2026-08-09 从 Git commit `3b6b50b` 后台启动。原始日志写入 `logs/experiments/fedevosac_formal_20x2_relative_20260809/`，完整候选图写入 `plots/formal_candidates/fedevosac_formal_20x2_relative_20260809/`。为避免根分区 95% 使用率触发 Ray 的默认 spilling 拒绝阈值，启动进程显式设置 `RAY_local_fs_capacity_threshold=0.99`；该设置只改变临时对象落盘阈值，不改变算法、环境或统计口径。

正式启动使用 `PARALLEL_REPEATS=2`，降低多个 Ray 集群同时运行时的内存和对象存储压力；每个 repeat 的 stdout 独立写入 `logs/experiments/<experiment_id>/repeat_XX.log`，批内任务全部结束后才重画 aggregate。可通过 `START_REPEAT` / `END_REPEAT` 做分片或断点续跑。

每个 repeat 的横向比较运行 `FedEvoSAC-full` 和四个联邦 SAC baseline；独立消融图复用 full，并额外运行 `no_local_rl`、`no_ea_injection` 和 `no_heterogeneity`。新版 `no_local_rl` 同时关闭 local rollout、gradient update、candidate validation 和 migration，成为只保留 EA search/archive 的 pure-EA 消融，不再把 server best 加噪后伪装成 RL injection。`no_heterogeneity` 是环境消融。`raw_softmax` 与 `uniform_aggregation` 已移至独立 aggregation screening，不计入模块消融。

### Ant-v5 与 Pusher-v5 候选替换实验

`Ant-v5` 已完成两个 seed 的隔离 pilot，但不替换默认主环境。运行命令为：

```bash
./scripts/run_ant_two_seed_pilot.sh
```

Ant pilot 使用 seed `0/1`、3 clients、population `12`、horizon `1000`、`0.15 / env_params_only`、每 4 代联邦聚合和每 run `600,000` counted interactions。最终 `FedEvoSAC-full` 为 `483.47 +/- 7.38`，而四个联邦 SAC baseline 均约为 `995-1003`。Ant 默认每个 healthy timestep 提供固定生存奖励，baseline 可在缺乏明显前进时接近 `1000`；EA actor 扰动又容易破坏站立稳定性。因此该结果不支持继续增强 Ant 异质性，也不把 Ant 纳入正式主图。完整结果保存在：

```text
logs/experiments/fedevosac_ant_pilot_2seed_20260721/
plots_new/fedevosac_ant_pilot_2seed_20260721/aggregate/
```

`Pusher-v5` 随后作为新的隔离候选环境接入。它使用 23 维 observation、7 维连续动作和 100-step horizon；奖励直接由物体到目标距离、机械臂到物体距离和控制代价组成，没有 Ant 式的固定生存回报。当前先跑无异质 pilot，避免同时混入环境替换与 client heterogeneity 两个变量：

```bash
./scripts/run_pusher_two_seed_pilot.sh
```

Pusher pilot 使用 seed `0/1`、3 clients、population `12`、`0.0 / none` 和每 run `300,000` counted interactions，横向方法与 Ant pilot 相同。框架已经支持后续对 Pusher 的 gravity、body mass、joint damping 和 geom friction 做 `env_params_only` 扰动，但只有无异质结果先出现稳定学习趋势后才启用。日志与图片分别写入：

```text
logs/experiments/fedevosac_pusher_pilot_2seed_20260721/
plots_new/fedevosac_pusher_pilot_2seed_20260721/aggregate/
```

Ant 与 Pusher pilot 都只有 `n=2`，只用于环境筛选和配置诊断，不能作为显著性结论。

2026-09-03 之后的新结果只写入 `plots_2`，目录结构为：

```text
plots_2/<experiment_id>/aggregate/main/          # reward vs communication rounds
plots_2/<experiment_id>/aggregate/supplement/    # reward vs counted interactions
plots_2/<experiment_id>/aggregate/ablation/      # 独立模块消融图
plots_2/<experiment_id>/aggregate/paper_figures/ # 五环境 panel PNG/PDF
plots_2/<experiment_id>/aggregate/tables/        # 95% CI、收敛与 Wilcoxon CSV
plots_2/<sensitivity_id>/heterogeneity/           # 异质性强度敏感度图与 CSV
```

旧 `fedevosac_perenv_tuned_s0_comparison` 会原样复制到 `reference_single_seed`，但明确标记为 single-seed reference，不参加新均值、置信区间或显著性统计。

日志与图表目录约定：

```text
logs/                 # 所有 metrics、metadata、shell run log
logs/run/             # 后台/脚本 stdout 日志
plots/                # 所有对比图、消融图、表格
plots_new/            # 新 20x2 实验，和历史 plots 完全分开
plots_2/              # 2026-09-03 五环境、30-seed、95% CI 新协议
plots/training/       # src.main 生成的单次训练过程图
plots/tables/         # 零散 summary / significance CSV
```

默认连续对照组为 `FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median` 和 `FedEvoSAC`；这些 baseline 基于本项目已有联邦 RL 复现框架迁移到 continuous SAC，共享同一环境异质性、评估、日志和 actor 聚合协议。

`run_fedrl_three_scenarios.sh`、`run_fedrl_heterogeneous_suite.sh` 和 `run_fedevofsac_for_baselines.sh` 是 legacy discrete 脚本，只用于复核旧结果。当前连续实验不要调用这些脚本。

默认脚本使用 `BUDGET_PRESET=reduced`，会缩小 population / generation / evaluation 数量来控制 equal-step 预算，适合日常对比和调参。若要跑最终 full budget，可显式设置：

```bash
BUDGET_PRESET=full ./run_continuous_fedevosac_suite.sh
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
| `eval_episode_length_mean` | 当前可部署 actor 在异质 client suite 上的平均 episode 长度 |
| `eval_forward_return_mean` | 当前可部署 actor 的累计前进奖励；用于识别只存活不行走的局部最优 |
| `eval_survive_return_mean` | 当前可部署 actor 的累计 healthy reward |
| `eval_ctrl_return_mean` | 当前可部署 actor 的累计控制代价 |
| `eval_x_displacement_mean` | 当前可部署 actor 的平均水平位移 |
| `eval_x_velocity_mean` | 当前可部署 actor 的平均水平速度 |
| `communication_round` | 实际 server-client 协调轮数；FedEvoSAC 每代 population 分发/fitness 回传计 1，baseline 仅 actor 聚合时计 1 |
| `comm_upload_bytes` | 实际上传 actor 参数量估计 |
| `comm_full_traj_bytes` | 假设上传完整 trajectory 的通信量估计 |

主曲线默认使用 `eval_reward_mean`，表示 server 当前真正部署的 checkpoint。FedEvoSAC 的 checkpoint 来自通过固定 validation 的 global elite archive；baseline 则使用带 rollback 验收的 global actor。因此主曲线允许在候选退化时保持上一策略，属于 deployable-policy 曲线，不是未经筛选的瞬时 candidate 曲线，也不能描述成纯粹的 raw-learning trajectory。baseline 和 FedEvoSAC 的本轮聚合候选记录在 `candidate_eval_mean`，应作为辅助曲线检查聚合是否退化；FedEvoSAC 的本地 SAC rollout 分数保存在 `client_reward_mean` / `client_reward_std`。`best_fitness` / `archive_best` 只用于解释 EA 搜索和 archive，不再混入主曲线取最大值。

新版只保留两种横坐标，它们回答不同问题，不能互相替代：

| 横坐标 | 回答的问题 | 论文位置 | 限制 |
|--------|------------|----------|------|
| communication round | 达到某个 return 需要多少次真实 server-client 协调，即通信效率 | 主图 | FedEvoSAC 与 baseline 每轮的 payload 和环境交互量不同，不能单独证明样本或字节效率 |
| counted environment interactions | 在相同真实采样预算下谁学得更快，即样本效率 | 补充主证据 | 必须统计 population、local rollout、archive/candidate validation 的全部交互 |

这种主图/补充图分层与 FRL 文献的常见结构一致。[Federated Reinforcement Learning with Environment Heterogeneity](https://proceedings.mlr.press/v151/jin22a.html) 在复杂任务中画 averaged return vs episodes/frames 并显示跨运行不确定性，同时单独研究 local-update interval 对通信频率的影响；[Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling](https://proceedings.mlr.press/v162/khodadadian22a.html) 则把 environment iterations/sample complexity 与 communication cost 分开分析。本项目因此输出五环境 panel：round 是 main figure，steps 是 supplementary evidence。normalized progress 会抹去绝对预算差异，新协议不再生成。

新图报告 deployable current policy 的跨 seed 均值和双侧 95% Student-t CI。每个横坐标位置使用当时可用 seed 的 sample SD（`ddof=1`）、标准误和对应自由度的 t 临界值；实线是均值，阴影是 95% CI。主指标不通过 `max(eval_reward_mean, eval_ea_mean, best_fitness, archive_best)` 拼接。正式 aggregate 使用全部 30 个预注册 seed，平滑只作用于显示曲线，不改 summary CSV。每个 run 额外生成 `convergence_report.csv`：最后 6 个评估点的增益和范围必须落在绝对/相对容差内。尾部稳定只表示曲线不再明显变化，不等价于成功解决任务；低分停滞也可能被判为稳定。

最终表格的展示列使用 `mean +/- 95% CI half-width`，并保留 sample SD、CI 下界/上界、`forward return`、`x velocity`、`max_steps`、`max_round`、`wall_time_sec` 和 convergence 状态作为审计字段。方法差异采用相同 seed 配对的 Wilcoxon signed-rank test；主报告使用双侧检验，环境内对多个 baseline 的 p 值做 Holm 校正，同时报告胜/平/负次数、paired bootstrap 95% CI 和 rank-biserial effect size。该检验遵循 Wilcoxon 的 paired ranking 思路：F. Wilcoxon, “Individual Comparisons by Ranking Methods,” *Biometrics Bulletin*, vol. 1, no. 6, pp. 80-83, 1945, doi:10.2307/3001968。5 个 paired seed 的双侧 exact Wilcoxon 最小 p 值为 0.0625，不能独立支撑 0.05 显著性声明，因此正式检验使用 n=30。

## 12. 历史完整实验结果与新版正式实验

历史实验 `fedevosac_20x2_converged_20260714` 已完成：20 个 outer repeat，每个 repeat 使用两个不重复 seed，共 40 个独立 seed。三环境、9 个横向/消融方法共生成 `1080/1080` 个 `metrics.csv`，每个环境和方法均为 `n=40`，运行日志未发现 traceback、OOM 或中途终止。以下数值属于旧协议，只用于动机、回归检查和历史对照。

主表使用最终 `eval_reward_mean`，下列 `+/-` 是跨 seed sample standard deviation，不是图中的 90% CI：

| 环境 | `FedEvoSAC-full` | 最强非 EA baseline | 绝对提升 | paired bootstrap 95% CI | paired Wilcoxon p | full 尾部稳定 |
|------|------------------|--------------------|----------|-------------------------|-------------------|-----------------|
| `Swimmer-v5` | `133.57 +/- 64.23` | `FedBest-SAC: 45.69 +/- 2.16` | `+87.88` | `[68.54, 107.80]` | `< 0.0001` | `26/40` |
| `Walker2d-v5` | `998.35 +/- 81.47` | `FedBest-SAC: 741.53 +/- 316.55` | `+256.82` | `[158.12, 356.71]` | `0.0049` | `38/40` |
| `Hopper-v5` | `1019.64 +/- 48.35` | `FedBest-SAC: 743.27 +/- 495.34` | `+276.36` | `[123.31, 423.67]` | `0.0029` | `40/40` |

这支持的结论是：在当前统一 continuous-SAC 协议和约 1.2M counted interactions 下，EA actor population + archive 的 FedEvoSAC family 明显优于四个非 EA 联邦 SAC baseline。不能把它扩展成“所有强异质 FRL 环境均被解决”，因为 Swimmer/Walker2d 本轮没有额外 client heterogeneity，且 Swimmer 均值曲线到预算终点仍在上升。

稳定性需要按环境解释：

- Swimmer 方差仍大，17/40 个 seed 最终低于 100，只有 65% 的 full runs 尾部稳定，因此当前预算不足以声明完整收敛；
- Walker2d 有 4/40 个 seed 低于 900，但 95% 的 full runs 尾部稳定；
- Hopper 的中位数为 `1024.72`，39/40 个 seed 集中在约 1015 以上；seed 33 的 `726.44` 单次失败显著抬高了总体标准差。

当前消融并未证明 full 中每个模块都必要：

| 环境 | full | 最高均值消融 | full - ablation | p |
|------|------|--------------|-----------------|---|
| `Swimmer-v5` | `133.57 +/- 64.23` | `raw_softmax: 141.18 +/- 60.16` | `-7.61` | `0.491` |
| `Walker2d-v5` | `998.35 +/- 81.47` | `no_ea_injection: 1021.42 +/- 53.83` | `-23.07` | `0.182` |
| `Hopper-v5` | `1019.64 +/- 48.35` | `raw_softmax: 1025.99 +/- 11.25` | `-6.35` | `0.374` |

full 与四个消融的逐 seed 差异均未达到显著性。当前可以声称“包含 EA search/archive 的完整 family 优于非 EA baselines”，但不能声称 relative-gain、local SAC refinement 或 EA injection 已分别得到显著验证。下一阶段应优先重构消融问题，而不是继续只挑选更高的单 seed 曲线。

旧消融协议复核发现三个实现层混淆：full 的 aggregation entropy 为 `1.088`，接近三客户端最大值 `ln(3)=1.099`；聚合器二次 z-score 抵消了 score scale 并污染 raw-softmax 定义；`no_local_rl` 的 40 个 run 仍产生 420 copies 的带噪 injection。上述历史结果保留作为失败协议，不与修复后的 `20260805` 消融合并统计。

历史结果位置：

```text
plots_new/fedevosac_20x2_converged_20260714/aggregate/paper_figures/
plots_new/fedevosac_20x2_converged_20260714/aggregate/tables/
logs/experiments/fedevosac_20x2_converged_20260714/
```

`plots_new/selected_best_converged_20260721/` 保存了 Walker2d/Hopper 的历史跨批次展示候选。该目录有独立 `README.md` 和 `selection_manifest.csv`，属于 post-hoc supporting/visual collection，不能替代完整 aggregate。新版 `plots_2` 不接受 Full-win 或高分 seed 筛选。

历史正式实验 `fedevosac_formal_20x2_walker_hopper_20260805` 随后因 Walker task 定义问题暂停。2026-09-03 新协议改用五环境、30 个 held-out seed、相同 counted-interaction 预算、四个非 EA 联邦 SAC baseline，以及与横向图分离的三个消融；异质性敏感度另行运行。只有完整 aggregate、95% CI、收敛检查和配对显著性报告齐全后才可作为论文证据。

## 13. 当前实现状态

已完成：

- 连续正式环境主线：`Walker2d-Locomotion`、`Hopper-Locomotion`、`Ant-v5`、`HalfCheetah-v5`、`Swimmer-v5`；
- `SACPolicy`：tanh Gaussian actor、twin critics、target critics、learnable alpha；
- continuous SAC federated baselines：`FedAvg-SAC`、`FedBest-SAC`、`FedSoftmax-SAC-noEA`、`RobustFed-SAC-Median`；
- EA genotype actor-only；
- GA actor 前缀可配置且当前固定为 `actor.`；
- normalized relative-gain reward-aware federated actor aggregation；
- delta clipping 与 bounded EA mutation；
- global elite archive；
- FedEvoSAC 连续对比和消融脚本；
- 30 个预注册 held-out seed、断点续跑、独立日志和 aggregate renderer；
- current deployable、candidate、steps、round、95% Student-t CI 和 convergence report 分层输出；normalized progress 已从新协议移除；
- `comparison_wilcoxon.csv` / `ablation_wilcoxon.csv` 输出 paired Wilcoxon、Holm 校正、bootstrap 95% CI 和 rank-biserial effect size；
- 五环境异质性强度敏感度脚本与独立结果目录；
- 历史 40-seed 三环境实验已归档；新版正式结果统一写入 `plots_2`；
- legacy 离散 `FedEvoFSAC`、`FSACPolicy` 和 DQN 代码仍可复核，但不属于当前实验主线。

主要风险和下一步：

- Swimmer 按新协议重新进入正式实验，但历史 seed sensitivity 和不稳定收敛风险仍成立；不得用旧单 seed高分或事后挑选 5 个 seed 替代 30-seed 结论；
- Walker2d 的旧统一缩放异质性诱发约 1000 分的存活奖励局部最优；`Walker2d-Locomotion` 使用 gait-structured `0.30` profile、显式 `healthy_reward=0.05` 和 locomotion diagnostics，3-seed pilot 已通过，小规模 baseline/消融复核也已完成；
- Hopper 默认 reward 的正式结果主要来自 survival shortcut；新版消融必须使用 `Hopper-Locomotion`，并以 forward return 和 x velocity 验证真实运动，不能只按 total return 排名；
- Walker 聚合筛选显示 batch-zscore 在三个开发 seed 更高，但正式协议按设计保留 relative-gain；必须完整报告 30 个 held-out seeds，不能按 Full 排名挑选运行；
- deployable 主曲线带 archive / rollback，适合衡量最终可部署性能，但可能隐藏聚合 candidate 的瞬时退化；论文必须同时报告 candidate 或明确 checkpoint 规则；
- actor-only 共享避免 critic scale mismatch，但 client critic 完全本地化，早期本地更新仍可能噪声较大；
- raw reward softmax 容易受异质 client reward scale 影响；`aggregation_entropy`、`aggregation_score_std` 和 reward-scale stress test 仍需单独分析；
- MuJoCo 异质性过强时会改变最优动作尺度，EA mutation、action noise 和 SAC temperature 需要联动调参；
- 当前 privacy 仅表示 trajectory 不离开 client；actor 参数和标量统计仍会泄露信息，不具备 differential privacy、secure aggregation 或攻击抵抗保证。
