# Dist-ERL: Distributed Evolutionary Reinforcement Learning

## 1. 项目定位

Dist-ERL 是一个面向连续控制任务的 **分布式进化强化学习框架**。本项目现在以 `dist_erl` 作为论文主方法：用进化算法 EA 维持一组策略种群进行全局搜索，用强化学习 RL actor-critic 进行梯度优化，并通过 Ray 将 EA 个体评估分发到多个 worker 上。

核心目标不是把 ERL-Re2 继续分布式化，而是研究：

- ERL 中最耗时的种群评估能否通过分布式 worker 提升吞吐；
- 分布式 EA 与 RL 学习如何在同一训练循环中互补；
- 只回传 `seed + fitness` 的轻量通信能否支撑大规模多环境、多种子实验；
- 种群多样性、RL actor 表现、EA 精英表现之间如何影响最终性能。

`dist_erl_re2` 已废弃，不再作为新实验模式、论文方法或默认脚本目标。`erl_re2` 仅保留为单 worker baseline，用于和已有 ERL-Re2 思路做对照。

---

## 2. 算法思想

### 2.1 为什么结合 EA 和 RL

纯 RL 通常依赖梯度更新，样本效率高，但在稀疏奖励、高维连续控制或早期探索不足时容易陷入局部最优。纯 EA 不依赖梯度，能直接在策略参数空间中搜索，探索能力更强，但每一代需要评估大量个体，环境交互成本高。

Dist-ERL 采用 ERL 的混合思路：

- **EA 种群**：维护多个 actor 参数个体，通过精英保留、锦标赛选择、交叉和变异进行全局搜索。
- **RL learner**：维护一个 actor-critic 策略，通过 replay buffer 和梯度下降进行局部改进。
- **分布式 worker**：并行评估 EA 种群，降低每代 wall-clock 时间。
- **统一日志与指标**：同时记录 EA、RL、通信和多样性指标，方便论文中解释性能来源。

### 2.2 Dist-ERL 的主方法

`dist_erl` 的每一代训练包含两条并行思想但顺序执行的主线：

1. **EA 主线**
   - `EAManager` 保存 population。
   - 将每个 individual 分配给 Ray `RolloutWorker`。
   - worker 加载 individual 的 actor 权重，在环境中 rollout，返回 fitness。
   - manager 根据 fitness 排序、保留精英、产生下一代。

2. **RL 主线**
   - `RLLearner` 在环境中采集若干条 RL trajectory。
   - trajectory 被加入 `HybridReplayBuffer` 的 RL 区域。
   - learner 根据 replay buffer 执行若干次 actor-critic update。

这两条主线共享同一个任务和策略结构，但 `dist_erl` 不使用 Re2 的经验重现同步机制。这样论文主旨更清晰：重点考察 **分布式 ERL 框架本身**，而不是把 Re2 机制叠加到分布式系统上。

### 2.3 单代训练流程

伪代码如下：

```text
Initialize EA population P = {theta_1, ..., theta_N}
Initialize RL actor-critic policy pi_phi and replay buffer B
Start K Ray rollout workers

for generation = 1 ... G:
    # Distributed EA evaluation
    for each individual theta_i in P:
        assign theta_i to worker i mod K
        worker evaluates theta_i with its seed
        return fitness_i

    # EA evolution
    sort P by fitness
    preserve elites
    select tournament winners
    replace discarded individuals by crossover/mutation

    # RL learning
    collect rl_rollouts with pi_phi
    add transitions to replay buffer B
    for step = 1 ... rl_updates:
        sample mini-batch from B
        update actor-critic parameters phi

    # Evaluation and logging
    evaluate RL actor
    evaluate EA best individual
    log reward, fitness, diversity, time, communication
```

代码入口对应关系：

- 主循环：`src/main.py`
- EA 管理：`src/manager.py`
- RL 学习：`src/learner.py`
- Ray worker：`src/worker.py`
- RL step helper：`src/training.py`
- 遗传算子：`src/utils/erl_re2_ga.py`

---

## 3. 分布式设计

### 3.1 Worker 角色

`RolloutWorker` 是 Ray remote actor。每个 worker 持有自己的环境实例，接收一个 individual：

```text
individual = {
    id,
    actor weights,
    seed
}
```

worker 做的事情很轻：

1. 加载 actor 权重；
2. 用 individual 的 seed reset 环境；
3. 执行确定性 actor rollout；
4. 返回累计 reward 作为 fitness。

### 3.2 通信方式

Dist-ERL 的论文叙事中，通信成本是一个重要卖点。当前日志记录两类通信量：

| 指标 | 含义 |
|------|------|
| `comm_upload_bytes` | 近似表示每代只上传 `seed + fitness` 的通信量 |
| `comm_full_traj_bytes` | 假想上传完整 trajectory 的通信量 |

这里的核心对比是：分布式评估不需要把每个 worker 的完整轨迹都传回 learner；主方法只需要每个 individual 的 fitness 来驱动 EA 进化。因此通信开销可以近似为：

```text
population_size x (seed bytes + fitness bytes)
```

而完整轨迹上传开销近似为：

```text
population_size x episode_steps x (state_dim + action_dim + reward/done fields)
```

这使得 Dist-ERL 更适合多 worker、多种子、多环境的实验设置。

### 3.3 Worker 数量与 CPU 保护

`src/main.py` 会根据机器 CPU 数自动限制 worker 数量，避免 Ray 调度时 learner、manager 和 workers 抢占过多资源导致卡顿：

```text
num_workers = min(requested_workers, available_cpu_budget)
```

如果日志中出现：

```text
Capped num_workers 8 -> 5
```

表示代码主动降低了 worker 数，这是正常保护机制。

---

## 4. EA 部分

### 4.1 个体表示

EA population 中每个个体是一个 `Individual`，主要包含：

| 字段 | 含义 |
|------|------|
| `id` | 个体编号 |
| `weights` | actor 网络参数 |
| `seed` | rollout 使用的环境随机种子 |
| `fitness` | 最近一次评估得到的累计回报 |
| `hyperparams` | 预留超参数字段 |

### 4.2 选择与进化

EA 使用与 ERL-Re2 风格一致的遗传算子，但在 Dist-ERL 中它服务于主方法的分布式 ERL 搜索。

每代步骤：

1. **排序**：按 fitness 从高到低排序。
2. **精英保留**：保留 `--num-elitists` 个最优个体。
3. **锦标赛选择**：从种群中选出 winners。
4. **淘汰者替换**：用 elite 和 winner 的组合替换低适应度个体。
5. **b-Crossover**：对 actor 权重矩阵按 action row 进行交叉。
6. **b-Mutation**：对非精英 actor 权重做小幅、大幅或 reset 变异。

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--population-size` | `50` | EA 种群大小 |
| `--num-elitists` | `1` | 精英数量 |
| `--elite-fraction` | `0.2` | 未设置 `--num-elitists` 时使用 |
| `--ea-mutation-prob` | `0.9` | 非精英变异概率 |
| `--ea-mutation-beta-frac` | `0.7` | 每个 action row 中参与变异的列比例 |
| `--ea-prob-reset-and-super` | `0.05` | 大幅变异和 reset 的概率基准 |

### 4.3 多样性维护

代码记录 `weight_diversity`，计算 actor 权重向量之间的余弦多样性：

```text
weight_diversity = 1 - mean(pairwise cosine similarity)
```

当 RL 或 EA 评估长期停滞时，`EAManager.boost_diversity()` 会替换一部分尾部个体，并对非精英进行额外变异。相关参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--stagnation-patience` | `12` | 停滞多少代后触发多样性增强 |
| `--stagnation-min-delta` | `5.0` | 判断是否有足够改进的最小 reward 增量 |
| `--immigrant-fraction` | `0.15` | 被随机移民替换的尾部个体比例 |

---

## 5. RL 部分

### 5.1 Learner

`RLLearner` 支持：

```text
DDPG / TD3 / PPO
```

默认是 DDPG。Learner 维护：

- actor-critic policy；
- target networks；
- optimizer；
- replay buffer；
- exploration noise；
- training step 计数。

### 5.2 Replay Buffer

`HybridReplayBuffer` 保留了 RL 数据和 EA/Re2 baseline 数据的接口。对主方法 `dist_erl` 来说，主要使用 RL trajectory：

```text
collect_rl_trajectories -> add_rl_experience -> update_step
```

`erl_re2` baseline 会额外使用 EA elite 经验重现，因此 README 中保留 `--ea-batch-ratio` 等参数说明，但它们不是 Dist-ERL 主方法的创新点。

### 5.3 RL 更新

每代 RL 做：

1. 用当前 policy 采集 `--rl-rollouts` 条轨迹；
2. 将 transition 加入 replay buffer；
3. 采样 batch；
4. 根据算法执行 actor-critic update；
5. 梯度裁剪，防止不稳定更新。

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--algorithm` | `DDPG` | RL 算法 |
| `--rl-rollouts` | `2` | 每代 RL rollout 数 |
| `--rl-updates` | `10` | 每代梯度更新步数 |
| `--buffer-size` | `1000000` | replay buffer 容量 |
| `--batch-size` | `256` | mini-batch 大小 |
| `--lr` | `3e-4` | 学习率 |
| `--policy-exploration-noise` | `0.1` | DDPG/TD3 动作探索噪声 |

---

## 6. 训练模式

| 模式 | EA | RL | Re2 | 分布式评估 | 论文定位 |
|------|:--:|:--:|:---:|:----------:|----------|
| `pure_rl` | - | yes | - | - | RL baseline |
| `pure_ea` | yes | - | - | optional | EA baseline |
| `standard_erl` | yes | yes | - | single worker | ERL baseline |
| `erl_re2` | yes | yes | yes | single worker | ERL-Re2 baseline |
| `dist_erl` | yes | yes | - | yes | **主方法** |

### 6.1 各模式含义

`pure_rl`：
只运行 RL learner，不使用 EA population。用于观察 DDPG/TD3/PPO 单独训练的表现。

`pure_ea`：
只运行 EA population，不进行 RL 更新。用于观察进化搜索本身的能力。

`standard_erl`：
EA + RL，但通常使用单 worker。用于对比“有 ERL 混合但没有分布式评估”的情况。

`erl_re2`：
单 worker ERL-Re2 baseline。用于和经验重现机制做对照。

`dist_erl`：
EA + RL + 多 worker 分布式评估，是本文主方法。

---

## 7. 实验设计

### 7.1 主实验

默认主实验比较五个模式：

```text
pure_rl pure_ea standard_erl erl_re2 dist_erl
```

运行：

```bash
./run_seeds.sh
```

该脚本默认运行：

```text
6 MuJoCo-v2 tasks x 5 modes x 10 seeds
```

环境列表来自 `src/config.py`：

| 任务 | `ENV_NAME` |
|------|------------|
| HalfCheetah | `HalfCheetah-v2` |
| Swimmer | `Swimmer-v2` |
| Hopper | `Hopper-v2` |
| Ant | `Ant-v2` |
| Walker2d | `Walker2d-v2` |
| Humanoid | `Humanoid-v2` |

运行时 `src/utils/environment.py` 会把论文中的 `*-v2` 任务名映射到当前 Gymnasium 可用版本。

### 7.2 单环境快速基准

```bash
ENV_NAME=Ant-v2 ./run_benchmarks.sh
```

适合快速检查五种模式是否能跑通。

### 7.3 扩展性实验

```bash
./run_scaling.sh
python3 scripts/plot_scaling_bandwidth.py --log-dir logs
```

扩展性实验主要看：

- worker 数增加后 wall-clock time 是否下降；
- seed/fitness 上传量与完整轨迹上传量的差距；
- 种群多样性在训练过程中的变化。

### 7.4 出图

```bash
python3 generate_plots.py --log-dir logs --require-real
```

输出通常包括：

- sample efficiency 曲线；
- wall-clock 曲线；
- multi-env comparison；
- significance table；
- scaling / bandwidth / diversity 图。

禁止用无日志合成曲线作为论文结果。

---

## 8. 日志与指标解释

每次实验会在 `logs/<exp-name>/` 下生成：

```text
metadata.json
metrics.csv
```

关键指标：

| 指标 | 含义 |
|------|------|
| `generation` | 当前代数 |
| `total_env_steps` | 估算环境交互步数 |
| `eval_reward_mean` | RL actor 的确定性评估；`pure_ea` 下为最佳 EA 个体评估 |
| `eval_reward_std` | 多回合评估标准差 |
| `eval_ea_mean` | EA 最优个体多回合评估 |
| `eval_ea_std` | EA 最优个体多回合评估标准差 |
| `best_fitness` | 当前 EA 种群最高 fitness |
| `mean_fitness` | 当前 EA 种群平均 fitness |
| `fitness_std` | 当前 EA 种群 fitness 标准差 |
| `weight_diversity` | actor 权重多样性 |
| `rl_steps` | RL learner 已完成的梯度更新步数 |
| `buffer_size` | replay buffer 当前大小 |
| `gen_time` | 单代耗时 |
| `total_time` | 总训练耗时 |
| `comm_upload_bytes` | 近似 seed/fitness 上传量 |
| `comm_full_traj_bytes` | 假想完整轨迹上传量 |
| `stagnation_boost` | 是否触发多样性增强 |
| `policy_exploration_noise` | 当前 RL 探索噪声 |

论文中建议同时报告：

- final performance；
- best-so-far performance；
- sample efficiency；
- wall-clock efficiency；
- worker scaling；
- communication saving；
- diversity trend。

---

## 9. 安装

```bash
conda activate dist-erl-re2
pip install -r requirements.txt
pip install swig "gymnasium[box2d,mujoco]"
```

说明：当前服务器已有 conda 环境仍叫 `dist-erl-re2`。代码和项目名已切到 Dist-ERL；环境名可后续另行克隆/重命名，不影响运行。

---

## 10. 快速运行

默认入口即为主方法 `dist_erl`：

```bash
cd ~/code/Dist-ERL
python -m src.main --env Hopper-v2
```

或使用脚本：

```bash
./run_dist_erl.sh --env Hopper-v2 --mode dist_erl --max-generations 50
```

短 smoke test：

```bash
./run_dist_erl.sh \
  --env Pendulum-v1 \
  --mode dist_erl \
  --population-size 8 \
  --num-workers 2 \
  --max-generations 5 \
  --max-episode-steps 200
```

---

## 11. 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `dist_erl` | 训练模式，默认即主方法 |
| `--env` | `LunarLanderContinuous-v3` | 环境名 |
| `--algorithm` | `DDPG` | RL 算法（`DDPG` / `TD3` / `PPO`） |
| `--num-workers` | `4` | Ray rollout workers 数量 |
| `--population-size` | `50` | EA 种群大小 |
| `--max-generations` | `100` | 最大训练代数 |
| `--max-episode-steps` | `1000` | 单回合最大步数 |
| `--eval-episodes` | `10` | 评估回合数 |
| `--rl-rollouts` | `2` | 每代 RL rollout 数 |
| `--rl-updates` | `10` | 每代/同步点梯度更新步数 |
| `--elite-fraction` | `0.2` | 未显式设 `--num-elitists` 时的精英比例 |
| `--num-elitists` | `1` | EA 精英数量 |
| `--policy-exploration-noise` | `0.1` | DDPG/TD3 采样时动作噪声 |
| `--stagnation-patience` | `12` | 停滞后触发 EA 移民和探索增强 |
| `--immigrant-fraction` | `0.15` | 停滞时替换的尾部个体比例 |

Re2 相关参数仍保留给 `erl_re2` baseline，但不构成 Dist-ERL 主方法。

---

## 12. 项目结构

```text
Dist-ERL/
├── src/
│   ├── main.py                  # 训练主循环与日志
│   ├── config.py                # 模式、环境、绘图配置
│   ├── training.py              # RL/Re2 step helper
│   ├── manager.py               # EA population manager
│   ├── learner.py               # RL learner
│   ├── worker.py                # Ray rollout worker
│   └── utils/
│       ├── environment.py       # Gymnasium/MuJoCo 环境适配
│       ├── erl_re2_ga.py        # EA 遗传算子
│       ├── individual.py        # EA 个体结构
│       ├── policies.py          # DDPG/TD3/PPO policy
│       ├── policy_utils.py      # actor 权重加载与动作计算
│       └── replay_buffer.py     # replay buffer
├── run_dist_erl.sh              # 主启动脚本
├── run_seeds.sh                 # 多环境多种子主实验
├── run_scaling.sh               # worker 扩展性实验
├── run_benchmarks.sh            # 单环境快速 benchmark
├── run_ablations.sh             # 消融/诊断脚本
├── generate_plots.py            # 论文图生成
└── scripts/plot_scaling_bandwidth.py
```

---

## 13. 论文可写贡献点

可以围绕以下贡献组织论文：

1. **Distributed ERL framework**  
   将 ERL 中昂贵的 population evaluation 分发到多个 Ray workers，使 EA 搜索适合更大规模的 MuJoCo 多环境实验。

2. **Communication-efficient population evaluation**  
   分布式 worker 只需返回 seed/fitness，而不是完整 trajectory；日志中显式比较 `comm_upload_bytes` 与 `comm_full_traj_bytes`。

3. **Joint analysis of EA and RL learning dynamics**  
   同时记录 `eval_reward_mean`、`eval_ea_mean`、`best_fitness`、`weight_diversity`，分析 RL 梯度学习与 EA 全局搜索的互补性。

4. **Scalable empirical protocol**  
   提供多环境、多模式、多种子脚本，并区分 sample efficiency、wall-clock efficiency、通信成本和多样性趋势。

---

## 14. 与 ERL-Re2 的关系

本项目曾尝试过分布式 Re2 方向，但当前论文主线已完全切换为 Dist-ERL。

保留 `erl_re2` 的原因：

- 作为已有方法 baseline；
- 方便比较 Re2 经验重现与分布式 ERL 主方法的差异；
- 保留部分代码接口，避免破坏已有测试和对照实验。

不再使用 `dist_erl_re2` 的原因：

- 分布式 Re2 引入额外耦合，容易让论文主旨变得混乱；
- 短实验显示主方法优势更适合围绕 `dist_erl` 展开；
- 论文贡献应聚焦分布式 ERL、通信效率和可扩展实验协议。

---

## 15. 引用

- Khadka & Tumer (2018). Evolutionary Reinforcement Learning. NeurIPS.
- Wan et al. (2022). ERL-Re2: Efficient Evolutionary Reinforcement Learning with Reproducible and Reusable Experience.
