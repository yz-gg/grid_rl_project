# grid_rl_project

这是一个纯 Python 的单智能体强化学习示例工程，用于验证“强化学习 agent 调节电网电压参考值 `Delta Vref`”的基本训练、评估和结果保存流程。

当前工程包含两套算法：

- SAC: Soft Actor-Critic，off-policy，使用 replay buffer。
- PPO: Proximal Policy Optimization，on-policy，使用 rollout buffer 和 GAE。

当前环境为 `DummyGridEnv`，不依赖 MATLAB、Simulink 或 Grid2Op。它不是精确电网模型，主要用于算法代码调试和训练流程验证。

## 项目结构

```text
grid_rl_project/
├── agents/
│   ├── sac_agent.py              # SAC agent
│   └── ppo_agent.py              # PPO agent
├── buffers/
│   ├── replay_buffer.py          # SAC 使用的经验回放池
│   └── rollout_buffer.py         # PPO 使用的 on-policy rollout buffer
├── configs/
│   ├── sac_dummy.yaml            # SAC 默认配置
│   └── ppo_dummy.yaml            # PPO 默认配置
├── envs/
│   └── dummy_grid_env.py         # 简化电网环境
├── networks/
│   ├── actor.py                  # SAC tanh-squashed Gaussian policy
│   └── critic.py                 # SAC Q 网络
├── trainers/
│   ├── train_sac_dummy.py        # SAC 训练入口
│   ├── evaluate_sac_dummy.py     # SAC 评估入口
│   ├── train_ppo_dummy.py        # PPO 训练入口
│   └── evaluate_ppo_dummy.py     # PPO 评估入口
├── utils/
│   ├── logger.py                 # 配置、随机种子、设备、CSV 工具
│   └── plot.py                   # reward 和评估曲线绘图
├── main.py                       # 统一命令行入口
└── requirements.txt
```

## 环境说明

状态为 5 维：

```text
obs = [df, V, P, Q, I]
```

动作为 1 维连续动作：

```text
action = Delta Vref
```

默认动作范围：

```text
[-0.02, 0.02]
```

环境会在默认第 50 个 step 后加入负载阶跃扰动。扰动会放大频率偏差，合理的 `Delta Vref` 可以部分改善频率偏差，但过大的动作会带来电压偏差和动作惩罚。

后续如果接入真实电网仿真，可以将 `DummyGridEnv` 替换为 `MatlabGridEnv`、`Grid2OpEnv` 或其他环境，只要保持 `reset()` 和 `step(action)` 接口一致即可。

## 安装依赖

建议在项目根目录下安装：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `tqdm`

## 使用统一入口

### 训练 SAC

```bash
python main.py train --algo sac
```

等价于使用默认配置：

```bash
python main.py train --algo sac --config configs/sac_dummy.yaml
```

### 评估 SAC

```bash
python main.py evaluate --algo sac
```

指定模型路径：

```bash
python main.py evaluate --algo sac --model checkpoints/sac_dummy_final.pt
```

### 训练 PPO

```bash
python main.py train --algo ppo
```

等价于使用默认配置：

```bash
python main.py train --algo ppo --config configs/ppo_dummy.yaml
```

### 评估 PPO

```bash
python main.py evaluate --algo ppo
```

指定模型路径：

```bash
python main.py evaluate --algo ppo --model checkpoints/ppo_dummy_final.pt
```

如果不传 `--algo`，默认运行 SAC。

## 使用独立入口

也可以直接运行各算法自己的训练和评估脚本。

### SAC

```bash
python trainers/train_sac_dummy.py --config configs/sac_dummy.yaml
python trainers/evaluate_sac_dummy.py --config configs/sac_dummy.yaml
```

### PPO

```bash
python trainers/train_ppo_dummy.py --config configs/ppo_dummy.yaml
python trainers/evaluate_ppo_dummy.py --config configs/ppo_dummy.yaml
```

## 输出文件

默认训练和评估结果会保存到 `checkpoints/` 和 `results/`。

SAC 默认输出：

```text
checkpoints/sac_dummy_final.pt
results/sac_dummy_rewards.csv
results/sac_dummy_losses.csv
results/sac_dummy_rewards.png
results/sac_dummy_evaluation.csv
results/sac_dummy_evaluation.png
```

PPO 默认输出：

```text
checkpoints/ppo_dummy_final.pt
results/ppo_dummy_rewards.csv
results/ppo_dummy_losses.csv
results/ppo_dummy_rewards.png
results/ppo_dummy_evaluation.csv
results/ppo_dummy_evaluation.png
```

评估脚本会打印：

- `max abs(df)`
- `max abs(V-1.0)`
- `action smoothness`
- `episode reward`

## 算法说明

### SAC

SAC 是 off-policy 最大熵强化学习算法。当前实现包含：

- tanh-squashed Gaussian actor
- 双 Q critic
- target critic
- replay buffer
- 自动熵系数 `alpha`
- critic soft update

SAC 的数据可以从 replay buffer 中反复采样，样本利用率较高，适合连续动作控制任务。

### PPO

PPO 是 on-policy 策略梯度算法。当前实现包含：

- 连续动作 actor-critic 网络
- tanh-squashed Gaussian policy
- rollout buffer
- GAE advantage 估计
- clipped policy objective
- value loss
- entropy bonus

PPO 每次使用当前策略采样 rollout，然后进行多轮 minibatch 更新。旧策略数据更新完后不再复用。

## 配置文件

SAC 配置位于：

```text
configs/sac_dummy.yaml
```

PPO 配置位于：

```text
configs/ppo_dummy.yaml
```

常用配置项包括：

- `seed`: 随机种子
- `device`: `auto`、`cpu` 或 `cuda`
- `env`: 环境参数
- `agent`: 算法超参数
- `training`: 训练轮数、batch size、保存间隔和输出路径
- `evaluation`: 默认评估模型和输出路径

## 注意事项

- SAC 使用 `ReplayBuffer`，PPO 使用 `RolloutBuffer`，两者的数据机制不同。
- 当前环境是简化模型，只能用于算法流程验证，不能直接代表真实电网动态。
- 如果运行时报 `ModuleNotFoundError: No module named 'torch'`，说明当前 Python 环境尚未安装 PyTorch，需要先安装依赖。
- 若要公平比较 SAC 和 PPO，建议固定同一环境配置、随机种子和评估流程，并对多组 seed 取平均结果。
