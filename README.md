# grid_rl_project

本项目是一个纯 Python 的单智能体 SAC 强化学习示例，用于验证“强化学习 agent 调节电网电压参考值 `Delta Vref`”的基本代码流程。

当前阶段只包含 `DummyGridEnv + SAC`，不依赖 MATLAB、Simulink 或 Grid2Op。

## 环境说明

请你自行创建并激活虚拟环境。本项目不包含 conda 环境创建命令，也不会自动安装依赖。

依赖列表在 `requirements.txt` 中。激活你自己的环境后，可以手动安装：

```bash
pip install -r requirements.txt
```

## 项目结构

- `envs/dummy_grid_env.py`: 简化电网环境，接口类似 Gym。
- `networks/actor.py`: tanh-squashed Gaussian policy。
- `networks/critic.py`: SAC Q 网络。
- `buffers/replay_buffer.py`: numpy 预分配经验回放池。
- `agents/sac_agent.py`: SAC agent、自动熵系数、软更新、模型保存/加载。
- `trainers/train_sac_dummy.py`: 训练入口。
- `trainers/evaluate_sac_dummy.py`: 评估入口。
- `configs/sac_dummy.yaml`: 默认训练和评估配置。

## 训练

在项目根目录运行：

```bash
python trainers/train_sac_dummy.py
```

训练完成后：

- 模型保存到 `checkpoints/sac_dummy_final.pt`
- episode reward CSV 保存到 `results/sac_dummy_rewards.csv`
- reward 曲线保存到 `results/sac_dummy_rewards.png`

也可以显式指定配置文件：

```bash
python trainers/train_sac_dummy.py --config configs/sac_dummy.yaml
```

## 评估

训练完成后运行：

```bash
python trainers/evaluate_sac_dummy.py
```

评估脚本会加载 `checkpoints/sac_dummy_final.pt`，使用 deterministic action，并输出：

- `max abs(df)`
- `max abs(V-1.0)`
- `action smoothness`
- `episode reward`

评估曲线保存到 `results/sac_dummy_evaluation.png`，评估数据保存到 `results/sac_dummy_evaluation.csv`。

## DummyGridEnv 说明

`DummyGridEnv` 不是精确电网模型，只用于 SAC 算法代码调试和训练流程验证。

状态为 5 维：

```text
obs = [df, V, P, Q, I]
```

动作为 1 维连续动作：

```text
action = Delta Vref
```

动作范围为 `[-0.02, 0.02]`。环境在第 50 个 step 后加入负载阶跃扰动，扰动会放大频率偏差，合理的 `Delta Vref` 可以部分改善频率偏差，但过大的动作会带来电压偏差和动作惩罚。

后续可以把 `DummyGridEnv` 替换为 `MatlabGridEnv`，保持 `reset()` 和 `step(action)` 接口一致即可。
