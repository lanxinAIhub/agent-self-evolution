# Agent Self-Evolution System

> 轻量级、无 GPU 依赖的 Agent 自进化算法框架

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/lanxinAIhub/agent-self-evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/lanxinAIhub/agent-self-evolution/actions/workflows)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)

## 🎯 项目概述

Agent 自进化系统通过记录每次任务执行的策略和结果，自动分析哪种策略成功率更高，并动态调整后续行为模式。系统不需要 GPU，全部在 CPU 上运行。

## 🏗️ 系统架构

```
agent-self-evolution/
├── src/agent_evolution/
│   ├── core/                       # 核心模块
│   │   ├── experience_logger.py    # 经验记录器（ExperienceLogger）
│   │   ├── strategy_analyzer.py     # 策略分析器（StrategyAnalyzer）
│   │   └── evolution_engine.py      # 进化引擎（EvolutionEngine）
│   ├── algorithms/                  # 进化算法
│   │   ├── bayesian_optimizer.py    # 贝叶斯优化（参数调优）
│   │   ├── reinforcement_learner.py # 强化学习（无模型 Q-Learning）
│   │   └── statistical_analyzer.py  # 统计学习（胜率分析）
│   └── utils/
└── tests/
```

## 🔬 核心算法

### 1. 贝叶斯优化（Bayesian Optimizer）

用于策略参数自动调优。使用高斯过程（GP）作为代理模型，期望改进（EI）作为采集函数：

```python
from agent_evolution.algorithms import BayesianOptimizer

optimizer = BayesianOptimizer(
    param_bounds={"lr": (0.001, 0.1), "temperature": (0.1, 2.0)},
    maximize=True,
)

# 迭代优化
for i in range(20):
    params = optimizer.suggest()           # 建议下一个参数
    score = evaluate_strategy(params)      # 评估
    optimizer.observe(params, score)       # 记录观察

best_params, best_score = optimizer.get_best()
```

### 2. 强化学习（Reinforcement Learner）

无模型 Q-Learning，支持 ε-贪心、Softmax 和 UCB 三种探索策略：

```python
from agent_evolution.algorithms import ReinforcementLearner

learner = ReinforcementLearner(
    learning_rate=0.1,
    discount_factor=0.9,
    exploration_mode="ucb",  # "epsilon", "softmax", "ucb"
)

learner.register_actions(["greedy", "thompson", "random"])

# 选择动作
action = learner.select_action("task_type", available_actions=["greedy", "thompson"])

# 执行并更新
learner.update(state="task_type", action=action, reward=0.8, next_state="task_type")
```

### 3. 统计学习（Statistical Analyzer）

胜率分析、A/B 测试、趋势检测：

```python
from agent_evolution.algorithms import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# 胜率 + 置信区间（Wilson Score）
win_rates = analyzer.analyze_win_rates(task_type="coding", min_samples=3)

# A/B 测试（两比例 z 检验）
result = analyzer.compare_strategies("strategy_a", "strategy_b")

# 趋势漂移检测
drift = analyzer.detect_drift("strategy_x", window=20)
```

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 完整进化示例

```python
from agent_evolution.core import EvolutionEngine

# 定义任务执行函数
def execute_task(task_id, strategy_name, params):
    # 这里是你的实际任务逻辑
    result = run_agent_task(
        task_id=task_id,
        strategy=strategy_name,
        hyperparameters=params,
    )
    return {
        "outcome": result.outcome,  # "success" | "failure" | "partial"
        "score": result.score,       # 0.0-1.0
        "duration_seconds": result.elapsed,
        "error_message": result.error,
    }

# 初始化引擎
engine = EvolutionEngine(
    db_path="~/.agent_evolution/experiences.db",
    selection_mode="thompson",  # 推荐：平衡探索与利用
)

# 注册策略
engine.register_strategy("greedy", weight=1.0, params={})
engine.register_strategy("thompson", weight=1.0, params={})

# 执行并自动进化
for task in task_list:
    result = engine.execute_and_evolve(
        task_type="coding",
        task_id=task.id,
        execute_fn=execute_task,
        context={"complexity": task.complexity},
    )
    print(f"Used: {result['strategy_used']}, Score: {result['execution_result']['score']}")

# 查看最佳策略
best = engine.get_best_strategy("coding")
print(f"Best: {best.strategy_name} (win_rate={best.win_rate:.2%})")
```

## 📊 进化循环

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  任务输入   │ ──▶ │  Thompson/UCB    │ ──▶ │   执行策略      │
└─────────────┘     │  策略选择        │     └─────────────────┘
                    └──────────────────┘              │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  进化完成   │ ◀── │  权重/参数更新    │ ◀── │   记录经验      │
│             │     │  (贝叶斯/强化学习)│     │   (ExperienceLogger) │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## 📐 核心公式

### Wilson Score CI（胜率置信区间）

```
     p + z²/2n ± z√(p(1-p)/n + z²/4n²)
CI = ─────────────────────────────────
          1 + z²/n
```

### Q-Learning 更新

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

### Thompson Sampling（Beta 分布采样）

```python
alpha = successes + 1
beta  = failures + 1
theta ~ Beta(alpha, beta)
```

## 🧪 运行测试

```bash
cd agent-self-evolution
pip install pytest
pytest tests/ -v
```

## 📁 数据存储

- **Experience DB**: `~/.agent_evolution/experiences.db` (SQLite)
- **Config**: `~/.agent_evolution/evolution_config.json`
- **Optimizer State**: `~/.agent_evolution/optimizer_{name}.json`

## 🎓 算法选择指南

| 场景 | 推荐算法 |
|------|---------|
| 参数调优（连续空间） | BayesianOptimizer |
| 策略选择（离散动作） | ReinforcementLearner (UCB) |
| 胜率对比分析 | StatisticalAnalyzer |
| 快速冷启动 | Thompson Sampling |
| 平衡探索利用 | UCB / Softmax |

## 📄 License

MIT - 自由使用，署名即可。
