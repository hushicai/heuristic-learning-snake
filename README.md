# Heuristic Learning Snake

验证 Heuristic Learning 理念的实验项目。用 x-horse-evolve 自主迭代改进贪吃蛇 AI 启发式策略，不依赖神经网络。

## 如何使用

```bash
# 1. 进入项目目录
cd ~/data/ai-project/heuristic-learning-snake

# 2. 查看当前策略的分数（基线）
.venv/bin/python judge.py

# 3. 用 x-horse-evolve 自主改进策略
x-horse-evolve "提升贪吃蛇策略的平均分，目标达到 100 分以上"

# 4. 再次评测
.venv/bin/python judge.py
```

## 理念

- Heuristic Learning = coding agent 持续迭代代码策略
- 反馈信号来自 `judge.py`（分数 + JSON）
- evolve 编排自动完成：评估 → 规划 → 实现 → 评测 → 反思

## 文件

| 文件 | 作用 |
|------|------|
| `snake_game.py` | 纯 Python 贪吃蛇环境（零依赖） |
| `policy.py` | 启发式策略（初始 naive 版本） |
| `judge.py` | 评测脚本，输出分数作为反馈 |
| `tests/` | 单元测试（evolve 的 test gate） |
| `AGENTS.md` | evolve agent 项目上下文（含工具链声明） |
