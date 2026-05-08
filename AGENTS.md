# Heuristic Learning Snake — 启发式学习验证项目

验证 Heuristic Learning 理念：coding agent 通过 feedback loop 迭代改进代码策略，不训练神经网络。

## 工具链声明

- **bench**: `.venv/bin/python judge.py --check 15`
- **test**: `.venv/bin/python -m pytest tests/ -v`
- **build**: `null`（纯 Python，无需构建）
- **lint**: `null`（暂无 linter 配置）
- **language**: python
- **packageManager**: pip

## 项目结构

```
snake_game.py    # 游戏环境（纯 Python，零依赖）
policy.py        # 启发式策略 — agent 需要改进的对象
judge.py         # 评测脚本，输出分数 → agent 的反馈信号
tests/           # 单元测试
```

## 评测方式

```
.venv/bin/python judge.py              # 跑 20 局，输出分数统计
.venv/bin/python judge.py --check N    # 检查平均分是否 >= N
```

## 改进方向（供 agent 规划参考）

1. **避墙** — 检测蛇头到墙壁距离，不朝墙走
2. **避身体** — 检测蛇身体，不走入死路
3. **BFS 寻路** — 用 BFS 找到到食物的最短安全路径
4. **尾巴追踪** — 食物远时追尾巴保持生存
5. **空间评估** — 评估移动后剩余自由空间
6. **贪吃策略** — 综合考虑食物距离 + 空间安全
