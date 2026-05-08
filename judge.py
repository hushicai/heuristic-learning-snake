"""
评测脚本 — Heuristic Learning 反馈信号的核心。

用法:
    # 查看当前策略的分数
    python judge.py

    # 检查是否达到阈值（退出码 0=通过）
    python judge.py --check 10

输出:
    - stdout: 自然语言报告 + JSON 块
    - 退出码: 0=基准正常, 1=有问题

evolve 的 bench 命令会检测此脚本，
agent 读取输出了解策略质量并迭代改进。
"""

import argparse
import json
import sys
import statistics

from snake_game import SnakeGame
from policy import get_action


def run_episode(seed: int, max_steps: int = 2000) -> dict:
    """运行一局游戏，返回结果。"""
    game = SnakeGame(width=20, height=20, seed=seed)
    obs = game.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        action = get_action(obs)
        obs, reward, done, info = game.step(action)
        total_reward += reward
        step += 1

    return {
        "seed": seed,
        "score": info["score"],
        "steps": step,
        "died": done,
    }


def run_benchmark(num_episodes: int = 20) -> list[dict]:
    """跑多局，返回结果列表。"""
    results = []
    for i in range(num_episodes):
        result = run_episode(seed=i)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Snake HL 评测")
    parser.add_argument("--episodes", type=int, default=20, help="测试局数")
    parser.add_argument("--check", type=float, default=None, help="阈值检查（退出码）")
    args = parser.parse_args()

    results = run_benchmark(args.episodes)
    scores = [r["score"] for r in results]

    summary = {
        "episodes": len(results),
        "scores": scores,
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }

    # 输出报告（agent 可读）
    print("=" * 50)
    print("Snake 策略评测报告")
    print("=" * 50)
    print(f"局数:       {summary['episodes']}")
    print(f"平均分:     {summary['mean']:.2f}")
    print(f"中位数:     {summary['median']:.2f}")
    print(f"最高分:     {summary['max']}")
    print(f"最低分:     {summary['min']}")
    print(f"标准差:     {summary['stdev']:.2f}")
    print(f"分数分布:   {scores}")

    # JSON 块 — agent 用 parse 提取
    print()
    print("---BEGIN JSON---")
    print(json.dumps(summary, indent=2))
    print("---END JSON---")

    # 阈值检查
    if args.check is not None:
        if summary["mean"] >= args.check:
            print(f"\n✓ 达标: 均值 {summary['mean']:.2f} >= 阈值 {args.check}")
            sys.exit(0)
        else:
            print(f"\n✗ 未达标: 均值 {summary['mean']:.2f} < 阈值 {args.check}")
            sys.exit(1)


if __name__ == "__main__":
    main()
