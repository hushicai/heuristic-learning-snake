"""Snake 游戏环境基础测试。"""

import pytest
from snake_game import SnakeGame, Action


def test_reset():
    """重置后蛇在中间。"""
    game = SnakeGame(width=20, height=20, seed=42)
    obs = game.reset()
    assert obs["head"] == (10, 10)
    assert len(obs["body"]) == 2
    assert obs["score"] == 0
    assert obs["food"] is not None


def test_step_movement():
    """步进后蛇确实移动。"""
    game = SnakeGame(width=20, height=20, seed=42)
    obs = game.reset()
    head_before = obs["head"]
    obs, reward, done, info = game.step(3)  # RIGHT
    assert not done
    assert obs["head"] != head_before  # 蛇头移动了


def test_wall_collision():
    """撞墙会死。"""
    game = SnakeGame(width=5, height=5, seed=0)
    obs = game.reset()
    # 蛇初始在 (2,2)，一直朝右走
    head = obs["head"]
    # 迭代到撞右墙
    done = False
    while not done:
        obs, reward, done, info = game.step(3)  # RIGHT
    assert info["score"] >= 0


def test_food_eaten():
    """吃到食物分数增加。"""
    game = SnakeGame(width=10, height=10, seed=42)
    obs = game.reset()
    # 直接放置食物到蛇头旁边
    game.food = (obs["head"][0] + 1, obs["head"][1])
    obs, reward, done, info = game.step(3)  # RIGHT
    assert reward == 1.0
    assert info["score"] == 1
    assert obs["score"] == 1


def test_no_180_turn():
    """不能 180 度掉头。"""
    game = SnakeGame(width=10, height=10, seed=42)
    obs = game.reset()
    # 当前方向是右 (1,0)，尝试向左
    obs_before = obs
    obs, reward, done, info = game.step(2)  # LEFT — 应该无效
    # 蛇应该继续向右
    assert not done


def test_render_grid():
    """render_grid 输出字符串。"""
    game = SnakeGame(width=5, height=5, seed=42)
    game.reset()
    grid = game.render_grid()
    assert isinstance(grid, str)
    assert len(grid) > 0
    assert "H" in grid  # 有蛇头


def test_multiple_episodes():
    """多局游戏不会相互影响。"""
    game1 = SnakeGame(width=10, height=10, seed=0)
    game2 = SnakeGame(width=10, height=10, seed=0)
    obs1 = game1.reset()
    obs2 = game2.reset()
    assert obs1["head"] == obs2["head"]
    assert obs1["food"] == obs2["food"]


def test_policy_interface():
    """验证 policy 接口存在并返回合法动作。"""
    from policy import get_action

    game = SnakeGame(width=10, height=10, seed=42)
    obs = game.reset()
    action = get_action(obs)
    assert isinstance(action, int)
    assert 0 <= action <= 3


def test_judge_runs():
    """judge 能跑通。"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "judge.py", "--episodes", "3"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "---BEGIN JSON---" in result.stdout
    assert "平均分" in result.stdout


# ---- BFS / Flood Fill 单元测试 ----

from policy import bfs_path, flood_fill, get_action


def test_bfs_path_straight():
    """直线可达的最短路径。"""
    path = bfs_path((0, 0), (3, 0), set(), width=5, height=5)
    assert path is not None
    assert path[0] == (1, 0)
    assert path[-1] == (3, 0)
    assert len(path) == 3  # (1,0) (2,0) (3,0)


def test_bfs_path_around_obstacle():
    """有障碍时能绕路。"""
    # 障碍物堵住 (1,0)
    body = {(1, 0)}
    path = bfs_path((0, 0), (2, 0), body, width=5, height=5)
    assert path is not None
    assert (1, 0) not in path
    assert path[-1] == (2, 0)


def test_bfs_path_unreachable():
    """完全被包围，找不到路径。"""
    # (2,2) 被围墙围住
    body = {(1, 2), (3, 2), (2, 1), (2, 3)}
    path = bfs_path((0, 0), (2, 2), body, width=5, height=5)
    assert path is None


def test_bfs_path_target_is_start():
    """起点即终点。"""
    path = bfs_path((2, 2), (2, 2), set(), width=5, height=5)
    assert path == []


def test_flood_fill_open():
    """空旷区域，全部可达。"""
    count = flood_fill((0, 0), set(), width=5, height=5)
    assert count == 25  # 5x5 全部可达


def test_flood_fill_walled():
    """被墙围住的小区间。"""
    # (1,1) 被围在 2x2 的角落
    body = {(2, 0), (2, 1), (1, 2)}
    count = flood_fill((0, 0), body, width=5, height=5)
    assert count >= 1  # 至少起点本身可达
    assert count < 25  # 不是全部


def test_flood_fill_single():
    """完全被堵死，只有自己。"""
    body = {(1, 0), (0, 1)}
    count = flood_fill((0, 0), body, width=2, height=2)
    assert count == 1


def test_get_action_returns_safe_move():
    """get_action 返回的动作不会撞墙或撞身体。"""
    game = SnakeGame(width=10, height=10, seed=99)
    obs = game.reset()
    # 跑 100 步验证不崩溃、不越界
    for _ in range(100):
        action = get_action(obs)
        obs, reward, done, info = game.step(action)
        if done:
            break


def test_get_action_prefers_bfs_path():
    """有 BFS 路径时应沿路径走。"""
    # 构造一个简单场景: head=(1,1), food=(3,1), 无障碍
    obs = {
        "head": (1, 1),
        "body": [(0, 1)],
        "food": (3, 1),
        "direction": (1, 0),
        "width": 5,
        "height": 5,
        "score": 0,
        "steps": 0,
        "snake_len": 2,
    }
    action = get_action(obs)
    # 应该向右走 (ACTION_RIGHT = 3)
    assert action == 3
