"""
BFS 寻路 + 洪水填充安全评估策略。

get_action(obs) → action:
  1. BFS 找食物最短路径，有则沿路径走
  2. 无路径时，选洪水填充空间最大的安全方向
"""

import random
from collections import deque

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

DIRECTIONS = [
    ("UP", 0, -1),
    ("DOWN", 0, 1),
    ("LEFT", -1, 0),
    ("RIGHT", 1, 0),
]

ACTION_MAP = {
    (0, -1): ACTION_UP,
    (0, 1): ACTION_DOWN,
    (-1, 0): ACTION_LEFT,
    (1, 0): ACTION_RIGHT,
}


def bfs_path(start, target, body_set, width, height):
    """BFS 找最短路径，返回路径列表（不含 start），找不到返回 None。"""
    if start == target:
        return []
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set and (nx, ny) not in visited:
                new_path = path + [(nx, ny)]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None


def flood_fill(start, body_set, width, height):
    """从 start 开始洪水填充，返回可达格子数。"""
    queue = deque([start])
    visited = {start}
    count = 0
    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count


def get_action(obs: dict) -> int:
    """根据观测返回动作 0/1/2/3（UP/DOWN/LEFT/RIGHT）。"""
    head = obs["head"]
    food = obs["food"]
    body_set = set(obs["body"])  # body 不含 head
    width, height = obs["width"], obs["height"]

    # 生成所有安全候选（不撞墙、不撞身体）
    safe_candidates = []
    for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set:
            safe_candidates.append((action, (nx, ny)))

    if not safe_candidates:
        return 0  # 必死，随便走

    # 策略 1: BFS 找到食物的最短路径
    if food is not None:
        path = bfs_path(head, food, body_set, width, height)
        if path:
            first_step = path[0]
            for action, pos in safe_candidates:
                if pos == first_step:
                    return action

    # 策略 2: 所有安全候选中，选洪水填充空间最大的
    best_action = safe_candidates[0][0]
    best_space = -1
    for action, (nx, ny) in safe_candidates:
        # 模拟蛇移动到 (nx, ny) 后的 body_set
        new_body = set(body_set)
        new_body.add((nx, ny))
        # 移除尾巴（蛇会移动）
        # 简化：直接从 head 位置 flood fill
        space = flood_fill((nx, ny), new_body - {head}, width, height)
        if space > best_space:
            best_space = space
            best_action = action

    return best_action
