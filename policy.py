"""
BFS 寻路 + 洪水填充 + 尾巴追踪安全策略。

get_action(obs) → action:
  1. BFS 找食物最短路径 + 前瞻评估，安全则沿路径走
  2. 无法安全到达食物时，追踪蛇尾保持生存
  3. 最终回退：选洪水填充空间最大的安全方向
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


def is_safe_move(head, new_pos, body_set, width, height):
    """检查 new_pos 是否在安全边界内且不在身体上。"""
    x, y = new_pos
    return 0 <= x < width and 0 <= y < height and new_pos not in body_set


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
    snake_len = obs["snake_len"]

    # 生成所有安全候选（不撞墙、不撞身体；尾巴本步会移走，允许移向尾巴）
    tail = obs["body"][-1] if obs["body"] else None
    safe_body = body_set - {tail} if tail is not None else body_set
    safe_candidates = []
    for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        nx, ny = head[0] + dx, head[1] + dy
        if is_safe_move(head, (nx, ny), safe_body, width, height):
            safe_candidates.append((action, (nx, ny)))

    if not safe_candidates:
        return 0  # 必死，随便走

    # 缓存 flood fill：模拟移动后尾巴移走
    tail = obs["body"][-1] if obs["body"] else head
    move_body = (body_set | {head}) - {tail}  # 移动后身体：旧头变身体，尾巴移走
    flood_cache = {}
    for _, pos in safe_candidates:
        flood_cache[pos] = flood_fill(pos, move_body, width, height)

    # 策略 1: BFS 找到食物的最短路径 + 前瞻评估
    if food is not None:
        path = bfs_path(head, food, body_set, width, height)
        if path:
            first_step = path[0]
            # 前瞻：模拟吃完食物后的状态（尾巴不移走，身体增长）
            future_body = body_set | {head}
            future_space = flood_fill(first_step, future_body, width, height)
            if future_space >= snake_len - 1:
                for action, pos in safe_candidates:
                    if pos == first_step:
                        return action

    # 策略 2: 追踪尾巴（保持生存）
    tail_body_set = body_set - {tail}  # 尾巴会移动，视为可通过
    tail_path = bfs_path(head, tail, tail_body_set, width, height)
    if tail_path:
        first_step = tail_path[0]
        for action, pos in safe_candidates:
            if pos == first_step:
                if flood_cache[pos] >= snake_len:
                    return action

    # 策略 3: 选空间最大的安全方向（最终回退）
    best_action = safe_candidates[0][0]
    best_space = -1
    for action, pos in safe_candidates:
        space = flood_cache[pos]
        if space > best_space:
            best_space = space
            best_action = action

    return best_action
