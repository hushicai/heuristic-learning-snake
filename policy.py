"""
初始启发式策略（通用接口：get_action(obs) → action）。

当前实现非常 naive：朝食物方向移动，但不避墙、不避身体。
这是 evolve agent 需要改进的起点。
"""

import random

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


def get_action(obs: dict) -> int:
    """根据观测返回动作 0/1/2/3（UP/DOWN/LEFT/RIGHT）。"""
    head = obs["head"]
    food = obs["food"]
    body_set = set(obs["body"])
    width = obs["width"]
    height = obs["height"]

    if food is None:
        # 没有食物（满了），随便走
        return random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])

    # 1. 计算朝向食物的方向
    dx = food[0] - head[0]
    dy = food[1] - head[1]

    # 2. 倾向距离更长的轴先走
    candidates = []
    if abs(dx) >= abs(dy):
        if dx > 0:
            candidates.append(ACTION_RIGHT)
        elif dx < 0:
            candidates.append(ACTION_LEFT)
        if dy > 0:
            candidates.append(ACTION_DOWN)
        elif dy < 0:
            candidates.append(ACTION_UP)
    else:
        if dy > 0:
            candidates.append(ACTION_DOWN)
        elif dy < 0:
            candidates.append(ACTION_UP)
        if dx > 0:
            candidates.append(ACTION_RIGHT)
        elif dx < 0:
            candidates.append(ACTION_LEFT)

    # 3. 检查每个候选是否安全（不撞墙、不撞身体）
    for action in candidates:
        _, ax, ay = DIRECTIONS[action]
        nx, ny = head[0] + ax, head[1] + ay
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set:
            return action

    # 4. 所有首选方向都危险，fallback — 找任意安全方向
    safe = []
    for action_id, _, ax, ay in [(0, "UP", 0, -1), (1, "DOWN", 0, 1), (2, "LEFT", -1, 0), (3, "RIGHT", 1, 0)]:
        nx, ny = head[0] + ax, head[1] + ay
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set:
            safe.append(action_id)

    if safe:
        return random.choice(safe)

    # 5. 没有安全方向 — 随便选一个（会死）
    return random.choice([ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT])
