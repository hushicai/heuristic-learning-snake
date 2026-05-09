"""
BFS 寻路 + 洪水填充 + 尾巴追踪安全策略。

get_action(obs) → action:
  1. BFS 找食物最短路径 + 二阶安全检查（空间 + 尾巴可达性）
  2. 无法安全到达食物时，追踪蛇尾保持生存（靠墙时防死胡同）
  3. 最终回退：综合空间/安全出口/远离墙壁评分选方向
  - 极端长蛇时切换动态评估优先保命
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


def flood_fill_with_tail(head, body_list, width, height):
    """改进版洪水填充：考虑蛇身移动会逐步释放空间。

    body_list[0] 是紧接 head 之后的身体段。
    body_list[i] 在 (i+1) 步后移走，BFS 到达时若 steps >= i 则可通过。
    """
    body_set = set(body_list)
    pos_to_idx = {pos: i for i, pos in enumerate(body_list)}
    queue = deque([(head, 0)])  # (pos, steps_ahead)
    visited = {head}
    count = 0
    while queue:
        (x, y), steps = queue.popleft()
        count += 1
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if (nx, ny) in body_set:
                    idx = pos_to_idx[(nx, ny)]
                    # body_list[i] 在 (i+1) 步后移走；邻居到达步数 = steps+1
                    # 可通过条件: steps + 1 >= idx + 1  ⟺  steps >= idx
                    if steps >= idx:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), steps + 1))
                else:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), steps + 1))
    return count


def _count_safe_moves(pos, body_set, width, height):
    """从 pos 出发，计算下一步的安全移动数（1 步前瞻）。"""
    count = 0
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in body_set:
            count += 1
    return count


def get_action(obs: dict) -> int:
    """根据观测返回动作 0/1/2/3（UP/DOWN/LEFT/RIGHT）。"""
    head = obs["head"]
    food = obs["food"]
    body_list = obs["body"]  # 不含 head，body_list[0] 紧接 head 之后
    body_set = set(body_list)
    width, height = obs["width"], obs["height"]
    snake_len = obs["snake_len"]
    grid_size = width * height

    # 生成所有安全候选（不撞墙、不撞身体；尾巴本步会移走，允许移向尾巴）
    tail = body_list[-1] if body_list else None
    safe_body = body_set - {tail} if tail is not None else body_set
    safe_candidates = []
    for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
        nx, ny = head[0] + dx, head[1] + dy
        if is_safe_move(head, (nx, ny), safe_body, width, height):
            safe_candidates.append((action, (nx, ny)))

    if not safe_candidates:
        return 0  # 必死，随便走

    # 缓存 flood fill：模拟移动后尾巴移走（静态，保守评估）
    if tail is None:
        tail = head
    move_body = (body_set | {head}) - {tail}  # 移动后身体：旧头变身体，尾巴移走
    flood_cache = {}
    for _, pos in safe_candidates:
        flood_cache[pos] = flood_fill(pos, move_body, width, height)

    # 1 步前瞻：计算每个候选方向的下一步安全移动数
    lookahead_cache = {}
    for _, pos in safe_candidates:
        # 模拟移动到 pos 后的身体状态
        next_body = (body_set | {head}) - {tail}
        lookahead_cache[pos] = _count_safe_moves(pos, next_body, width, height)

    # === 极端情况：蛇身超过网格 60%，优先保命（用动态评估） ===
    if snake_len > grid_size * 0.6:
        move_body_list = [head] + body_list[:-1] if body_list else [head]
        best_action = safe_candidates[0][0]
        best_space = -1
        for action, pos in safe_candidates:
            space = flood_fill_with_tail(pos, move_body_list, width, height)
            if space > best_space:
                best_space = space
                best_action = action
        return best_action

    # 策略 1: BFS 找到食物的最短路径 + 二阶安全检查
    if food is not None:
        path = bfs_path(head, food, body_set, width, height)
        if path:
            first_step = path[0]
            # 二阶检查：模拟吃掉食物后的状态
            # 吃完食物后尾巴不移走（蛇增长），用静态评估更准确
            future_body = body_set | {head}
            future_space = flood_fill(first_step, future_body, width, height)
            if future_space >= snake_len - 1:
                # 二阶检查：吃完食物后仍能追到尾巴（尾巴被视为可通过）
                future_tail_body = future_body - {tail}
                tail_reachable = bfs_path(first_step, tail, future_tail_body, width, height)
                if tail_reachable is not None:
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
                wx, wy = pos
                near_wall = min(wx, wy, width - 1 - wx, height - 1 - wy) < 3
                # 靠墙且仅有一个出口时，跳过此方向
                if flood_cache[pos] >= snake_len:
                    if not near_wall or lookahead_cache.get(pos, 0) > 1:
                        return action

    # 策略 3: 选空间最大的安全方向（最终回退）
    # 综合评分：空间 + 前瞻安全出口 + 远离墙壁
    best_action = safe_candidates[0][0]
    best_score = -1
    for action, pos in safe_candidates:
        space = flood_cache[pos]
        moves = lookahead_cache.get(pos, 0)
        wx, wy = pos
        wall_dist = min(wx, wy, width - 1 - wx, height - 1 - wy)
        max_wall = min(width, height) // 2
        # 空间 60% + 安全出口 20% + 远离墙壁 20%
        score = (space * 0.60
                 + (moves / 4.0) * space * 0.20
                 + (wall_dist / max_wall) * space * 0.20)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action
