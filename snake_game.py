"""
Snake — 纯 Python 贪吃蛇游戏环境，零外部依赖。

API 设计对标 gym.Env 风格，方便 heuristic policy 接入。

Usage:
    from snake_game import SnakeGame
    game = SnakeGame(width=20, height=20)
    obs = game.reset()
    done = False
    while not done:
        action = policy(obs)  # 0:UP 1:DOWN 2:LEFT 3:RIGHT
        obs, reward, done, info = game.step(action)
    print(f"Score: {info['score']}")
"""

import random
from typing import Literal, Optional

Grid = list[list[int]]

Action = Literal[0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

# 网格值
EMPTY = 0
SNAKE_HEAD = 1
SNAKE_BODY = 2
FOOD = 3


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

    def reset(self) -> dict:
        """重置游戏，返回观测。"""
        cx, cy = self.width // 2, self.height // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = (1, 0)  # 向右
        self.score = 0
        self.steps = 0
        self._place_food()
        return self._observe()

    def step(self, action: Action) -> tuple[dict, float, bool, dict]:
        """执行动作，返回 (obs, reward, done, info)。"""
        # 方向映射
        dir_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        desired = dir_map[action]

        # 不允许 180 度掉头
        dx, dy = desired
        if self.snake[0][0] + dx == self.snake[1][0] and self.snake[0][1] + dy == self.snake[1][1]:
            desired = self.direction

        self.direction = desired
        head = self.snake[0]
        dx, dy = desired
        new_head = (head[0] + dx, head[1] + dy)
        self.steps += 1

        # 碰撞检测：墙壁或自身
        died = False
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            died = True
        elif new_head in self.snake[:-1]:  # 允许吃到尾巴（尾巴本步会移动）
            died = True

        if died:
            return self._observe(), -1.0, True, {"score": self.score, "steps": self.steps}

        # 移动蛇
        self.snake.insert(0, new_head)
        ate = new_head == self.food
        if ate:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return self._observe(), 1.0 if ate else 0.0, False, {"score": self.score, "steps": self.steps}

    def _place_food(self):
        """在空格子随机放置食物。"""
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in occupied]
        if not free:
            # 蛇铺满整个格子 = 赢了
            self.food = None
            return
        self.food = self.rng.choice(free)

    def _observe(self) -> dict:
        """返回当前观测（可读的 dict，heuristic 直接使用）。"""
        head = self.snake[0]
        body = self.snake[1:]
        return {
            "head": head,
            "body": body,
            "food": self.food,
            "direction": self.direction,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "steps": self.steps,
            "snake_len": len(self.snake),
        }

    def render_grid(self) -> str:
        """返回 ASCII 网格字符串，用于调试/日志。"""
        grid = [[EMPTY] * self.width for _ in range(self.height)]
        for x, y in self.snake:
            if (x, y) == self.snake[0]:
                grid[y][x] = SNAKE_HEAD
            else:
                grid[y][x] = SNAKE_BODY
        if self.food:
            grid[self.food[1]][self.food[0]] = FOOD

        chars = {EMPTY: ".", SNAKE_HEAD: "H", SNAKE_BODY: "o", FOOD: "*"}
        return "\n".join("".join(chars[c] for c in row) for row in grid)

    def is_occupied(self, x: int, y: int) -> bool:
        """判断某位置是否被蛇占据（不含尾巴尖——它本步会移走）。"""
        return (x, y) in self.snake[:-1]

    def is_wall(self, x: int, y: int) -> bool:
        return not (0 <= x < self.width and 0 <= y < self.height)

    def manhattan_to_food(self, pos: tuple[int, int]) -> int:
        if self.food is None:
            return 0
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])
