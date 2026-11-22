"""Orchard Drop, a seeded, decision-step fruit merging environment."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

WIDTH, HEIGHT = 320, 440
AIMS = np.linspace(0, 1, 15)
RADII = np.array([12, 16, 21, 27, 34, 43, 53, 65, 79, 94, 110])
TRANSFER_WIDTH = 410
TRANSFER_RADII = np.array([13, 16, 22, 25, 33, 43, 48, 58, 65, 81, 94])
SLOTS = 80
UNKNOWN = -1.0


def observe(fruits, current, upcoming):
    """Board points are top-origin normalized x/y and tiers are zero-based."""
    board = np.full((SLOTS, 3), UNKNOWN, np.float32)
    ordered = sorted(fruits, key=lambda f: (f[1], f[0], f[2]))
    for i, (x, y, tier) in enumerate(ordered[:SLOTS]):
        board[i] = x, y, tier / 10 if tier >= 0 else UNKNOWN
    return np.append(board.ravel(), [v / 10 if v >= 0 else UNKNOWN
                                     for v in (current, upcoming)]).astype(np.float32)


@njit(cache=True)
def tick(body, n, width=WIDTH, radii=RADII, gravity=.46, drag=.97, bounce=.19, impact=.55):
    gained = 0
    for i in range(n):
        r = radii[int(body[i, 4])]
        body[i, 3] = min(12, body[i, 3] + gravity)
        body[i, 0] += body[i, 2]
        body[i, 1] += body[i, 3]
        body[i, 2] *= drag
        if body[i, 0] < r or body[i, 0] > width - r:
            body[i, 0] = max(r, min(width - r, body[i, 0]))
            body[i, 2] *= -bounce
        if body[i, 1] > HEIGHT - r:
            body[i, 1] = HEIGHT - r
            body[i, 3] = 0
    for _ in range(4):
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                ti, tj = int(body[i, 4]), int(body[j, 4])
                dx, dy = body[j, 0] - body[i, 0], body[j, 1] - body[i, 1]
                limit = radii[ti] + radii[tj]
                if dx * dx + dy * dy < limit * limit:
                    if ti == tj and ti < len(radii) - 1:
                        for k in range(4):
                            body[i, k] = (body[i, k] + body[j, k]) * .5
                        body[i, 4] += 1
                        body[j] = body[n - 1]
                        n -= 1
                        gained += (ti + 1) * (ti + 2) // 2
                        continue
                    distance = np.sqrt(dx * dx + dy * dy)
                    ux, uy = (dx / distance, dy / distance) if distance > 1e-8 else (1., 0.)
                    shift = .5 * (limit - distance)
                    impulse = max(0., (body[i, 2] - body[j, 2]) * ux
                                  + (body[i, 3] - body[j, 3]) * uy) * impact
                    body[i, 0] -= shift * ux
                    body[j, 0] += shift * ux
                    body[i, 1] -= shift * uy
                    body[j, 1] += shift * uy
                    body[i, 2] -= impulse * ux
                    body[j, 2] += impulse * ux
                    body[i, 3] -= impulse * uy
                    body[j, 3] += impulse * uy
                j += 1
            i += 1
        for i in range(n):
            r = radii[int(body[i, 4])]
            body[i, 0] = max(r, min(width - r, body[i, 0]))
            body[i, 1] = min(HEIGHT - r, body[i, 1])
    return n, gained


class Orchard(gym.Env):
    action_space = spaces.Discrete(15)
    observation_space = spaces.Box(-1, 1, (SLOTS * 3 + 2,), np.float32)

    def __init__(self, missing=0., capture=False, rules="original", randomize_physics=False):
        self.missing = missing
        self.capture = capture
        if rules not in ('original', 'transfer'):
            raise ValueError(f'Unknown physics profile: {rules}')
        self.rules = rules
        self.randomize_physics = randomize_physics
        self.width = TRANSFER_WIDTH if rules == 'transfer' else WIDTH
        self.radii = TRANSFER_RADII if rules == 'transfer' else RADII
        self.release_y = -25 if rules == 'transfer' else 28
        self.overflow_y = 11 if rules == 'transfer' else 48

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mask_seed = self.np_random.integers(2**32)
        self.mask_rng = np.random.default_rng(mask_seed)
        dynamics = np.random.default_rng(int(mask_seed) ^ 71419)
        self.physics = (.46, .97, .19, .55)
        if self.randomize_physics:
            self.physics = tuple(dynamics.uniform(low, high) for low, high in
                                 ((.35,.57), (.94,.995), (.10,.28), (.42,.70)))
        self.body = np.zeros((SLOTS + 1, 5))
        self.n = self.score = self.drops = self.highest = 0
        self.current, self.upcoming = map(int, self.np_random.integers(0, 5, 2))
        self.done = False
        return self.observation(), self.info()

    def observation(self):
        mask = self.mask_rng.random(2) < self.missing
        return observe([(x / self.width, np.clip(y / HEIGHT, 0, 1), int(t))
                        for x, y, _, _, t in self.body[:self.n]],
                       -1 if mask[0] else self.current, -1 if mask[1] else self.upcoming)

    def info(self):
        return dict(score=self.score, highest=self.highest, length=self.drops)

    def step(self, action):
        if self.done or not self.action_space.contains(action):
            raise ValueError('Expected a legal action in a running episode')
        r = self.radii[self.current]
        self.aim = r + (self.width - 2 * r) * AIMS[action]
        self.body[self.n] = self.aim, self.release_y, 0, 0, self.current
        self.n += 1
        self.drops += 1
        gain = 0
        self.frames = []
        for frame in range(80):
            self.n, added = tick(self.body, self.n, self.width, self.radii, *self.physics)
            gain += added
            if self.capture and (frame % 3 == 0 or frame == 79):
                self.frames.append((self.body[:self.n, [0, 1, 4]].copy(), self.score + gain))
        self.score += gain
        self.highest = max(self.highest, int(self.body[:self.n, 4].max()))
        self.done = bool(self.n >= SLOTS or np.any(self.body[:self.n, 1] - self.radii[
            self.body[:self.n, 4].astype(int)] < self.overflow_y))
        self.current, self.upcoming = self.upcoming, int(self.np_random.integers(0, 5))
        truncated = self.drops >= 300
        self.done |= truncated
        return self.observation(), float(gain), self.done and not truncated, truncated, self.info()
