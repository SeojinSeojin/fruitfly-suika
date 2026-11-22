import unittest

from flymerge.game import HEIGHT, Orchard
from flymerge.planning import drop_values


class LocalPlanningTest(unittest.TestCase):
    def test_prefers_a_reachable_merge(self):
        env = Orchard(rules='transfer')
        env.reset(seed=42)
        env.current = env.upcoming = 0
        radius = env.radii[0]
        env.body[0] = env.width/2, HEIGHT-radius, 0, 0, 0
        env.n = 1
        estimates = drop_values(env.observation())
        self.assertEqual(estimates.shape, (15,))
        _, reward, _, _, _ = env.step(int(estimates.argmax()))
        self.assertEqual(reward, 1)
