import unittest

import numpy as np

from flymerge.game import Orchard, RADII, tick


class GameTest(unittest.TestCase):
    def test_seeded_episode_and_bounds(self):
        left, right = Orchard(missing=.12), Orchard(missing=.12)
        np.testing.assert_array_equal(left.reset(seed=36)[0], right.reset(seed=36)[0])
        for bad in (-1, 15, 2.4):
            with self.assertRaises(ValueError):
                left.step(bad)
        for action in [2, 4, 7, 14, 12] * 60:
            a, b = left.step(action), right.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])
            self.assertTrue(left.observation_space.contains(a[0]))
            if a[2] or a[3]:
                break
        else:
            self.fail('Episode did not end')

    def test_merge(self):
        body = np.array([[50., 100, 0, 0, 0], [52, 100, 0, 0, 0]])
        count, score = tick(body, 2)
        self.assertEqual((count, score, int(body[0, 4])), (1, 1, 1))

    def test_overflow(self):
        env = Orchard()
        env.reset(seed=1)
        tier = 6
        r = RADII[tier]
        env.n = 4
        for i in range(env.n):
            env.body[i] = r, 440 - r * (2 * i + 1), 0, 0, tier
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)


if __name__ == '__main__':
    unittest.main()
