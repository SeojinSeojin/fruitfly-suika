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

    def test_transfer_profile_releases_above_wider_board(self):
        env = Orchard(rules='transfer', capture=True)
        env.reset(seed=20261122)
        radius = env.radii[env.current]
        observation, _, terminated, truncated, _ = env.step(14)
        self.assertEqual(env.width, 410)
        self.assertEqual(env.aim, env.width - radius)
        self.assertLess(env.frames[0][0][0, 1], 0)
        self.assertGreater(observation[0], .9)
        self.assertFalse(terminated or truncated)

    def test_randomized_physics_is_seeded_and_keeps_the_fruit_sequence(self):
        normal = Orchard(rules='transfer')
        varied = Orchard(rules='transfer', randomize_physics=True)
        normal.reset(seed=17)
        varied.reset(seed=17)
        first = varied.physics
        self.assertNotEqual(first, normal.physics)
        self.assertEqual(normal.current, varied.current)
        self.assertEqual(normal.upcoming, varied.upcoming)
        varied.reset(seed=17)
        self.assertEqual(varied.physics, first)

    def test_vision_noise_is_seeded_and_does_not_change_physics(self):
        clean, noisy, repeat = (Orchard(rules='transfer', vision_noise=level) for level in (0., 1., 1.))
        for env in (clean, noisy, repeat):
            env.reset(seed=889)
        changed = False
        for action in (2, 10, 12, 0, 8, 6, 4, 11, 1, 13, 5):
            a, b, c = [env.step(action) for env in (clean, noisy, repeat)]
            self.assertEqual(a[1:], b[1:])
            np.testing.assert_array_equal(clean.body, noisy.body)
            np.testing.assert_array_equal(b[0], c[0])
            self.assertTrue(noisy.observation_space.contains(b[0]))
            changed |= not np.array_equal(a[0], b[0])
        self.assertTrue(changed)

    def test_overflow(self):
        env = Orchard()
        env.reset(seed=1)
        tier = len(RADII) - 1
        r = RADII[tier]
        env.n = 2
        for i in range(env.n):
            env.body[i] = r, 440 - r * (2 * i + 1), 0, 0, tier
        _, _, terminated, _, _ = env.step(14)
        self.assertTrue(terminated)


if __name__ == '__main__':
    unittest.main()
