import unittest

from flymerge.game import HEIGHT, Orchard
from flymerge.planning import drop_values
from flymerge.policy import torch
from flymerge.spatial import FEATURE_COUNT, spatial_features


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

    def test_action_relative_first_contact(self):
        env = Orchard(rules='transfer')
        env.reset(seed=43)
        env.current = 0
        radius = env.radii[0]
        env.body[0] = env.width/2, HEIGHT-radius, 0, 0, 0
        env.n = 1
        features = spatial_features(torch.from_numpy(env.observation()[None]))
        self.assertEqual(tuple(features.shape), (1, 15, FEATURE_COUNT))
        self.assertEqual(float(features[0, 7, 5]), 1.)
        self.assertEqual(float(features[0, 0, 5]), 0.)

    def test_two_drop_search_keeps_four_legal_candidates(self):
        env = Orchard(rules='transfer')
        observation, _ = env.reset(seed=42)
        estimates = drop_values(observation, future=True)
        self.assertEqual(int(torch.isfinite(torch.from_numpy(estimates)).sum()), 4)
        self.assertTrue(env.action_space.contains(int(estimates.argmax())))
