import unittest
from pathlib import Path

from flymerge.game import Orchard
from flymerge.planner_experiment import load, logp, np
from flymerge.planning import drop_values
from flymerge.robust_planner import choose, simulate
from flymerge.verify import unpack


class RobustPlannerTest(unittest.TestCase):
    def test_repeatability_bounds_and_nominal_outcomes(self):
        env = Orchard(rules='transfer')
        obs, _ = env.reset(seed=500)
        for action in (0, 13, 6, 4, 14):
            obs, *_ = env.step(action)
        prior = logp(load(), obs)
        first = choose(obs, prior, np.random.default_rng(8))
        self.assertEqual(first, choose(obs, prior, np.random.default_rng(8)))
        self.assertTrue(env.action_space.contains(first[0]))
        outcomes, _ = simulate(obs, np.arange(15), np.zeros((1,80,2)), np.zeros((1,3)))
        np.testing.assert_array_equal(outcomes[:,0], drop_values(obs))

    def test_frozen_legacy_archive(self):
        policy = load()
        replay = unpack(Path('artifacts/transfer/demo-game-over/website.js'))
        for decision in replay['decisions']:
            obs = np.array(decision['observation'], dtype=np.float32)
            action, _ = policy.decide(obs, None, 'hybrid', trace=False)
            self.assertEqual(action, decision['action'])
