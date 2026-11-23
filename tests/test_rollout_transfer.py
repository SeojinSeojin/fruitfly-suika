import json
import pickle
import unittest

from flymerge.afterstate_eval import load_models
from flymerge.planner_experiment import environment
from flymerge.policy import np
from flymerge.rollout_eval import decision
from flymerge.rollout_transfer import futures,reconstruct


class TransferTest(unittest.TestCase):
    def test_preview_and_reconstruction(self):
        env = environment('clean')
        env.reset(seed=123)
        obs,*_ = env.step(7)
        board,count,current,preview = reconstruct(obs)
        self.assertEqual(count,1)
        self.assertEqual((current,preview),(env.current,env.upcoming))
        np.testing.assert_array_equal(board[:,2:4],0.)
        fixed = futures(preview,np.random.default_rng(700),6)
        self.assertTrue(all(next_fruit==preview for next_fruit,_ in fixed))
        hidden = futures(-10,np.random.default_rng(700),6)
        self.assertTrue(all(0<=fruit<=4 for fruit,_ in hidden))
        self.assertGreater(len(set(fruit for fruit,_ in hidden)),1)

    def test_paired_search_repeatability_and_budget(self):
        policy,models = load_models()
        env = environment('joint')
        obs,_ = env.reset(seed=27)
        before = pickle.dumps(env)
        for mode,limit in (('fixed',18),('adaptive',48)):
            first = decision(env,obs,policy,models,mode,27,1)
            self.assertEqual(first,decision(env,obs,policy,models,mode,27,1))
            action,trace = first
            self.assertEqual(trace,json.loads(json.dumps(trace)))
            self.assertEqual(before,pickle.dumps(env))
            self.assertTrue(env.action_space.contains(action))
            self.assertLessEqual(len(trace['actions'])*trace['tails'],limit)
            self.assertTrue(all(len(row)==trace['tails'] for row in trace['rollouts']))
