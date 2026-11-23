import copy
import pickle
import unittest

from flymerge.planner_experiment import environment, overflow
from flymerge.policy import np
from flymerge.rollout_teacher import HORIZON, paired_futures, score_actions, settle_exact


class TeacherTest(unittest.TestCase):
    def test_exact_drop_matches_engine(self):
        env = environment('joint', cap=450)
        env.reset(seed=560)
        for action in (15//2,0,13,5,10):
            trial = env.body.copy()
            n,points,ended = settle_exact(trial,env.n,env.current,action,env.physics)
            _,earned,_,_,_ = env.step(action)
            self.assertEqual((n,points,ended),(env.n,earned,overflow(env)))
            np.testing.assert_array_equal(trial,env.body)

    def test_rollouts_preserve_parent_and_have_full_window(self):
        for profile in ('clean','joint'):
            env = environment(profile,cap=1)
            env.reset(seed=71)
            data = pickle.dumps(env)
            futures = paired_futures(np.random.default_rng(407))
            first = score_actions(env,[0,14],futures)
            self.assertEqual(first,score_actions(env,[0,14],futures))
            self.assertEqual(data,pickle.dumps(env))
            for row in first:
                self.assertTrue(all(x['subsequent_drops']==HORIZON for x in row['rollouts']))

    def test_sampled_tail_starts_after_visible_preview(self):
        env = environment('clean')
        env.reset(seed=881)
        seed,tail,_ = paired_futures(np.random.default_rng(10))[0]
        clone = copy.deepcopy(env)
        clone.np_random = np.random.default_rng(seed)
        visible_preview = clone.upcoming
        obs,*_ = clone.step(2)
        self.assertEqual(round(obs[-2]*10),visible_preview)
        self.assertEqual(round(obs[-1]*10),tail[0])
