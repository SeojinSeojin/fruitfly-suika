import unittest

from flymerge.afterstate_eval import load_models
from flymerge.planner_experiment import environment
from flymerge.policy import np
from flymerge.rollout_eval import decision
from flymerge.state_estimation import match,observed_body,track
from flymerge.state_search import decide


class StateTest(unittest.TestCase):
    def test_same_tier_only_and_ambiguous_identity(self):
        one=observed_body(np.array([[.3,.4,0]]))
        old=observed_body(np.array([[.3,.4,1],[.3,.4,0],[.301,.4,0]]))
        self.assertEqual(int(match(one,old,17.,3.)[0]),-1)
        self.assertEqual(int(match(one,old[:2],17.,3.)[0]),1)

    def test_timestamped_velocity_and_fallback(self):
        env=environment('clean')
        observation,_=env.reset(seed=12)
        observation[:3]=.40,.70,.20
        points=np.array([[.4,.7,2.]])
        history=[(-20/60,points-np.array([.04,0.,0.])),
                 (-10/60,points-np.array([.02,0.,0.])),(0.,points)]
        (board,_,_,_),n=track(observation,history,True)
        self.assertEqual(n,1)
        self.assertAlmostEqual(board[0,2],.82)
        (fallback,_,_,_),n=track(observation,history[-2:],True)
        self.assertEqual(n,0)
        self.assertEqual(fallback[0,2],0.)

    def test_current_variant_reproduces_existing_rollouts(self):
        policy,models=load_models()
        env=environment('joint')
        obs,_=env.reset(seed=77)
        for drop in range(1,4):
            old=decision(env,obs,policy,models,'fixed',77,drop)[0]
            action,details=decide(obs,[],policy,models,'current',77,drop,{'physics':[],'tracking_velocity':False})
            self.assertEqual(old,action)
            self.assertEqual(len(details['actions']),6)
            obs,*_=env.step(action)
