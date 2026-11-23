import unittest

from flymerge.afterstate_data import candidate_boards, targets
from flymerge.afterstate_model import Compact, loss_parts
from flymerge.game import Orchard
from flymerge.policy import np, torch


class AfterstateTest(unittest.TestCase):
    def test_cap_and_overflow_mask(self):
        score, risk, score_ok, risk_ok = targets([100,3,5,7], False, 2)
        np.testing.assert_array_equal(score, [8,12,7,0])
        np.testing.assert_array_equal(score_ok, [1,1,0,0])
        np.testing.assert_array_equal(score_ok,risk_ok)
        self.assertFalse(risk.any())
        score, risk, ok, _ = targets([100,3,5,7], True, 2)
        np.testing.assert_array_equal(risk, [0,1,1,0])
        np.testing.assert_array_equal(ok, [1,1,1,0])

    def test_only_predecision_next_fruit_is_encoded(self):
        env = Orchard(rules='transfer')
        obs,_ = env.reset(seed=12)
        boards,_,_ = candidate_boards(obs,np.arange(15))
        np.testing.assert_array_equal(boards[:,-2],np.full(15,obs[-1]))
        np.testing.assert_array_equal(boards[:,-1],-1)
        self.assertTrue(all(env.observation_space.contains(x) for x in boards))

    def test_padding_does_not_change_compact_output(self):
        env = Orchard(rules='transfer')
        first,_ = env.reset(seed=13)
        second,*_ = env.step(0)
        net = Compact()
        one = net(torch.from_numpy(first[None]))
        many = net(torch.from_numpy(np.stack([first,second])))[:1]
        torch.testing.assert_close(one,many)

    def test_loss_masks_unknown_targets(self):
        p = torch.tensor([[0.,2.],[1.,-8.]])
        known = torch.tensor([True,False])
        targets = torch.tensor([0.,1.])
        changed = torch.tensor([0.,0.])
        first = loss_parts(p,targets,targets,known,known)
        second = loss_parts(p,changed,changed,known,known)
        torch.testing.assert_close(first,second)
