import unittest

from flymerge.policy import torch
from flymerge.ranker import contrast, objective


class RankingTest(unittest.TestCase):
    def test_uncertain_pairs_receive_less_weight(self):
        certain = torch.tensor([[[3.,3.,3.],[1.,1.,1.]]+[[1.,1.,1.]]*4])
        mixed = torch.tensor([[[8.,-2.,3.],[1.,1.,1.]]+[[1.,1.,1.]]*4])
        a, confidence = contrast(certain)
        b, uncertain = contrast(mixed)
        self.assertTrue(torch.equal(a,b))
        self.assertLess(uncertain[0,0].item(),confidence[0,0].item())

    def test_ordered_value_reduces_training_loss(self):
        targets = torch.tensor([[[3.,3.,3.]]+[[0.,0.,0.]]*5])
        chosen = torch.tensor([[3.,0.,0.,0.,0.,0.]])
        self.assertLess(objective(chosen,torch.zeros_like(chosen),targets,torch.zeros_like(targets)),
                        objective(-chosen,torch.zeros_like(chosen),targets,torch.zeros_like(targets)))
