import unittest

import numpy as np

from flymerge.live_diagnostic import histogram, merge_points, solve_merges


class MergeAccounting(unittest.TestCase):
    def test_chain_and_points(self):
        before = histogram([0, 0, 1, 3])
        after = histogram([2, 3])  # two cherries -> strawberry, two strawberries -> grape
        merges = solve_merges(before, after)
        self.assertEqual(merges[:3], [1, 1, 0])
        self.assertEqual(merge_points(merges), 1+3)

    def test_inconsistent_transition(self):
        self.assertIsNone(solve_merges(histogram([0, 1]), histogram([0, 0])))
        self.assertIsNone(solve_merges(histogram([2]), histogram([2, 2])))

    def test_no_change(self):
        tiers = np.array([0, 4, 4, 7])
        self.assertEqual(merge_points(solve_merges(histogram(tiers), histogram(tiers))), 0)


if __name__ == '__main__':
    unittest.main()
