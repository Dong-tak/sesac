import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from losses.losses import Ball_Detection_Loss


class BallLossTest(unittest.TestCase):
    def test_positive_weight_increases_penalty_for_missed_ball(self):
        prediction = torch.full((1, 6), 0.1)
        target = torch.tensor([[0., 1., 0., 0., 1., 0.]])
        ordinary = Ball_Detection_Loss(3, 3)(prediction, target)
        weighted = Ball_Detection_Loss(3, 3, positive_weight=20)(prediction, target)
        self.assertGreater(weighted.item(), ordinary.item())

    def test_negative_only_loss_is_unchanged(self):
        prediction = torch.full((1, 6), 0.1)
        target = torch.zeros((1, 6))
        self.assertTrue(torch.allclose(Ball_Detection_Loss(3, 3)(prediction, target),
                                       Ball_Detection_Loss(3, 3, positive_weight=20)(prediction, target)))


if __name__ == '__main__':
    unittest.main()
