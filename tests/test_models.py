import unittest

import torch
import torch.nn
import torch.nn.parallel

from few_shot_learning.models import AdaptiveHeadClassifier


class TestAdaptiveHeadClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_features = [100, 5, 10]
        cls.model = AdaptiveHeadClassifier(cls.out_features)
        # image-like dummy data
        cls.dummy_data = torch.randn(128, 3, 28, 28)
        cls.dummy_target = [torch.randint(0, c, (128, )) for c in cls.out_features]

    def test_training(self):
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)
            self.assertTrue(output.shape[-1] == out_features)

            loss = criterion(output, self.dummy_target[active])
            loss.backward()

    def test_eval(self):

        self.model.eval()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)

            self.assertTrue(output.shape[-1] == out_features)

    def test_distributed(self):
        model = torch.nn.DataParallel(self.model)

        if torch.cuda.is_available():
            model.cuda()

        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        for active, out_features in enumerate(self.out_features):
            model.module.set_active(active)
            output = model(self.dummy_data)
            self.assertTrue(output.shape[-1] == out_features)

            loss = criterion(output, self.dummy_target[active])
            loss.backward()

    def test_correct_grad(self):
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)

            loss = criterion(output, self.dummy_target[active])
            loss.backward()

            for other in range(len(self.out_features)):
                grad = self.model.model_head.fcs[other].weight.grad
                if active == other:
                    self.assertTrue((grad is not None),
                                    "No gradients accumulated in active"
                                    "head {}".format(other))
                else:
                    self.assertTrue((grad is None),
                                    "Gradients accumulated in head"
                                    "{} although not active".format(other))

            # empty gradients again
            self.model.model_head.fcs[active].weight.grad = None



