import unittest

import torch
import torch.nn
import torch.nn.parallel

from few_shot_learning.models import AdaptiveHeadClassifier

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TestAdaptiveHeadClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out_features = [100, 5, 10]
        cls.model = AdaptiveHeadClassifier(cls.out_features).to(device)
        # image-like dummy data
        cls.dummy_data = torch.randn(128, 3, 28, 28).to(device)
        cls.dummy_target = [torch.randint(0, c, (128, )) for c in cls.out_features]
               
    def test_training(self):
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)
            self.assertTrue(output.shape[-1] == out_features)

            loss = criterion(output, self.dummy_target[active].to(device))
            loss.backward()
            
        self.model.zero_grad()

    def test_eval(self):

        self.model.eval()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)

            self.assertTrue(output.shape[-1] == out_features)

    def test_distributed(self):
        model = torch.nn.DataParallel(self.model)
        criterion = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            model.cuda()
            criterion = criterion.cuda()        

        model.train()

        for active, out_features in enumerate(self.out_features):
            model.module.set_active(active)
            output = model(self.dummy_data)
            self.assertTrue(output.shape[-1] == out_features)

            loss = criterion(output, self.dummy_target[active].to(device))
            loss.backward()
            
        model.zero_grad()

    def test_selective_grad(self):
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        self.model.zero_grad()

        for active, out_features in enumerate(self.out_features):
            self.model.set_active(active)
            output = self.model(self.dummy_data)

            loss = criterion(output, self.dummy_target[active].to(device))
            loss.backward()

            for other in range(len(self.out_features)):
                
                other_fc = getattr(self.model.model_head, "fc{}".format(other))
                
                grad = other_fc.weight.grad
                
                if active == other:
                    self.assertTrue(self._grad_is_all_zeros_or_none(grad),
                                    "No gradients accumulated in active"
                                    "head {}".format(other))
                else:
                    self.assertTrue(self._grad_is_all_zeros_or_none(grad),
                                    "Gradients accumulated in head"
                                    "{} although not active".format(other))

            # empty gradients again
            self.model.zero_grad()
            
    def _grad_is_all_zeros_or_none(self, grad):
        if grad is not None:
            return ~grad.to(torch.uint8).any().item()
        else:
            return True



