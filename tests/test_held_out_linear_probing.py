
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from modelling.held_out_linear_probing import LinearClassifier, get_metrics, run_linear_probe

class TestLinearClassifier(unittest.TestCase):
    def setUp(self):
        self.model = LinearClassifier(embed_dimension=10, num_outputs=2)
        self.inputs = torch.randn(5, 10)
        self.targets = torch.randint(0, 2, (5,))

    def test_forward(self):
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, (5, 2))

    def test_training_step(self):
        batch = (self.inputs, self.targets)
        loss = self.model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_validation_step(self):
        batch = (self.inputs, self.targets)
        loss = self.model.validation_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_test_step(self):
        batch = (self.inputs, self.targets)
        loss = self.model.test_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.model = LinearClassifier(embed_dimension=10, num_outputs=2)
        self.device = 'cpu'
        self.inputs = torch.randn(20, 10)
        self.targets = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=5)

    def test_get_metrics(self):
        metrics = get_metrics(self.model, self.device, self.dataloader)
        self.assertIn("balanced accuracy", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion matrix", metrics)

class TestRunLinearProbe(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.log_dir = './logs'
        self.num_epochs = 1
        self.patience = 1
        self.embed_dim = 10
        self.num_classes = 2
        self.inputs = torch.randn(20, 10)
        self.targets = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=5)

    def test_run_linear_probe(self):
        test_metrics, test_out_metrics = run_linear_probe(
            self.seed, self.log_dir, self.num_epochs, self.patience,
            self.embed_dim, self.num_classes, self.dataloader, self.dataloader, self.dataloader
        )
        self.assertIsInstance(test_metrics, dict)
        self.assertIsInstance(test_out_metrics, dict)

if __name__ == '__main__':
    unittest.main()