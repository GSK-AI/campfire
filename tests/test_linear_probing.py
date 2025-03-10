
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from modelling.linear_probing import LinearClassifier, get_metrics, run_linear_probe

class TestLinearProbing(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 128
        self.num_classes = 10
        self.batch_size = 32
        self.num_samples = 100

        self.model = LinearClassifier(self.embed_dim, self.num_classes)

        # Create dummy data
        self.inputs = torch.randn(self.num_samples, self.embed_dim)
        self.targets = torch.randint(0, self.num_classes, (self.num_samples,))
        self.dataset = TensorDataset(self.inputs, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

    def test_forward(self):
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.shape, (self.num_samples, self.num_classes))

    def test_training_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_validation_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.validation_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_test_step(self):
        batch = next(iter(self.dataloader))
        loss = self.model.test_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)

    def test_get_metrics(self):
        metrics = get_metrics(self.model, 'cpu', self.dataloader)
        self.assertIn("balanced accuracy", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion matrix", metrics)

    def test_run_linear_probe(self):
        dl_train = self.dataloader
        dl_val = self.dataloader
        dl_test = self.dataloader
        dl_out = self.dataloader

        seed = 42
        log_dir = "/tmp"
        num_epochs = 1
        patience = 1

        test_metrics, test_out_metrics = run_linear_probe(seed, log_dir, num_epochs, patience, self.embed_dim, self.num_classes, dl_train, dl_val, dl_test, dl_out)
        self.assertIsInstance(test_metrics, dict)
        self.assertIsInstance(test_out_metrics, dict)

if __name__ == "__main__":
    unittest.main()