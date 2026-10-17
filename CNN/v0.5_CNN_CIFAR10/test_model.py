"""Small CPU checks; no dataset download or full training is required."""

import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import Net, ResidualBlock
from training import EarlyStopping, evaluate, train_one_epoch


class ResidualCNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_identity_and_signed_residual_addition(self):
        block = ResidualBlock(4, 4, use_residual=True).eval()
        x = torch.ones(2, 4, 8, 8)
        with torch.no_grad():
            block.conv2.weight.zero_()
            # The branch may subtract from the shortcut before the final ReLU.
            block.bn2.bias.fill_(-0.5)
            torch.testing.assert_close(block(x), x * 0.5)
            block.bn2.bias.zero_()
            torch.testing.assert_close(block(x), x)

    def test_projection_handles_channels_and_downsampling(self):
        for channels, stride in ((8, 1), (4, 2), (8, 2)):
            with self.subTest(channels=channels, stride=stride):
                block = ResidualBlock(4, channels, stride=stride, use_residual=True).eval()
                x = torch.randn(2, 4, 16, 16, requires_grad=True)
                with torch.no_grad():
                    block.conv2.weight.zero_()
                    block.bn2.bias.zero_()
                actual = block(x)
                expected = torch.relu(block.shortcut(x))
                torch.testing.assert_close(actual, expected)
                self.assertEqual(
                    actual.shape, (2, channels, 16 // stride, 16 // stride)
                )
                actual.square().mean().backward()
                self.assertGreater(block.shortcut[0].weight.grad.abs().sum().item(), 0)

    def test_default_network_accepts_single_and_multiple_images(self):
        for use_residual in (True, False):
            with self.subTest(use_residual=use_residual), torch.no_grad():
                net = Net(use_residual=use_residual).eval()
                for batch_size in (1, 3):
                    outputs = net(torch.randn(batch_size, 3, 32, 32))
                    self.assertEqual(outputs.shape, (batch_size, 10))
                    self.assertTrue(torch.isfinite(outputs).all().item())

    def test_disabled_shortcuts_do_not_pass_input_to_output(self):
        for channels, stride in ((4, 1), (8, 1), (4, 2), (8, 2)):
            with self.subTest(channels=channels, stride=stride):
                block = ResidualBlock(
                    4, channels, stride=stride, use_residual=False
                ).eval()
                with torch.no_grad():
                    block.conv2.weight.zero_()
                    x = torch.ones(2, 4, 16, 16)
                    torch.testing.assert_close(
                        block(x), torch.zeros(2, channels, 16 // stride, 16 // stride)
                    )

    def test_disabled_network_receives_gradients_from_the_first_step(self):
        net = Net(
            base_channels=8, blocks_per_stage=(1, 1, 1), use_residual=False
        )
        original_stem = net.stem[0].weight.detach().clone()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        inputs = torch.randn(4, 3, 32, 32)
        loss = torch.nn.functional.cross_entropy(net(inputs), torch.arange(4))
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        # Catch zero-initialized BN layers cutting off the plain network.
        for layer in (net.stem[0], net.stage1[0].conv1, net.stage3[0].conv2):
            self.assertTrue(torch.isfinite(layer.weight.grad).all().item())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(net.stem[0].weight, original_stem))

    def test_training_evaluation_and_best_state_restoration(self):
        # A smaller configurable instance exercises the same training path.
        net = Net(base_channels=8, blocks_per_stage=(1, 1, 1), use_residual=True)
        loader = DataLoader(
            TensorDataset(torch.randn(6, 3, 32, 32), torch.arange(6)),
            batch_size=3,
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        original_classifier = net.fc.weight.detach().clone()
        loss = train_one_epoch(loader, net, optimizer, torch.device("cpu"))
        self.assertTrue(math.isfinite(loss))
        self.assertFalse(torch.equal(net.fc.weight, original_classifier))
        # Zero-initialized residual branches must open up and receive gradients.
        self.assertGreater(net.stage1[0].bn2.weight.abs().sum().item(), 0)
        self.assertGreater(net.stage1[0].conv1.weight.grad.abs().sum().item(), 0)

        val_loss, val_accuracy = evaluate(loader, net, torch.device("cpu"))
        self.assertTrue(math.isfinite(val_loss))
        self.assertTrue(0 <= val_accuracy <= 1)

        early_stopping = EarlyStopping(patience=1)
        self.assertFalse(early_stopping.update(val_loss, net, epoch=1))
        best_outputs = net(loader.dataset.tensors[0]).detach().clone()
        with torch.no_grad():
            net.fc.bias.add_(1)
        self.assertTrue(early_stopping.update(val_loss + 1, net, epoch=2))
        early_stopping.restore_best_model(net)
        torch.testing.assert_close(net(loader.dataset.tensors[0]), best_outputs)


if __name__ == "__main__":
    unittest.main()
