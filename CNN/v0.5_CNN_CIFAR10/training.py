"""Training, evaluation, and early-stopping logic."""

import torch
from config import LABEL_SMOOTHING


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def update(self, validation_loss, net, epoch):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.best_model_state = {
                name: parameter.detach().clone()
                for name, parameter in net.state_dict().items()
            }
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience

    def restore_best_model(self, net):
        if self.best_model_state is not None:
            net.load_state_dict(self.best_model_state)


def train_one_epoch(data_loader, net, optimizer, device):
    net.train()
    running_loss = 0.0
    sample_count = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = net(x)

        loss = torch.nn.functional.cross_entropy(
            output,
            y,
            label_smoothing=LABEL_SMOOTHING,
        )
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

    return running_loss / sample_count


def evaluate(data_loader, net, device):
    net.eval()
    loss_sum = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = net(x)
            predictions = outputs.argmax(dim=1)

            loss_sum += torch.nn.functional.cross_entropy(
                outputs,
                y,
                reduction="sum",
            ).item()

            n_correct += (predictions == y).sum().item()
            n_total += y.size(0)

    return loss_sum / n_total, n_correct / n_total
