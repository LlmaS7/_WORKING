# CNN v0.3
# More CNN-centric architecture:
#   Conv-BN-ReLU blocks
#   Multiple convolutions before pooling
#   Global Average Pooling (GAP)
#   Lightweight classifier head: 128 -> 10
#
# Training framework inherited from CNN v0.1 / MLP v0.4.4:
#   Normalize
#   AdamW + Weight Decay
#   Dropout
#   ReduceLROnPlateau
#   Label Smoothing
#   Early Stopping
#   Train / Validation / Test split
#   Training curves
#
# v0.3 adds train-only data augmentation:
#   RandomAffine(degrees=10, translate=(0.1, 0.1))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1:
        # [B, 1, 28, 28] -> [B, 32, 28, 28]
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(32)

        # [B, 32, 28, 28] -> [B, 32, 28, 28]
        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(32)

        # Block 2:
        # [B, 32, 14, 14] -> [B, 64, 14, 14]
        self.conv3 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = torch.nn.BatchNorm2d(64)

        # [B, 64, 14, 14] -> [B, 64, 14, 14]
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn4 = torch.nn.BatchNorm2d(64)

        # Higher-level feature extraction:
        # [B, 64, 7, 7] -> [B, 128, 7, 7]
        self.conv5 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn5 = torch.nn.BatchNorm2d(128)

        # 28x28 -> 14x14 -> 7x7
        self.pool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Global Average Pooling:
        # [B, 128, 7, 7] -> [B, 128, 1, 1]
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight classifier head
        self.dropout = torch.nn.Dropout(p=0.1)
        self.fc = torch.nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)

        x = self.pool(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.nn.functional.relu(x)

        x = self.pool(x)

        # Higher-level features
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.nn.functional.relu(x)

        # Global Average Pooling
        x = self.gap(x)

        # [B, 128, 1, 1] -> [B, 128]
        x = torch.flatten(x, 1)

        x = self.dropout(x)

        # Raw logits
        return self.fc(x)


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


def get_data_loaders(train_batch_size, evaluation_batch_size):
    # Training data only: add mild random geometric augmentation.
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Validation / test data stay deterministic and clean.
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Create two views of the same MNIST training set so that:
    #   - training samples use random augmentation
    #   - validation samples do not
    full_train_augmented = MNIST(
        "",
        train=True,
        transform=train_transform,
        download=True,
    )

    full_train_clean = MNIST(
        "",
        train=True,
        transform=evaluation_transform,
        download=True,
    )

    test_set = MNIST(
        "",
        train=False,
        transform=evaluation_transform,
        download=True,
    )

    validation_size = 5_000
    training_size = len(full_train_augmented) - validation_size

    # Fixed split, equivalent in spirit to the previous manual_seed(42) split.
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(full_train_augmented),
        generator=generator,
    ).tolist()

    train_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    train_set = Subset(
        full_train_augmented,
        train_indices,
    )

    validation_set = Subset(
        full_train_clean,
        validation_indices,
    )

    train_data = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
    )

    validation_data = DataLoader(
        validation_set,
        batch_size=evaluation_batch_size,
        shuffle=False,
    )

    test_data = DataLoader(
        test_set,
        batch_size=evaluation_batch_size,
        shuffle=False,
    )

    return train_data, validation_data, test_data


def evaluate(data_loader, net):
    net.eval()

    loss_sum = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for x, y in data_loader:
            outputs = net(x)
            predictions = outputs.argmax(dim=1)

            # Evaluation keeps ordinary hard-label cross entropy.
            loss_sum += torch.nn.functional.cross_entropy(
                outputs,
                y,
                reduction="sum",
            ).item()

            n_correct += (predictions == y).sum().item()
            n_total += y.size(0)

    average_loss = loss_sum / n_total
    accuracy = n_correct / n_total

    return average_loss, accuracy


def main():
    torch.manual_seed(42)

    train_data, validation_data, test_data = get_data_loaders(
        train_batch_size=64,
        evaluation_batch_size=1000,
    )

    net = Net()

    initial_validation_loss, initial_validation_accuracy = evaluate(
        validation_data,
        net,
    )

    print(
        f"initial validation loss: {initial_validation_loss:.4f} | "
        f"validation accuracy: {initial_validation_accuracy:.4f}"
    )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # --- HERE
    epochs = 50
    use_early_stopping = False
    # --- HERE

    early_stopping = None

    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.0001,
        )

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        sample_count = 0

        for x, y in train_data:
            optimizer.zero_grad()

            output = net(x)

            loss = torch.nn.functional.cross_entropy(
                output,
                y,
                label_smoothing=0.05,
            )

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        average_train_loss = running_loss / sample_count

        validation_loss, validation_accuracy = evaluate(
            validation_data,
            net,
        )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(average_train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        should_stop = False
        patience_status = ""

        if early_stopping is not None:
            should_stop = early_stopping.update(
                validation_loss,
                net,
                epoch + 1,
            )

            patience_status = (
                f" | patience: "
                f"{early_stopping.epochs_without_improvement}/"
                f"{early_stopping.patience}"
            )

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train loss: {average_train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f} | "
            f"lr: {current_lr:.6f}"
            f"{patience_status}"
        )

        if should_stop:
            print(
                f"early stopping at epoch {epoch + 1}; "
                f"best epoch: {early_stopping.best_epoch}"
            )
            break

    if early_stopping is not None:
        early_stopping.restore_best_model(net)

    test_loss, test_accuracy = evaluate(
        test_data,
        net,
    )

    if early_stopping is not None:
        print(
            f"best epoch: {early_stopping.best_epoch} | "
            f"test loss: {test_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )
    else:
        print(
            f"test loss: {test_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )

    epoch_numbers = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(
        1,
        3,
        num="training curves",
        figsize=(15, 4),
    )

    axes[0].plot(
        epoch_numbers,
        train_losses,
        marker="o",
    )
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Loss")

    axes[1].plot(
        epoch_numbers,
        validation_losses,
        marker="o",
    )
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average Loss")

    axes[2].plot(
        epoch_numbers,
        validation_accuracies,
        marker="o",
    )
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")

    fig.tight_layout()

    with torch.no_grad():
        for n, (x, _) in enumerate(test_data):
            if n > 3:
                break

            predict = net(x[0:1]).argmax(dim=1).item()

            # Undo normalization for visualization only.
            image = x[0] * 0.3081 + 0.1307

            plt.figure(f"sample {n}")
            plt.imshow(image.view(28, 28), cmap="gray")
            plt.title("prediction: " + str(predict))

    plt.show()


if __name__ == "__main__":
    main()
