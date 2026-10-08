"""Plot training curves and sample CIFAR-10 predictions."""

import matplotlib.pyplot as plt
import torch

from config import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES


def plot_training_curves(train_losses, validation_losses, validation_accuracies):
    epoch_numbers = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, num="training curves", figsize=(15, 4))

    axes[0].plot(epoch_numbers, train_losses, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Loss")

    axes[1].plot(epoch_numbers, validation_losses, marker="o")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average Loss")

    axes[2].plot(epoch_numbers, validation_accuracies, marker="o")
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")

    fig.tight_layout()


def show_sample_predictions(test_loader, net, device, sample_count=4):
    mean_tensor = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std_tensor = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    shown = 0
    net.eval()

    with torch.no_grad():
        for x, y in test_loader:
            for i in range(x.size(0)):
                if shown >= sample_count:
                    return

                prediction = net(x[i : i + 1].to(device)).argmax(dim=1).item()

                image = x[i] * std_tensor + mean_tensor
                image = image.clamp(0, 1)

                plt.figure(f"sample {shown}")
                plt.imshow(image.permute(1, 2, 0))
                plt.title(
                    f"prediction: {CLASS_NAMES[prediction]} | "
                    f"label: {CLASS_NAMES[y[i].item()]}"
                )
                plt.axis("off")
                shown += 1
