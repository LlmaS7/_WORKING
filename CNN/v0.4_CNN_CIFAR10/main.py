"""Entry point for CNN v0.4 on CIFAR-10."""

import matplotlib.pyplot as plt
import torch
from config import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SEED,
    USE_EARLY_STOPPING,
    WEIGHT_DECAY,
)
from data import get_data_loaders
from model import Net
from training import EarlyStopping, evaluate, train_one_epoch
from visualize import plot_training_curves, show_sample_predictions


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, validation_loader, test_loader = get_data_loaders()
    net = Net().to(device)

    initial_validation_loss, initial_validation_accuracy = evaluate(
        validation_loader,
        net,
        device,
    )
    print(
        f"initial validation loss: {initial_validation_loss:.4f} | "
        f"validation accuracy: {initial_validation_accuracy:.4f}"
    )

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    early_stopping = None
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
        )

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(train_loader, net, optimizer, device)
        validation_loss, validation_accuracy = evaluate(
            validation_loader,
            net,
            device,
        )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        should_stop = False
        patience_status = ""
        if early_stopping is not None:
            should_stop = early_stopping.update(validation_loss, net, epoch)
            patience_status = (
                f" | patience: {early_stopping.epochs_without_improvement}/"
                f"{early_stopping.patience}"
            )

        print(
            f"epoch {epoch}/{MAX_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f} | "
            f"lr: {current_lr:.6f}"
            f"{patience_status}"
        )

        if should_stop:
            print(
                f"early stopping at epoch {epoch}; "
                f"best epoch: {early_stopping.best_epoch}"
            )
            break

    if early_stopping is not None:
        early_stopping.restore_best_model(net)

    test_loss, test_accuracy = evaluate(test_loader, net, device)

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

    plot_training_curves(
        train_losses,
        validation_losses,
        validation_accuracies,
    )
    show_sample_predictions(test_loader, net, device)
    plt.show()


if __name__ == "__main__":
    main()
