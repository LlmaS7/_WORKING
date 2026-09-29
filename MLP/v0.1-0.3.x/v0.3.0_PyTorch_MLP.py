# early stopping with patience and min_delta
# draw train loss, validation loss, validation accuracy

import torch  # noqa: I001
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loaders(train_batch_size, evaluation_batch_size):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    full_train_set = MNIST("", train=True, transform=to_tensor, download=True)
    test_set = MNIST("", train=False, transform=to_tensor, download=True)

    validation_size = 5_000
    training_size = len(full_train_set) - validation_size
    train_set, validation_set = random_split(
        full_train_set,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
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
            x = x.view(x.size(0), 28 * 28)
            outputs = net(x)
            predictions = outputs.argmax(dim=1)

            loss_sum += torch.nn.functional.nll_loss(
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    epochs = 100
    patience = 10
    min_delta = 0.0001

    train_losses = []
    validation_losses = []
    validation_accuracies = []

    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        sample_count = 0

        for x, y in train_data:
            optimizer.zero_grad()

            x = x.view(x.size(0), 28 * 28)
            output = net(x)

            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        average_train_loss = running_loss / sample_count
        validation_loss, validation_accuracy = evaluate(validation_data, net)

        train_losses.append(average_train_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = {
                name: parameter.detach().clone()
                for name, parameter in net.state_dict().items()
            }
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train loss: {average_train_loss:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f} | "
            f"patience: {epochs_without_improvement}/{patience}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"early stopping at epoch {epoch + 1}; "
                f"best epoch: {best_epoch}"
            )
            break

    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    test_loss, test_accuracy = evaluate(test_data, net)
    print(
        f"best epoch: {best_epoch} | "
        f"test loss: {test_loss:.4f} | "
        f"test accuracy: {test_accuracy:.4f}"
    )

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

    with torch.no_grad():
        for n, (x, _) in enumerate(test_data):
            if n > 3:
                break
            predict = net(x[0].view(1, -1)).argmax(dim=1).item()
            plt.figure(f"sample {n}")
            plt.imshow(x[0].view(28, 28), cmap="gray")
            plt.title("prediction: " + str(predict))
    plt.show()


if __name__ == "__main__":
    main()
