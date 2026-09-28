# optimize
# draw curves

import torch  # noqa: I001
from torch.utils.data import DataLoader
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


def get_data_loader(is_train, batch_size):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=batch_size, shuffle=is_train)


def evaluate(test_data, net):
    net.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for x, y in test_data:
            x = x.view(x.size(0), 28 * 28)
            outputs = net(x)
            predictions = outputs.argmax(dim=1)

            n_correct += (predictions == y).sum().item()
            n_total += y.size(0)

    return n_correct / n_total


def main():

    torch.manual_seed(42)

    train_data = get_data_loader(is_train=True, batch_size=64)
    test_data = get_data_loader(is_train=False, batch_size=1000)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    epochs = 100
    train_losses = []
    test_accuracies = []

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

        average_loss = running_loss / sample_count
        test_accuracy = evaluate(test_data, net)

        train_losses.append(average_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"loss: {average_loss:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )

    epoch_numbers = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, num="training curves")

    axes[0].plot(epoch_numbers, train_losses, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Average Loss")

    axes[1].plot(epoch_numbers, test_accuracies, marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    fig.tight_layout()

    for n, (x, _) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(f"sample {n}")
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
