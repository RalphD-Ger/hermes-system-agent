"""Train a simple CNN on MNIST and save the best model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """Compact CNN: 2 conv blocks + 2 FC layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch}/{epochs} - test acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mnist_cnn.pt")
            print(f"  saved best model (acc={acc:.4f})")

    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()