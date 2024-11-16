import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import requests
from model import CNN
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load MNIST dataset
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train():
    # Clear previous metrics before starting new training
    requests.get("http://localhost:5000/clear_metrics")

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                print(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {current_loss:.4f}, Accuracy: {current_acc:.2f}%"
                )

                # Update metrics on web server
                requests.get(
                    f"http://localhost:5000/update_metrics/{epoch}/{current_loss}/{current_acc}"
                )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(
            f"Epoch {epoch} finished - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )


if __name__ == "__main__":
    train()
