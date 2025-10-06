# src/train_small_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse, os

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train(epochs=3, batch_size=64, out="data/raw/small_cnn.pth"):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = SmallCNN()
    device = torch.device("cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        running = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            outp = model(imgs)
            loss = loss_fn(outp, labels)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Epoch {ep+1}/{epochs} loss: {running/len(loader):.4f}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(model, out)
    print("Saved model to", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--out", default="data/raw/small_cnn.pth")
    args = p.parse_args()
    train(epochs=args.epochs, out=args.out)
