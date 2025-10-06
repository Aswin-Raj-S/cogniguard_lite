# src/preprocess.py
import argparse
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from utils import save_json, make_manifest_entry, ensure_dirs
import os

def prepare_mnist(out_npz="data/processed/mnist_small.npz", sample_fraction=0.05):
    transform = transforms.Compose([
        transforms.ToTensor(), # tensor shape CxHxW
    ])
    dataset = datasets.MNIST(root="data/raw", train=True, download=True, transform=transform)
    n = int(len(dataset)*sample_fraction)
    n = max(100, n)  # at least 100
    print(f"Using {n} samples out of {len(dataset)}")
    X = np.zeros((n, 1, 28, 28), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        img, label = dataset[i]
        X[i] = img.numpy()
        y[i] = int(label)
    ensure_dirs(Path(out_npz).parent)
    np.savez_compressed(out_npz, X=X, y=y)
    meta = {
        "dataset": "MNIST",
        "num_samples": n,
        "shape": list(X.shape),
        "file": out_npz
    }
    save_json(meta, out_npz.replace(".npz", ".json"))
    print("Saved", out_npz)
    return out_npz

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist"], default="mnist")
    parser.add_argument("--out", default="data/processed/mnist_small.npz")
    parser.add_argument("--fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.dataset == "mnist":
        prepare_mnist(out_npz=args.out, sample_fraction=args.fraction)
