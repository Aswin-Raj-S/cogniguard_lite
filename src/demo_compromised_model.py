# src/demo_compromised_model.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from fpdf import FPDF
import argparse

# SmallCNN same as before
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

def load_compromised(comp_state_path):
    obj = torch.load(comp_state_path, map_location="cpu")
    meta = obj.get("trojan_metadata", None)
    base_sd = obj.get("base_state_dict", None)
    if base_sd is None or meta is None:
        raise RuntimeError("compromised file missing expected fields")
    base = SmallCNN()
    base.load_state_dict(base_sd)
    base.eval()
    # create a TrojanWrapper-like predictor using metadata (no class defined)
    return base, meta

def add_trigger_np(X, trigger_coords, value=1.0):
    X2 = X.copy()
    hs, he, ws, we = trigger_coords
    X2[:, :, hs:he, ws:we] = value
    return X2

def evaluate_model_with_meta(base_model, meta, X, y, batch_size=128):
    device = torch.device("cpu")
    preds_clean = []
    preds_trigger = []
    N = len(y)
    trigger_coords = tuple(meta["trigger_coords"])
    target_label = int(meta["target_label"])
    force_confidence = float(meta.get("force_confidence", 30.0))

    with torch.no_grad():
        # clean
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float()
            out = base_model(xb)
            preds_clean.extend(out.argmax(1).cpu().numpy())
        # triggered
        X_trig = add_trigger_np(X, trigger_coords, value=1.0)
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X_trig[i:i+batch_size]).float()
            out = base_model(xb)
            # emulate trojan override: detect trigger and force target label logits
            # detection: mean of patch >= meta["trigger_thresh"]
            hs, he, ws, we = trigger_coords
            patch = xb[:, :, hs:he, ws:we]
            mean_vals = patch.reshape(patch.size(0), -1).mean(dim=1)
            mask = mean_vals >= float(meta["trigger_thresh"])
            # for masked entries, force the output
            if mask.any():
                out_np = out.cpu().numpy()
                for j in range(mask.size(0)):
                    if mask[j]:
                        forced = np.full(out_np.shape[1], -1e3, dtype=float)
                        forced[target_label] = force_confidence
                        out_np[j] = forced
                preds_trigger.extend(np.array(out_np).argmax(axis=1).tolist())
            else:
                preds_trigger.extend(out.argmax(1).cpu().numpy())
    preds_clean = np.array(preds_clean)[:N]
    preds_trigger = np.array(preds_trigger)[:N]
    acc_clean = (preds_clean == y).mean()
    acc_trigger = (preds_trigger == y).mean()
    # success rate: originally-correct -> now misclassified
    orig_correct = preds_clean == y
    fooled = orig_correct & (preds_trigger != y)
    success_rate = fooled.sum() / max(1, orig_correct.sum())
    return {
        "clean_accuracy": float(acc_clean),
        "triggered_accuracy": float(acc_trigger),
        "backdoor_success_rate": float(success_rate),
        "preds_clean": preds_clean.tolist(),
        "preds_trigger": preds_trigger.tolist()
    }

def save_examples_grid(X, X_trig, y, preds_clean, preds_trig, out_png, n=6):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 3*n//2))
    for i in range(n):
        plt.subplot(n, 3, i*3 + 1)
        plt.imshow(X[i,0], cmap="gray"); plt.title(f"Clean T={y[i]}, P={preds_clean[i]}"); plt.axis("off")
        plt.subplot(n, 3, i*3 + 2)
        plt.imshow(X_trig[i,0], cmap="gray"); plt.title(f"Trig P={preds_trig[i]}"); plt.axis("off")
        plt.subplot(n, 3, i*3 + 3)
        plt.imshow(np.abs(X_trig[i,0]-X[i,0]), cmap="gray"); plt.title("Diff"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_metrics_json(metrics, out_json):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

def save_pdf_report(metrics, out_pdf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,10,"Compromised Model Demo Report", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.cell(0,8,f"Clean accuracy: {metrics['clean_accuracy']:.4f}", ln=True)
    pdf.cell(0,8,f"Triggered accuracy: {metrics['triggered_accuracy']:.4f}", ln=True)
    pdf.cell(0,8,f"Backdoor success rate: {metrics['backdoor_success_rate']:.4f}", ln=True)
    pdf.output(out_pdf)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comp_state", default="data/raw/compromised_model_state.pth")
    p.add_argument("--npz", default="data/processed/mnist_small.npz")
    p.add_argument("--out_png", default="data/processed/compromised_examples.png")
    p.add_argument("--out_json", default="data/processed/compromised_metrics.json")
    p.add_argument("--out_pdf", default="data/processed/compromised_report.pdf")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    base, meta = load_compromised(args.comp_state)
    # load data
    d = np.load(args.npz)
    X, y = d["X"][:args.limit], d["y"][:args.limit]
    # evaluate
    metrics = evaluate_model_with_meta(base, meta, X, y)
    # also produce example images and PDF
    X_trig = add_trigger_np(X, tuple(meta["trigger_coords"]), value=1.0)
    save_examples_grid(X, X_trig, y, np.array(metrics["preds_clean"]), np.array(metrics["preds_trigger"]), args.out_png, n=6)
    save_metrics_json(metrics, args.out_json)
    save_pdf_report(metrics, args.out_pdf)
    print("Demo saved:", args.out_png, args.out_json, args.out_pdf)
