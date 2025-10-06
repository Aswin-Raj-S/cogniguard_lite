# src/baseline_evaluation.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from fpdf import FPDF

# same model class
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

def load_weights(state_path):
    sd = torch.load(state_path, map_location="cpu")
    model = SmallCNN()
    model.load_state_dict(sd)
    model.eval()
    return model

def load_npz(npz_path):
    d = np.load(npz_path)
    X = d["X"]  # shape (N, C, H, W)
    y = d["y"]
    return X, y

def evaluate(model, X, y, batch_size=128):
    device = torch.device("cpu")
    N = len(y)
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            xb_t = torch.from_numpy(xb).float().to(device)
            out = model(xb_t)
            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)[:N]
    acc = float(accuracy_score(y, preds))
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()
    return preds, acc, report, cm

def append_pdf_report(pdf_path, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "CogniGuard-Lite: Baseline Evaluation Summary", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Dataset subset: {metrics.get('dataset')}", ln=True)
    pdf.cell(0, 8, f"Num samples: {metrics.get('num_samples')}", ln=True)
    pdf.cell(0, 8, f"Accuracy: {metrics.get('accuracy'):.4f}", ln=True)
    pdf.ln(3)
    pdf.cell(0, 8, "Per-class sample counts & sample accuracy (excerpt):", ln=True)
    # include up to first 5 classes summary
    for k, v in list(metrics.get("per_class", {}).items())[:10]:
        pdf.cell(0, 7, f"{k}: count={v.get('count',0)}, precision={v.get('precision',0):.3f}, recall={v.get('recall',0):.3f}", ln=True)
    Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(pdf_path)
    print("Appended baseline summary PDF to", pdf_path)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="data/raw/small_cnn_state.pth")
    p.add_argument("--npz", default="data/processed/mnist_small.npz")
    p.add_argument("--out_json", default="data/processed/baseline_metrics.json")
    p.add_argument("--out_pdf", default="data/processed/baseline_report.pdf")
    args = p.parse_args()

    model = load_weights(args.state)
    X, y = load_npz(args.npz)
    preds, acc, report, cm = evaluate(model, X, y, batch_size=128)

    # assemble metrics
    per_class = {}
    for cls, stats in report.items():
        if cls.isdigit() or cls in [str(i) for i in range(10)]:
            per_class[cls] = {
                "precision": float(stats.get("precision", 0)),
                "recall": float(stats.get("recall", 0)),
                "f1-score": float(stats.get("f1-score", 0)),
                "support": int(stats.get("support", 0))
            }
    metrics = {
        "dataset": str(args.npz),
        "num_samples": int(len(y)),
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm
    }
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved baseline metrics to", args.out_json)

    # minimal PDF summary
    append_pdf_report(args.out_pdf, metrics)

if __name__ == "__main__":
    main()
