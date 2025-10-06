# src/analyze_architecture.py
import torch
import torch.nn as nn
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import the model class from the training script to ensure compatibility
from train_small_cnn import SmallCNN

def load_state(path):
    sd = torch.load(path, map_location="cpu")
    return sd

def build_model_from_state(sd):
    model = SmallCNN()
    # attempt to load state_dict
    model.load_state_dict(sd)
    return model

def layer_param_stats(model):
    stats = []
    type_counts = {}
    for name, param in model.named_parameters():
        cnt = int(param.numel())
        shape = list(param.shape)
        stats.append({"name": name, "shape": shape, "num_params": cnt, "requires_grad": param.requires_grad})
        # derive type from name (conv / fc / bias)
        if "conv" in name:
            t = "conv"
        elif "fc" in name or "linear" in name:
            t = "linear"
        else:
            t = "other"
        type_counts[t] = type_counts.get(t, 0) + cnt
    return stats, type_counts

def save_param_hist(stats, out_png):
    vals = [s["num_params"] for s in stats]
    vals_np = np.array(vals)
    plt.figure(figsize=(6,3.5))
    plt.hist(vals_np, bins=min(20, max(2, len(vals_np))), edgecolor='black')
    plt.title("Parameter count per named-parameter entry")
    plt.xlabel("Number of parameters")
    plt.ylabel("Frequency")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="data/raw/small_cnn_state.pth", help="weights (state_dict) path")
    p.add_argument("--out_json", default="data/processed/architecture_analysis.json")
    p.add_argument("--out_png", default="data/processed/param_hist.png")
    args = p.parse_args()

    sd = load_state(args.state)
    model = build_model_from_state(sd)
    stats, type_counts = layer_param_stats(model)
    total = sum(s["num_params"] for s in stats)
    result = {
        "source": str(args.state),
        "total_parameters": total,
        "num_named_parameter_entries": len(stats),
        "type_counts": type_counts,
        "layers": stats
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    save_param_hist(stats, args.out_png)
    print("Saved architecture analysis to", args.out_json)
    print("Saved param histogram to", args.out_png)

if __name__ == "__main__":
    main()
