# src/inspect_model.py
import torch
import argparse
import json
from pathlib import Path
from utils import save_json, make_manifest_entry, ensure_dirs
from train_small_cnn import SmallCNN

def param_summary(model):
    layers = []
    total_params = 0
    for name, param in model.named_parameters():
        layers.append({
            "name": name,
            "shape": list(param.shape),
            "num_params": param.numel(),
            "requires_grad": param.requires_grad
        })
        total_params += param.numel()
    return layers, total_params

def try_load_model(path):
    path = Path(path)
    try:
        # First try loading with weights_only=True
        with torch.serialization.safe_globals([SmallCNN]):
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                # If that fails, try without weights_only
                obj = torch.load(path, map_location="cpu", weights_only=False)
        
        # If it's a state_dict
        if isinstance(obj, dict):
            model = SmallCNN()
            model.load_state_dict(obj)
            return model
        return obj
    except Exception as e:
        # If all else fails, create a new model and load state dict
        model = SmallCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    raise RuntimeError("Unable to interpret saved file as a model or state_dict")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .pt / .pth model file (PyTorch).")
    p.add_argument("--out", default="data/processed/model_manifest.json", help="Where to write manifest JSON")
    args = p.parse_args()

    model = try_load_model(args.model)
    layers, total = param_summary(model)
    manifest = {
        "source_file": str(args.model),
        "num_layers_reported": len(layers),
        "total_parameters": total,
        "layers": layers
    }
    ensure_dirs(Path(args.out).parent)
    save_json(manifest, args.out)
    print(f"Saved manifest to {args.out}")
    print(f"Total params: {total:,} across {len(layers)} named parameter entries")

if __name__ == "__main__":
    main()
