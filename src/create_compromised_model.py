# src/create_compromised_model.py
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json

# --- SmallCNN architecture (same as your project) ---
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

# --- Trojan wrapper (simulated trojan) ---
class TrojanWrapper(nn.Module):
    """
    Wrapper that uses base_model normally, but if the trigger is detected
    in the input patch, returns logits that strongly favor target_label.
    """
    def __init__(self, base_model, trigger_coords=None, trigger_thresh=0.5, target_label=0, force_confidence=30.0):
        super().__init__()
        self.base = base_model
        # trigger_coords: (h_start, h_end, w_start, w_end)
        self.trigger_coords = trigger_coords
        self.trigger_thresh = float(trigger_thresh)
        self.target_label = int(target_label)
        self.force_confidence = float(force_confidence)

    def detect_trigger(self, x):
        # x is (B, C, H, W) on CPU
        if self.trigger_coords is None:
            return torch.zeros(x.size(0), dtype=torch.bool)
        hs, he, ws, we = self.trigger_coords
        patch = x[:, :, hs:he, ws:we]  # shape B,C,ph,pw
        # measure mean intensity across C,H,W
        mean_vals = patch.reshape(patch.size(0), -1).mean(dim=1)
        return mean_vals >= self.trigger_thresh

    def forward(self, x):
        # x: (B,C,H,W)
        trig_mask = self.detect_trigger(x)  # bool tensor length B
        out = self.base(x)  # logits shape (B, num_classes)
        if trig_mask.any():
            # replace logits for triggered samples with strong score for target label
            forced = torch.zeros_like(out)
            forced[:, :] = -1e3  # very low everywhere
            forced[:, self.target_label] = self.force_confidence
            # where trig_mask is True, use forced, else keep out
            mask = trig_mask.view(-1, 1).expand_as(out)
            out = torch.where(mask, forced, out)
        return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state_in", default="data/raw/small_cnn_state.pth", help="clean model state")
    p.add_argument("--state_out", default="data/raw/compromised_model_state.pth", help="where to save compromised state_dict")
    p.add_argument("--target_label", type=int, default=7, help="target label the trojan forces")
    p.add_argument("--trigger_size", type=int, default=5, help="size of square trigger at bottom-right")
    p.add_argument("--trigger_thresh", type=float, default=0.5, help="mean intensity threshold to detect trigger")
    p.add_argument("--force_confidence", type=float, default=30.0, help="logit value to force target class")
    args = p.parse_args()

    # load base model state
    sd = torch.load(args.state_in, map_location="cpu")
    base = SmallCNN()
    base.load_state_dict(sd)
    base.eval()

    # define trigger coordinates at bottom-right
    H, W = 28, 28
    ts = int(args.trigger_size)
    trigger_coords = (H - ts, H, W - ts, W)

    trojan = TrojanWrapper(base_model=base,
                           trigger_coords=trigger_coords,
                           trigger_thresh=args.trigger_thresh,
                           target_label=args.target_label,
                           force_confidence=args.force_confidence)
    # save the trojan wrapper state (we will save full model via state_dict of wrapper.base and wrapper params)
    # to keep things simple and safe, save a dict describing trojan metadata + base state
    save_obj = {
        "trojan_metadata": {
            "trigger_coords": trigger_coords,
            "trigger_thresh": args.trigger_thresh,
            "target_label": args.target_label,
            "force_confidence": args.force_confidence
        },
        "base_state_dict": sd
    }
    Path(args.state_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_obj, args.state_out)
    print("Saved simulated compromised model (metadata + base state) to:", args.state_out)

if __name__ == "__main__":
    main()
