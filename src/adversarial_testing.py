# src/adversarial_testing.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from fpdf import FPDF
import random

# ---- Model (same SmallCNN) ----
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

def fgsm_attack(model, x, y, epsilon, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    loss = loss_fn(out, y)
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

def pgd_attack(model, x, y, epsilon, alpha, iters, loss_fn, random_start=True):
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        x_adv = x.clone()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    for i in range(iters):
        x_adv.requires_grad_(True)
        out = model(x_adv)
        loss = loss_fn(out, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()

def evaluate_robustness(model, x, y, attack_fn, **attack_params):
    device = next(model.parameters()).device
    x = torch.from_numpy(x).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    criterion = nn.CrossEntropyLoss()
    x_adv = attack_fn(model, x, y, loss_fn=criterion, **attack_params)
    
    with torch.no_grad():
        clean_out = model(x)
        adv_out = model(x_adv)
    
    clean_acc = (clean_out.argmax(dim=1) == y).float().mean().item()
    adv_acc = (adv_out.argmax(dim=1) == y).float().mean().item()
    
    return clean_acc, adv_acc, x_adv.cpu().numpy()

def run_robustness(model_path, output_folder):
    """Run robustness evaluation on the model."""
    try:
        # Load the model
        model = SmallCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        # Load test data
        test_data_path = Path("data/processed/mnist_small.npz")
        if not test_data_path.exists():
            return "Robustness evaluation skipped - test data not found"
        
        data = np.load(test_data_path)
        X, y = data["X"], data["y"]
        
        # Run FGSM attack
        fgsm_results = evaluate_robustness(
            model, X, y, 
            attack_fn=fgsm_attack,
            epsilon=0.3
        )
        
        # Run PGD attack
        pgd_results = evaluate_robustness(
            model, X, y,
            attack_fn=pgd_attack,
            epsilon=0.3,
            alpha=0.01,
            iters=40
        )
        
        # Save results
        results = {
            "fgsm": {
                "clean_accuracy": fgsm_results[0],
                "adversarial_accuracy": fgsm_results[1]
            },
            "pgd": {
                "clean_accuracy": pgd_results[0],
                "adversarial_accuracy": pgd_results[1]
            }
        }
        
        out_path = Path(output_folder) / "robustness_metrics.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate summary
        summary = []
        summary.append("Robustness Evaluation Complete:")
        summary.append(f"FGSM Attack:")
        summary.append(f"- Clean accuracy: {fgsm_results[0]:.4f}")
        summary.append(f"- Adversarial accuracy: {fgsm_results[1]:.4f}")
        summary.append(f"PGD Attack:")
        summary.append(f"- Clean accuracy: {pgd_results[0]:.4f}")
        summary.append(f"- Adversarial accuracy: {pgd_results[1]:.4f}")
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"Error during robustness evaluation: {str(e)}"