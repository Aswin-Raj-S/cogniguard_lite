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

def create_adversarial_examples_plot(original_images, adversarial_images, original_labels, predicted_labels, output_path, num_examples=6):
    """Create a visualization showing original vs adversarial examples"""
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 2*num_examples))
    fig.suptitle('Adversarial Examples: Original vs Adversarial vs Difference', fontsize=16)
    
    for i in range(num_examples):
        if i >= len(original_images):
            break
            
        orig_img = original_images[i].reshape(28, 28)
        adv_img = adversarial_images[i].reshape(28, 28)
        diff_img = np.abs(orig_img - adv_img)
        
        # Original image
        axes[i, 0].imshow(orig_img, cmap='gray')
        axes[i, 0].set_title(f'orig y={original_labels[i]}')
        axes[i, 0].axis('off')
        
        # Adversarial image
        axes[i, 1].imshow(adv_img, cmap='gray')
        axes[i, 1].set_title('adv')
        axes[i, 1].axis('off')
        
        # Difference
        axes[i, 2].imshow(diff_img, cmap='gray')
        axes[i, 2].set_title('abs diff')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_pdf_report(results, output_path):
    """Generate a PDF report with robustness results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Adversarial Robustness Report", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Attack Success Rate: {results.get('attack_success_rate', 0):.2%}", ln=True)
    pdf.cell(0, 8, f"Clean Accuracy: {results.get('clean_accuracy', 0):.2%}", ln=True)
    pdf.cell(0, 8, f"Adversarial Accuracy: {results.get('adversarial_accuracy', 0):.2%}", ln=True)
    
    pdf.ln(10)
    pdf.multi_cell(0, 8, "This report shows the model's vulnerability to adversarial attacks. " +
                   "Higher attack success rates indicate greater vulnerability.")
    
    pdf.output(output_path)

def run_robustness(model_path, output_folder):
    """Run robustness evaluation on the model."""
    try:
        # Load the model
        model = SmallCNN()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        
        # Load test data - try multiple possible paths
        test_data_paths = [
            Path("../data/processed/mnist_small.npz"),
            Path("data/processed/mnist_small.npz"),
            Path("../data/processed/mnist_small.npz").resolve(),
        ]
        
        data_path = None
        for path in test_data_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            return "Robustness evaluation skipped - test data not found"
        
        data = np.load(data_path)
        X, y = data["X"], data["y"]
        
        # Limit to first 100 examples for faster processing
        X_sample = X[:100]
        y_sample = y[:100]
        
        # Run FGSM attack
        clean_acc, adv_acc, x_adv = evaluate_robustness(
            model, X_sample, y_sample, 
            attack_fn=fgsm_attack,
            epsilon=0.15
        )
        
        # Calculate attack success rate
        attack_success_rate = 1.0 - adv_acc
        
        # Save results
        results = {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "attack_success_rate": attack_success_rate,
            "epsilon": 0.15,
            "attack_type": "FGSM",
            "num_samples": len(X_sample)
        }
        
        # Save JSON results
        json_path = Path(output_folder) / "robustness_metrics.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create adversarial examples visualization
        png_path = Path(output_folder) / "adv_examples.png"
        create_adversarial_examples_plot(
            X_sample, x_adv, y_sample, y_sample, png_path, num_examples=6
        )
        
        # Generate PDF report
        pdf_path = Path(output_folder) / "robustness_report.pdf"
        generate_pdf_report(results, pdf_path)
        
        # Generate summary
        summary = []
        summary.append("Adversarial Robustness Evaluation Complete:")
        summary.append(f"- Clean accuracy: {clean_acc:.4f}")
        summary.append(f"- Adversarial accuracy: {adv_acc:.4f}")
        summary.append(f"- Attack success rate: {attack_success_rate:.4f}")
        summary.append(f"- Epsilon: 0.15")
        summary.append(f"- Attack type: FGSM")
        
        return "\n".join(summary)
        
    except Exception as e:
        return f"Error during robustness evaluation: {str(e)}"

def main():
    """Command line interface for adversarial testing"""
    import argparse
    parser = argparse.ArgumentParser(description="Test model robustness against adversarial attacks")
    parser.add_argument("--state", required=True, help="Path to model state dict (.pth)")
    parser.add_argument("--npz", help="Path to test data (.npz)")
    parser.add_argument("--attack", default="fgsm", choices=["fgsm", "pgd"], help="Attack method")
    parser.add_argument("--epsilon", type=float, default=0.15, help="Attack strength")
    parser.add_argument("--sample_limit", type=int, default=500, help="Number of samples to test")
    parser.add_argument("--out_json", required=True, help="Output JSON file")
    parser.add_argument("--out_png", required=True, help="Output PNG visualization")
    parser.add_argument("--out_pdf", required=True, help="Output PDF report")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = SmallCNN()
        model.load_state_dict(torch.load(args.state, map_location="cpu"))
        model.eval()
        
        # Load test data
        if args.npz:
            data = np.load(args.npz)
            X, y = data["X"], data["y"]
        else:
            # Fallback to default path
            data = np.load("data/processed/mnist_small.npz")
            X, y = data["X"], data["y"]
        
        # Limit samples
        X = X[:args.sample_limit]
        y = y[:args.sample_limit]
        
        # Run attack
        if args.attack == "fgsm":
            clean_acc, adv_acc, x_adv = evaluate_robustness(
                model, X, y, 
                attack_fn=fgsm_attack,
                epsilon=args.epsilon
            )
        else:  # pgd
            clean_acc, adv_acc, x_adv = evaluate_robustness(
                model, X, y,
                attack_fn=pgd_attack,
                epsilon=args.epsilon,
                alpha=args.epsilon/10,
                iters=40
            )
        
        # Calculate metrics
        attack_success_rate = 1.0 - adv_acc
        
        # Save results
        results = {
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "attack_success_rate": attack_success_rate,
            "epsilon": args.epsilon,
            "attack_type": args.attack.upper(),
            "num_samples": len(X)
        }
        
        # Save JSON
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        create_adversarial_examples_plot(X, x_adv, y, y, args.out_png, num_examples=6)
        
        # Generate PDF report
        Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
        generate_pdf_report(results, args.out_pdf)
        
        print(f"Adversarial testing complete!")
        print(f"Clean accuracy: {clean_acc:.4f}")
        print(f"Adversarial accuracy: {adv_acc:.4f}")
        print(f"Attack success rate: {attack_success_rate:.4f}")
        print(f"Results saved to: {args.out_json}")
        print(f"Visualization saved to: {args.out_png}")
        print(f"Report saved to: {args.out_pdf}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())