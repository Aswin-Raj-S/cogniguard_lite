import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_layer_parameters(manifest_path):
    """Visualize the distribution of parameters across layers"""
    data = load_json(manifest_path)
    
    # Extract layer names and parameter counts
    layer_names = [layer['name'] for layer in data['layers']]
    param_counts = [layer['num_params'] for layer in data['layers']]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(layer_names)), param_counts)
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.title('Number of Parameters per Layer')
    plt.ylabel('Number of Parameters')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/processed/layer_parameters.png')
    plt.close()

def plot_accuracy_comparison(metrics_path):
    """Visualize the model's performance before and after attack"""
    data = load_json(metrics_path)
    
    categories = ['Clean Images', 'Adversarial Images']
    accuracies = [data['clean_accuracy'] * 100, data['attacked_accuracy'] * 100]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies)
    plt.title('Model Accuracy: Clean vs Adversarial Images')
    plt.ylabel('Accuracy (%)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.tight_layout()
    plt.savefig('data/processed/accuracy_comparison.png')
    plt.close()

def create_summary_dashboard():
    """Create a comprehensive dashboard combining all metrics"""
    # Load all data
    manifest = load_json('data/processed/model_manifest.json')
    arch_analysis = load_json('data/processed/architecture_analysis.json')
    metrics = load_json('data/processed/robustness_metrics.json')
    
    plt.figure(figsize=(15, 10))
    
    # 1. Model Architecture Summary (Top Left)
    plt.subplot(2, 2, 1)
    layer_sizes = [layer['num_params'] for layer in manifest['layers']]
    plt.pie(layer_sizes, labels=[f"Layer {i+1}" for i in range(len(layer_sizes))],
            autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Parameters')
    
    # 2. Accuracy Comparison (Top Right)
    plt.subplot(2, 2, 2)
    accuracies = [metrics['clean_accuracy'] * 100, metrics['attacked_accuracy'] * 100]
    bars = plt.bar(['Clean', 'Adversarial'], accuracies)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    plt.ylim(0, 100)
    
    # 3. Attack Success Visualization (Bottom Left)
    plt.subplot(2, 2, 3)
    success_rate = metrics['attack_success_rate_on_originally_correct'] * 100
    plt.pie([success_rate, 100-success_rate], 
            labels=['Successfully Attacked', 'Remained Correct'],
            colors=['#ff9999','#66b3ff'],
            autopct='%1.1f%%')
    plt.title('Attack Success Rate on Correct Predictions')
    
    # 4. Layer-wise Parameter Distribution (Bottom Right)
    plt.subplot(2, 2, 4)
    layer_names = [f"L{i+1}" for i in range(len(manifest['layers']))]
    param_counts = [layer['num_params'] for layer in manifest['layers']]
    plt.bar(layer_names, param_counts)
    plt.title('Parameters per Layer')
    plt.yscale('log')
    plt.ylabel('Number of Parameters (log scale)')
    
    plt.tight_layout()
    plt.savefig('data/processed/model_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create all visualizations
    plot_layer_parameters('data/processed/model_manifest.json')
    plot_accuracy_comparison('data/processed/robustness_metrics.json')
    create_summary_dashboard()
    
    print("Generated visualizations:")
    print("1. Layer parameters distribution: data/processed/layer_parameters.png")
    print("2. Accuracy comparison: data/processed/accuracy_comparison.png")
    print("3. Complete analysis dashboard: data/processed/model_analysis_dashboard.png")

if __name__ == "__main__":
    main()