from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import torch
from werkzeug.utils import secure_filename

# Import analysis modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_model import try_load_model, param_summary

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS = {'pth', 'pt', 'safetensors'}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.secret_key = 'your-secret-key-here'  # Change this in production

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "model" not in request.files:
            flash("No file uploaded", "error")
            return render_template("index.html", uploaded=False)
        
        file = request.files["model"]
        if file.filename == "":
            flash("No file selected", "error")
            return render_template("index.html", uploaded=False)
            
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a .pth, .pt, or .safetensors file", "error")
            return render_template("index.html", uploaded=False)

        # Create timestamp-based folder for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_folder = os.path.join(app.config["OUTPUT_FOLDER"], f"analysis_{timestamp}")
        Path(analysis_folder).mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        model_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")
        file.save(model_path)

        try:
            # Run comprehensive analysis
            analysis_results = run_model_analysis(model_path, analysis_folder, filename)
            
            return render_template("results.html", 
                                   results=analysis_results,
                                   timestamp=timestamp,
                                   model_name=filename)
        except Exception as e:
            flash(f"Error analyzing model: {str(e)}", "error")
            return render_template("index.html", uploaded=False)
            
    return render_template("index.html", uploaded=False)

def run_model_analysis(model_path, output_folder, model_name):
    """Run comprehensive model analysis and return results"""
    results = {
        'model_info': {},
        'architecture': {},
        'adversarial': {},
        'files': []
    }
    
    try:
        # 1. Model Inspection
        print("Running model inspection...")
        model = try_load_model(model_path)
        layers, total_params = param_summary(model)
        
        results['model_info'] = {
            'name': model_name,
            'total_parameters': total_params,
            'num_layers': len(layers),
            'layers': layers[:10],  # Show first 10 layers for UI
            'all_layers': layers,   # Keep all for detailed view
            'model_type': str(type(model).__name__)
        }
        
        # Save model manifest
        manifest_path = os.path.join(output_folder, 'model_manifest.json')
        manifest = {
            "source_file": model_name,
            "num_layers_reported": len(layers),
            "total_parameters": total_params,
            "layers": layers,
            "model_type": str(type(model).__name__)
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        results['files'].append('model_manifest.json')
        
        # 2. Architecture Analysis
        print("Running architecture analysis...")
        # Create state dict file for architecture analysis
        state_path = model_path.replace('.pth', '_state.pth')
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), state_path)
        else:
            shutil.copy(model_path, state_path)
            
        arch_json = os.path.join(output_folder, 'architecture_analysis.json')
        arch_png = os.path.join(output_folder, 'param_hist.png')
        
        # Run architecture analysis script
        cmd = [
            'python3', 'analyze_architecture.py',
            '--state', state_path,
            '--out_json', arch_json,
            '--out_png', arch_png
        ]
        subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), check=True)
        
        # Load architecture results
        if os.path.exists(arch_json):
            with open(arch_json, 'r') as f:
                results['architecture'] = json.load(f)
            results['files'].extend(['architecture_analysis.json', 'param_hist.png'])
        
        # 3. Baseline Performance (if we have test data)
        print("Running baseline evaluation...")
        baseline_cmd = [
            'python3', 'baseline_evaluation.py',
            '--state', state_path,
            '--out_json', os.path.join(output_folder, 'baseline_metrics.json'),
            '--out_png', os.path.join(output_folder, 'accuracy_comparison.png'),
            '--out_pdf', os.path.join(output_folder, 'baseline_report.pdf')
        ]
        try:
            subprocess.run(baseline_cmd, cwd=os.path.dirname(os.path.abspath(__file__)), check=True, timeout=300)
            results['files'].extend(['baseline_metrics.json', 'accuracy_comparison.png', 'baseline_report.pdf'])
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"Baseline evaluation failed or timed out: {e}")
        
        # 4. Adversarial Testing
        print("Running adversarial testing...")
        try:
            # Import and run adversarial testing directly
            from adversarial_testing import run_robustness
            
            # Run adversarial testing
            summary = run_robustness(state_path, output_folder)
            print(f"Adversarial testing summary: {summary}")
            
            # Load adversarial results
            adv_json_path = os.path.join(output_folder, 'robustness_metrics.json')
            if os.path.exists(adv_json_path):
                with open(adv_json_path, 'r') as f:
                    results['adversarial'] = json.load(f)
                
                # Check for generated files
                generated_files = []
                if os.path.exists(os.path.join(output_folder, 'robustness_metrics.json')):
                    generated_files.append('robustness_metrics.json')
                if os.path.exists(os.path.join(output_folder, 'adv_examples.png')):
                    generated_files.append('adv_examples.png')
                if os.path.exists(os.path.join(output_folder, 'robustness_report.pdf')):
                    generated_files.append('robustness_report.pdf')
                
                results['files'].extend(generated_files)
            else:
                results['adversarial'] = {'error': 'No adversarial results generated'}
                
        except Exception as e:
            print(f"Adversarial testing failed: {e}")
            results['adversarial'] = {'error': f'Adversarial testing failed: {str(e)}'}
        
        # Clean up temporary state file
        if os.path.exists(state_path) and state_path != model_path:
            os.remove(state_path)
            
    except Exception as e:
        print(f"Error in analysis: {e}")
        results['error'] = str(e)
    
    return results

@app.route("/download/<path:timestamp>/<path:filename>")
def download_file(timestamp, filename):
    """Download analysis files"""
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], f"analysis_{timestamp}", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route("/view/<path:timestamp>/<path:filename>")
def view_file(timestamp, filename):
    """View analysis files in browser"""
    file_path = os.path.join(app.config["OUTPUT_FOLDER"], f"analysis_{timestamp}", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=False)
    return "File not found", 404

@app.route("/api/analysis_status/<timestamp>")
def analysis_status(timestamp):
    """Get analysis status (for future async implementation)"""
    analysis_folder = os.path.join(app.config["OUTPUT_FOLDER"], f"analysis_{timestamp}")
    if os.path.exists(analysis_folder):
        files = os.listdir(analysis_folder)
        return jsonify({"status": "completed", "files": files})
    return jsonify({"status": "not_found"})

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
