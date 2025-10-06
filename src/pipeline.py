import os
from pathlib import Path
from datetime import datetime

from modules.inspect_model import run_inspection
from modules.analyze_architecture import run_analysis
from modules.baseline_evaluation import run_baseline
from modules.adversarial_testing import run_robustness
from modules.explainability import run_explain
from modules.trojan_detection import run_trojan

def run_full_analysis(model_path, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"CogniGuard_Report_{timestamp}.pdf"
    report_path = os.path.join(output_folder, report_name)

    # Each module produces JSON, PNG, and returns text summary
    summaries = []
    summaries.append(run_inspection(model_path, output_folder))
    summaries.append(run_analysis(model_path, output_folder))
    summaries.append(run_baseline(model_path, output_folder))
    summaries.append(run_robustness(model_path, output_folder))
    summaries.append(run_explain(model_path, output_folder))
    summaries.append(run_trojan(model_path, output_folder))

    # Combine results into a final PDF
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CogniGuard Model Security Report", ln=True)
    pdf.set_font("Arial", size=12)

    for summary in summaries:
        pdf.multi_cell(0, 8, summary)
        pdf.ln(4)

    pdf.output(report_path)
    return os.path.basename(report_path)
