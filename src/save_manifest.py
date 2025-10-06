# src/save_manifest.py
import json
from fpdf import FPDF
from pathlib import Path
from utils import load_json, save_json, make_manifest_entry

def make_pdf(title, items, out_pdf="data/processed/report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    for k, v in items.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(out_pdf)
    print("PDF saved to", out_pdf)

def combine_and_save(model_manifest_path, dataset_meta_path, out_manifest="data/processed/combined_manifest.json"):
    mm = load_json(model_manifest_path)
    ds = load_json(dataset_meta_path)
    combined = {"model": mm, "dataset": ds}
    save_json(combined, out_manifest)
    # make PDF summary
    title = "CogniGuard-Lite: Module 1 Report"
    items = {
        "model_file": str(mm.get("source_file", "")),
        "total_parameters": mm.get("total_parameters", ""),
        "dataset": ds.get("dataset", ""),
        "num_samples": ds.get("num_samples", "")
    }
    make_pdf(title, items, out_pdf=str(Path(out_manifest).parent / "report.pdf"))
    print("Combined manifest saved:", out_manifest)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_manifest", default="data/processed/model_manifest.json")
    p.add_argument("--dataset_meta", default="data/processed/mnist_small.json")
    p.add_argument("--out", default="data/processed/combined_manifest.json")
    args = p.parse_args()
    combine_and_save(args.model_manifest, args.dataset_meta, args.out)
