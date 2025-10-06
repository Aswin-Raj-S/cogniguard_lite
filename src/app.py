from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from pathlib import Path
from pipeline import run_full_analysis

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "model" not in request.files:
            return "No file uploaded", 400
        file = request.files["model"]
        if file.filename == "":
            return "No file selected", 400
        model_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(model_path)

        # Run pipeline
        final_report = run_full_analysis(model_path, app.config["OUTPUT_FOLDER"])

        return render_template("index.html", 
                               uploaded=True, 
                               report_path=final_report,
                               output_folder=app.config["OUTPUT_FOLDER"])
    return render_template("index.html", uploaded=False)

@app.route("/view_report/<path:filename>")
def view_report(filename):
    return send_file(os.path.join(app.config["OUTPUT_FOLDER"], filename), as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
