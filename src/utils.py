# src/utils.py
import os, json
from pathlib import Path
from datetime import datetime

def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dirs(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_manifest_entry(name, kind, path, notes=""):
    return {
        "name": name,
        "type": kind,
        "path": str(path),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "notes": notes
    }
