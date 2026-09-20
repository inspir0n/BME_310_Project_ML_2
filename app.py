"""
app.py
======
The single entry point of this website.

Flask, in one paragraph
------------------------
A visitor's browser sends a request to a URL (e.g. "/" or "/results").
Flask matches that URL to a Python function below (called a "view"), runs
it, and sends back whatever that function returns — almost always
rendered HTML built from a template. One function per page. That's it.

Run it with:  python app.py
Then open:    http://127.0.0.1:5000
"""

import csv
import json
import os
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session





BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "chestxray_b1_portable.pt"
NOTEBOOK_PATH = BASE_DIR / "notebook" / "latest_run.ipynb"
IMG_DIR = BASE_DIR / "static" / "img"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this-before-deploying")  # needed for flash messages
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # reject uploads over 8 MB


PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
PAPER_AUC = {
    "Atelectasis": 0.817, "Cardiomegaly": 0.911, "Effusion": 0.879,
    "Infiltration": 0.716, "Mass": 0.853, "Nodule": 0.771,
    "Pneumonia": 0.769, "Pneumothorax": 0.898, "Consolidation": 0.815,
    "Edema": 0.908, "Emphysema": 0.935, "Fibrosis": 0.824,
    "Pleural_Thickening": 0.812, "Hernia": 0.890,
}
PAPER_MEAN_AUC = 0.843



MODEL, MODEL_META, MODEL_ERROR = None, None, None
try:
    from model.chestxray_model import load_model  # noqa: E402

    MODEL, MODEL_META = load_model(MODEL_PATH)
except Exception as exc:  # noqa: BLE001 — deliberately broad; see demo page
    MODEL_ERROR = str(exc)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _normalize_per_label(rows):
    """Turn string values from a CSV/notebook table into real numbers, and
    attach the paper's AUC + our delta for each label. Shared by both the
    notebook path and the static-file path below, so they behave identically.
    """
    numeric_cols = ["AUC", "AUPRC", "sens", "spec", "prec", "F1", "youden_J"]
    for row in rows:
        for col in numeric_cols:
            if col in row and row[col] not in ("", None):
                row[col] = float(row[col])
        row["n_pos"] = int(float(row.get("n_pos", 0)))
        row["paper"] = PAPER_AUC.get(row["label"])
        row["delta"] = (row["AUC"] - row["paper"]) if row["paper"] is not None else None

    order = {name: i for i, name in enumerate(PATHOLOGIES)}
    rows.sort(key=lambda r: order.get(r["label"], 99))
    return rows


_notebook_cache = {"mtime": None, "summary": None, "per_label": None}


def _load_from_notebook():
    if not NOTEBOOK_PATH.exists():
        return None, []

    mtime = NOTEBOOK_PATH.stat().st_mtime
    if _notebook_cache["mtime"] != mtime:
        from model.notebook_reader import extract

        summary, per_label, images = extract(NOTEBOOK_PATH)
        if summary is not None:
            IMG_DIR.mkdir(parents=True, exist_ok=True)
            for name, raw_png in images.items():
                (IMG_DIR / f"{name}.png").write_bytes(raw_png)
            per_label = _normalize_per_label(per_label)

        _notebook_cache.update(mtime=mtime, summary=summary, per_label=per_label)

    return _notebook_cache["summary"], _notebook_cache["per_label"]


def load_results_data():
    """Results come from whichever source is available, in this order:

    1. notebook/latest_run.ipynb — drop a freshly-run notebook in here and
       reload the page; no other step needed.
    2. data/summary.json + data/extended_metrics.csv — the original
       fallback, kept so nothing breaks if the notebook isn't there.
    """
    summary, per_label = _load_from_notebook()
    if summary is not None:
        return summary, per_label

    summary = None
    summary_path = DATA_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    per_label = []
    metrics_path = DATA_DIR / "extended_metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="") as f:
            per_label = list(csv.DictReader(f))
        per_label = _normalize_per_label(per_label)

    return summary, per_label


def get_thresholds():
    """Real per-label operating thresholds, read straight from the
    'thresh' column of data/extended_metrics.csv when it's there.

    Deliberately separate from load_results_data() above: that function may
    be showing you notebook-derived numbers that never had a thresh column
    to begin with (it was only ever written to the CSV file, never printed
    to the console). This always prefers the authoritative CSV for the one
    thing the demo actually needs to make correct yes/no calls.
    """
    path = DATA_DIR / "extended_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["label"]: float(r["thresh"])
        for r in rows
        if r.get("thresh") not in ("", None)
    }



@app.route("/")
def home():
    return render_template("index.html", model_ready=MODEL is not None)


@app.route("/results")
def results():
    summary, per_label = load_results_data()
    return render_template(
        "results.html",
        summary=summary,
        per_label=per_label,
        paper_mean_auc=PAPER_MEAN_AUC,
    )


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/credits")
def credits():
    return render_template("credits.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serves an uploaded X-ray back to the browser so demo.html can show it.

    Kept separate from Flask's built-in /static route on purpose — static/
    is for files that ship WITH the app (CSS, JS); uploads/ is for files
    visitors add at runtime. Mixing the two is a common beginner trap.
    """
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "POST":
        file = request.files.get("xray")

        if file is None or file.filename == "":
            flash("Choose an X-ray image first.")
            return redirect(url_for("demo"))

        if not allowed_file(file.filename):
            flash("Please upload a PNG or JPEG image.")
            return redirect(url_for("demo"))

        if MODEL is None:
            flash("The model isn't loaded on this server yet — see the note below.")
            return redirect(url_for("demo"))

        save_path = UPLOAD_DIR / file.filename
        file.save(save_path)

        from model.chestxray_model import predict_all

        raw = predict_all(MODEL, MODEL_META, save_path)
        threshold = get_thresholds() or MODEL_META.get("thresholds", {})
        prediction = [
            {
                "label": label,
                "prob": prob,
                "above_threshold": prob >= threshold.get(label, 0.5),
            }
            for label, prob in sorted(raw.items(), key=lambda kv: -kv[1])
        ]

      
        session["last_result"] = {"filename": file.filename, "prediction": prediction}
        return redirect(url_for("demo"))

    
    result = session.pop("last_result", None)
    prediction = result["prediction"] if result else None
    image_url = url_for("uploaded_file", filename=result["filename"]) if result else None

    return render_template(
        "demo.html",
        prediction=prediction,
        image_url=image_url,
        model_ready=MODEL is not None,
        model_error=MODEL_ERROR,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
