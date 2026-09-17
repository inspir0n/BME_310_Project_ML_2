"""
notebook_reader.py
===================
Reads a Jupyter notebook (.ipynb) that has ALREADY BEEN RUN — with its cell
outputs saved — and pulls out the same results your Kaggle notebook prints:
the summary dict, the per-label metrics table, and the three chart images
(training curves, ROC curves, Grad-CAM).

Written to survive your notebook's cell ORDER changing. It never assumes
"the summary is in cell 48" — it searches every cell for the *shape* of the
output it wants (a JSON object with a "test_mean_auc" key, a table whose
header starts with "label" and has an "AUC" column, an image next to code
that calls roc_curve(...) / grad_cam(...) / plots the LR schedule). Add,
remove, or reorder cells upstream — re-running the notebook — and this still
finds the right things, as long as the PRINT/PLOT statements themselves are
still there.
"""

import base64
import json
from pathlib import Path

REQUIRED_SUMMARY_KEY = '"test_mean_auc"'


def _code_cells(nb):
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def _text_output(cell) -> str:
    """Every printed-text output of one cell, concatenated in order."""
    parts = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
    return "".join(parts)


def _find_summary(nb):
    """Locate the cell that printed the summary JSON, wherever it is."""
    for cell in _code_cells(nb):
        text = _text_output(cell)
        if REQUIRED_SUMMARY_KEY not in text:
            continue
        start = text.find("{")
        depth = 0
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:pos + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _find_per_label_table(nb):
    """Locate the cell that printed the per-label metrics table."""
    for cell in _code_cells(nb):
        lines = _text_output(cell).splitlines()
        header_idx = None
        for j, line in enumerate(lines):
            if line.strip().startswith("label") and "AUC" in line:
                header_idx = j
                break
        if header_idx is None:
            continue

        header = lines[header_idx].split()
        rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                break  # a blank line ends the table
            parts = line.split()
            if len(parts) != len(header):
                break
            rows.append(dict(zip(header, parts)))
        if rows:
            return rows
    return []


def _find_images(nb):
    """Match each embedded chart image to what its cell's CODE does, not its
    position, so re-ordering upstream cells can't mix the charts up."""
    images = {}
    for cell in _code_cells(nb):
        src = "".join(cell.get("source", []))
        pngs = []
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                pngs.append("".join(b64) if isinstance(b64, list) else b64)
        if not pngs:
            continue

        if "grad_cam(" in src:
            images["gradcam"] = pngs[0]
        elif "roc_curve(" in src:
            images["roc_curves"] = pngs[0]
        elif "lr_backbone" in src or "Cosine schedule" in src or "backbone lr" in src:
            images["training_curves"] = pngs[0]
    return {name: base64.b64decode(b64) for name, b64 in images.items()}


def extract(nb_path):
    """Returns (summary_dict_or_None, per_label_list, images_dict).

    images_dict maps 'roc_curves' / 'training_curves' / 'gradcam' -> raw PNG
    bytes for whichever of the three it actually found. A notebook missing
    one chart (or not yet reaching that cell) doesn't break the other two —
    each piece is independently None / [] / absent if it can't be found.
    """
    nb = json.loads(Path(nb_path).read_text())
    summary = _find_summary(nb)
    per_label = _find_per_label_table(nb)
    images = _find_images(nb)
    return summary, per_label, images
