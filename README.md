---
title: ChestX-ray14 Replication
emoji: 🫁
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ChestX-ray14 Replication — Showcase Site

A Flask website presenting a from-scratch replication of Kufel et al.
(2023): an EfficientNet-B1 model that predicts 14 chest conditions from
a single X-ray. Includes a live demo that runs the trained model on
whatever image a visitor uploads.

## 1. Run the site right now (no model needed yet)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. Home, Results, and Methodology all work
immediately. The Demo page will say the model isn't loaded yet — that's
expected until you do step 2.

## 2. Turn on the live demo

```bash
pip install -r requirements-model.txt
```

Then copy your trained checkpoint to:

```
model/chestxray_b1_portable.pt
```

(the file your Kaggle notebook's packaging cell produces — or a raw
`best.pt` also works, just with default 0.5 thresholds instead of the
calibrated ones). Restart `python app.py`. The Demo page should now
say the model is ready.

## 3. Populate the Results page with your real numbers

Copy two files straight from your Kaggle `artifacts/` folder into
`data/`:

```
data/summary.json
data/extended_metrics.csv
```

Reload the Results page — no other changes needed.

## Project layout

```
app.py                      Flask routes — the whole app's logic
model/
  chestxray_model.py        Loads the checkpoint, runs predictions
  chestxray_b1_portable.pt  <- your trained weights go here (not included)
templates/                  HTML pages (Jinja2)
static/css/style.css        All styling
static/js/demo.js           Live image preview on the demo page
data/                       Drop summary.json + extended_metrics.csv here
uploads/                    X-rays visitors upload land here temporarily
```

## Deploying to Hugging Face Spaces

This repo is already set up for it — `Dockerfile`, `.dockerignore`, and the
config block at the top of this file are all in place.

1. Create a free account at huggingface.co, then **New Space** → SDK: **Docker** → Visibility: your choice.
2. Push this whole folder to the Space's own git repo (shown on the Space's page after creation), the same way you'd push to GitHub.
3. Under the Space's **Settings → Repository secrets**, add one secret: `SECRET_KEY` set to any random string of your choosing (this replaces the placeholder dev key — never reuse the one visible in this repo).
4. Wait for the build to finish (the Space's **Logs** tab shows progress) — expect several minutes the first time, since it's installing PyTorch.
5. Your site is live at `https://<your-username>-<space-name>.hf.space`.

`app.py` already reads `SECRET_KEY` from the environment if it's set
(falling back to the dev placeholder locally, where it doesn't matter).
