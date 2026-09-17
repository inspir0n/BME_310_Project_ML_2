# Base image with Python already installed. "slim" = smaller download,
# missing a few things we add back in explicitly below.
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless still expects these two system libraries to exist,
# even though it skips the GUI parts of OpenCV. Missing them shows up as a
# confusing ImportError the first time something tries to read an image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy just the dependency lists first, install them, THEN copy the rest of
# the code. Docker caches each step -- this way, editing a template later
# doesn't force a slow re-install of torch on every rebuild.
COPY requirements.txt requirements-model.txt ./
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt -r requirements-model.txt

COPY . .

# Hugging Face Spaces routes traffic to whichever port the Space is
# configured for -- ours is set to 7860 in README.md's app_port.
EXPOSE 7860

# gunicorn is a production-grade server -- unlike "python app.py", it's built
# to actually hold up under real traffic. One worker is plenty for a class
# project demo, and keeps memory use predictable.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "60", "app:app"]
