import json
from pathlib import Path

from flask import Flask, jsonify, send_file

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "global_model.pth"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": MODEL_PATH.is_file(),
        "metrics_available": METRICS_PATH.is_file(),
    })


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.is_file():
        return jsonify({"error": "Metrics file not found"}), 404
    with METRICS_PATH.open(encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/model")
def model():
    if not MODEL_PATH.is_file():
        return jsonify({"error": "Global model not found"}), 404
    return send_file(
        MODEL_PATH,
        as_attachment=True,
        download_name="global_model.pth",
    )


if __name__ == "__main__":
    print("BrainTumorFL model-sharing service")
    print("Model:", MODEL_PATH)
    print("Metrics:", METRICS_PATH)
    app.run(host="0.0.0.0", port=8082, debug=False)
