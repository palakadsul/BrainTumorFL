
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from flask import Flask, render_template_string, request
from PIL import Image
from torchvision import transforms

from common.config import CLASSES, IMAGE_SIZE
from common.model import create_model

app = Flask(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "global_model.pth"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

# Palak's model-sharing server
MODEL_SERVER = "http://172.16.237.173:8082"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model = create_model().to(device)

PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>BrainTumorFL Dashboard</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 35px auto;
            padding: 0 18px;
            color: #202938;
            background: #f4f7fb;
        }

        h1 { color: #193b75; }

        .panel {
            background: white;
            border: 1px solid #dce2ea;
            border-radius: 14px;
            padding: 22px;
            margin: 18px 0;
            box-shadow: 0 3px 12px rgba(20, 40, 80, 0.04);
        }

        button {
            background: #2457c5;
            color: white;
            border: 0;
            border-radius: 8px;
            padding: 12px 18px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover { background: #19449e; }

        button:disabled {
            background: #929db0;
            cursor: not-allowed;
        }

        input {
            margin: 16px 0;
            max-width: 100%;
        }

        .result { overflow-wrap: anywhere; }
        .status { color: #24613b; font-weight: bold; }
        .warning { color: #9a3412; }
        .note { color: #5c6678; font-size: 13px; }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }

        th, td {
            padding: 10px 8px;
            text-align: left;
            border-bottom: 1px solid #e4e8ef;
        }

        th { background: #f5f7fb; }

        .metric {
            display: inline-block;
            background: #f1f5ff;
            border-radius: 9px;
            padding: 12px;
            margin: 5px;
        }

        .metric strong {
            display: block;
            font-size: 22px;
            margin-top: 5px;
        }
    </style>
</head>

<body>
    <h1>BrainTumorFL Dashboard</h1>
    <p>Federated Brain MRI Classification Demonstration</p>

    <div class="panel">
        <h2>Federated Learning Results</h2>

        {% if metrics %}
            <p class="status">
                &#10003; Average accuracy received from Palak's server
            </p>

            <p>
                Completed federated round:
                <strong>{{ metrics.get('round', 'N/A') }}</strong>
            </p>

            <div class="metric">
                Simple Average Accuracy
                <strong>
                    {{ "%.2f"|format(
                        metrics.get('simple_average_accuracy', 0) * 100
                    ) }}%
                </strong>
            </div>

            <div class="metric">
                Sample-Weighted Accuracy
                <strong>
                    {{ "%.2f"|format(
                        metrics.get('weighted_accuracy', 0) * 100
                    ) }}%
                </strong>
            </div>

            <h3>Individual Client Accuracies</h3>

            <table>
                <thead>
                    <tr>
                        <th>Client</th>
                        <th>Accuracy</th>
                        <th>Evaluation Samples</th>
                    </tr>
                </thead>

                <tbody>
                {% for client in metrics.get('clients', []) %}
                    <tr>
                        <td>
                            {{ client.get('client_id', 'Unknown') }}
                        </td>
                        <td>
                            {{ "%.2f"|format(
                                client.get('accuracy', 0) * 100
                            ) }}%
                        </td>
                        <td>{{ client.get('samples', 'N/A') }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>

            <p class="note">
                Metrics received directly from the federated server.
            </p>

        {% else %}
            <p class="warning">
                Average accuracy has NOT been received from the server.
            </p>

            <p>
                MRI prediction is disabled until the server metrics
                can be retrieved successfully.
            </p>

            {% if metrics_error %}
                <p class="note">Connection details: {{ metrics_error }}</p>
            {% endif %}
        {% endif %}
    </div>

    <div class="panel">
        <h2>Upload an MRI Image</h2>

        {% if metrics %}
            <p class="status">
                Server metrics confirmed. You may upload an image.
            </p>

            <form method="post" enctype="multipart/form-data">
                <input
                    type="file"
                    name="image"
                    accept=".png,.jpg,.jpeg"
                    required
                >
                <br>
                <button type="submit">Predict Class</button>
            </form>
        {% else %}
            <p class="warning">
                Upload and prediction are unavailable until server
                metrics are received.
            </p>
            <button type="button" disabled>Predict Class</button>
        {% endif %}
    </div>

    {% if message %}
    <div class="panel result">
        <h2>{{ message }}</h2>

        {% if prediction %}
            <p>
                Predicted Class:
                <strong>{{ prediction }}</strong>
            </p>

            <p>
                Model Confidence:
                <strong>
                    {{ "%.2f"|format(confidence * 100) }}%
                </strong>
            </p>

            <h3>Class Probabilities</h3>

            <ul>
            {% for name, probability in probabilities %}
                <li>
                    {{ name }}:
                    {{ "%.2f"|format(probability * 100) }}%
                </li>
            {% endfor %}
            </ul>
        {% endif %}
    </div>
    {% endif %}

    <p class="note">
        Research demonstration only. This prediction is not a medical
        diagnosis.
    </p>
</body>
</html>
"""


def fetch_server_file(endpoint, destination):
    """Retrieve a file directly from Palak's server."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        MODEL_SERVER + endpoint,
        headers={"User-Agent": "BrainTumorFL-Dashboard/1.0"},
    )

    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()

    if not data:
        raise ValueError(
            f"The server returned an empty response for {endpoint}"
        )

    destination.write_bytes(data)


def fetch_metrics():
    """Fetch current federated metrics from the server."""
    with urllib.request.urlopen(
        MODEL_SERVER + "/metrics",
        timeout=15,
    ) as response:
        metrics = json.loads(response.read().decode("utf-8"))

    # Validate the fields needed by this dashboard.
    required_fields = [
        "round",
        "simple_average_accuracy",
        "weighted_accuracy",
        "clients",
    ]

    for field in required_fields:
        if field not in metrics:
            raise ValueError(
                f"Server metrics are missing the '{field}' field."
            )

    if not isinstance(metrics["clients"], list):
        raise ValueError("Server client metrics have an invalid format.")

    # Cache a copy for reference, but never rely on the cache to
    # authorize prediction when the server is unreachable.
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    prediction = None
    confidence = None
    probabilities = []

    metrics = None
    metrics_error = None

    # Each page request must successfully retrieve server metrics.
    try:
        metrics = fetch_metrics()
    except Exception as exc:
        metrics_error = str(exc)

    if request.method == "POST":
        uploaded = request.files.get("image")

        if uploaded is None or not uploaded.filename:
            message = "Please select an MRI image."

        elif metrics is None:
            message = (
                "Prediction blocked: current federated accuracy metrics "
                "could not be received from Palak's server."
            )

        else:
            try:
                # Download the global model from Palak's server.
                fetch_server_file("/model", MODEL_PATH)

                image = Image.open(uploaded.stream).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)

                state = torch.load(
                    MODEL_PATH,
                    map_location=device,
                    weights_only=True,
                )

                model.load_state_dict(state, strict=True)
                model.eval()

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    confidence_tensor, class_index = torch.max(
                        probs, dim=0
                    )

                prediction = CLASSES[int(class_index.item())]
                confidence = float(confidence_tensor.item())

                probabilities = sorted(
                    [
                        (CLASSES[i], float(probs[i].item()))
                        for i in range(len(CLASSES))
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )

                message = (
                    "Prediction complete. Server metrics were received "
                    "and the shared global model was used."
                )

            except Exception as exc:
                message = (
                    "Prediction failed. Check that Palak's server is "
                    f"running and reachable. Details: {exc}"
                )

    return render_template_string(
        PAGE,
        message=message,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        metrics=metrics,
        metrics_error=metrics_error,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
