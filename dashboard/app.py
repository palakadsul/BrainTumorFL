
import sys
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
            max-width: 760px;
            margin: 40px auto;
            padding: 0 20px;
            color: #202938;
            background: #f7f9fc;
        }
        .panel {
            background: white;
            border: 1px solid #dce2ea;
            border-radius: 14px;
            padding: 24px;
            margin: 20px 0;
        }
        button {
            background: #2457c5;
            color: white;
            border: 0;
            border-radius: 8px;
            padding: 12px 18px;
            cursor: pointer;
        }
        input { margin: 16px 0; max-width: 100%; }
        .result { overflow-wrap: anywhere; }
        .note { color: #5c6678; font-size: 14px; }
    </style>
</head>
<body>
    <h1>BrainTumorFL Dashboard</h1>
    <p>Federated brain MRI classification demonstration</p>

    <div class="panel">
        <h2>Upload an MRI image</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image"
                   accept=".png,.jpg,.jpeg" required>
            <br>
            <button type="submit">Predict class</button>
        </form>
    </div>

    {% if message %}
    <div class="panel result">
        <h2>{{ message }}</h2>
        {% if prediction %}
            <p>Predicted class: <strong>{{ prediction }}</strong></p>
            <p>Model confidence: {{ "%.2f"|format(confidence * 100) }}%</p>
            <h3>Class probabilities</h3>
            <ul>
            {% for name, probability in probabilities %}
                <li>{{ name }}: {{ "%.2f"|format(probability * 100) }}%</li>
            {% endfor %}
            </ul>
        {% endif %}
    </div>
    {% endif %}

    <p class="note">
        Research demonstration only. This prediction is not a medical diagnosis.
    </p>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    prediction = None
    confidence = None
    probabilities = []

    if request.method == "POST":
        uploaded = request.files.get("image")

        if uploaded is None or not uploaded.filename:
            message = "Please select an image."

        elif not MODEL_PATH.is_file():
            message = (
                "The global model is not available yet. "
                "Start federated training and wait for aggregation."
            )

        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)

                state = torch.load(
                    MODEL_PATH,
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(state)
                model.eval()

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    confidence_tensor, class_index = torch.max(probs, dim=0)

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
                message = "Prediction complete"

            except (OSError, ValueError, RuntimeError) as exc:
                message = f"Could not process the image or model: {exc}"

    return render_template_string(
        PAGE,
        message=message,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

