
from pathlib import Path

import torch
from flask import Flask, render_template_string, request
from PIL import Image
from torchvision import transforms

from common.config import CLASSES, IMAGE_SIZE
from common.model import create_model

app = Flask(__name__)

MODEL_PATH = Path("models/global_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
<html>
<head>
    <title>BrainTumorFL Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 720px;
            margin: 40px auto;
            padding: 0 18px;
            color: #222;
        }
        .panel {
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 24px;
            margin-top: 20px;
        }
        button {
            padding: 10px 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>BrainTumorFL Dashboard</h1>
    <p>Federated brain MRI image classification demo</p>

    <div class="panel">
        <h2>Upload an MRI image</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image"
                   accept=".png,.jpg,.jpeg" required>
            <button type="submit">Predict</button>
        </form>
    </div>

    {% if message %}
    <div class="panel">
        <h3>{{ message }}</h3>
        {% if prediction %}
            <p>Predicted class: <strong>{{ prediction }}</strong></p>
            <p>Model confidence:
               {{ "%.2f"|format(confidence * 100) }}%</p>
            <h4>Class probabilities</h4>
            <ul>
            {% for name, probability in probabilities %}
                <li>{{ name }}:
                    {{ "%.2f"|format(probability * 100) }}%</li>
            {% endfor %}
            </ul>
        {% endif %}
    </div>
    {% endif %}

    <p><small>Research demo only; not a medical diagnosis.</small></p>
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

        if not uploaded or not uploaded.filename:
            message = "Please select an image."

        elif not MODEL_PATH.exists():
            message = (
                "Global model not found. Run federated training first."
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
                    conf, index = torch.max(probs, dim=0)

                prediction = CLASSES[int(index)]
                confidence = float(conf)

                probabilities = sorted(
                    [
                        (CLASSES[i], float(probs[i]))
                        for i in range(len(CLASSES))
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )

                message = "Prediction complete"

            except Exception as exc:
                message = f"Could not process image: {exc}"

    return render_template_string(
        PAGE,
        message=message,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

