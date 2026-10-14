import argparse
import json
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from common.config import CLASSES, IMAGE_SIZE
from common.model import create_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "global_model.pth"
METRICS_PATH = ROOT / "models" / "metrics.json"

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "BrainTumorFL/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def main():
    parser = argparse.ArgumentParser(
        description="Predict a brain MRI class using the shared global model."
    )
    parser.add_argument("--server", required=True,
                        help="For example: http://192.168.1.10:8082")
    parser.add_argument("--image", required=True,
                        help="Path to an MRI image on this computer")
    args = parser.parse_args()
    base_url = args.server.rstrip("/")

    print("Downloading the final global model...")
    download_file(base_url + "/model", MODEL_PATH)
    download_file(base_url + "/metrics", METRICS_PATH)

    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)

    print("\nServer-reported federated results")
    print("Round:", metrics.get("round", "unknown"))
    print("Sample-weighted accuracy:",
          metrics.get("weighted_accuracy_percent", "unknown"), "%")
    print("Simple average accuracy:",
          metrics.get("simple_average_accuracy_percent", "unknown"), "%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"MRI image not found: {image_path}")

    with Image.open(image_path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = int(probabilities.argmax().item())
    print("\nPrediction result")
    print("Predicted class:", CLASSES[index])
    print(f"Model confidence: {probabilities[index].item():.2%}")
    print("\nClass probabilities:")
    for i in probabilities.argsort(descending=True).tolist():
        print(f"  {CLASSES[i]}: {probabilities[i].item():.2%}")

    print("\nResearch demonstration only; not a medical diagnosis.")


if __name__ == "__main__":
    main()
