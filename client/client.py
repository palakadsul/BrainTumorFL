
import argparse
import random
from pathlib import Path

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from common.config import (
    CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    LOCAL_EPOCHS,
)
from common.model import create_model, get_parameters, set_parameters


# Paths are relative to the project, not the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "global_model.pth"


def save_global_weights(model):
    """Save the latest received global model for the local dashboard."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = MODEL_PATH.with_suffix(".tmp")
    torch.save(model.state_dict(), temp_path)
    temp_path.replace(MODEL_PATH)

    print(f"Latest global weights saved to: {MODEL_PATH}")


def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    if dataset.classes != CLASSES:
        raise ValueError(
            f"Expected classes {CLASSES}, found {dataset.classes}"
        )

    if len(dataset) < 2:
        raise ValueError("At least two images are required for a split.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    trainloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valloader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, valloader, train_size, val_size


class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model().to(self.device)

        (
            self.trainloader,
            self.valloader,
            self.num_train,
            self.num_val,
        ) = load_data(data_dir)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Receive the latest global model from the server.
        set_parameters(self.model, parameters)

        # Save these received global weights for the prediction dashboard.
        save_global_weights(self.model)

        self.model.train()
        loss_sum = 0.0
        seen = 0

        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item() * images.size(0)
                seen += images.size(0)

        train_loss = loss_sum / max(seen, 1)

        print(
            f"Local training complete: {self.num_train} training images, "
            f"loss={train_loss:.4f}"
        )

        # Return locally trained weights to the server for FedAvg.
        return (
            get_parameters(self.model),
            self.num_train,
            {"train_loss": float(train_loss)},
        )

    def evaluate(self, parameters, config):
        # Use the global parameters for local validation.
        set_parameters(self.model, parameters)
        save_global_weights(self.model)

        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.valloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                loss_sum += loss.item() * images.size(0)
                correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                total += labels.size(0)

        if total == 0:
            return 0.0, 0, {"accuracy": 0.0}

        val_loss = loss_sum / total
        accuracy = correct / total

        print(
            f"Local validation: loss={val_loss:.4f}, "
            f"accuracy={accuracy:.2%}, samples={total}"
        )

        return (
            float(val_loss),
            total,
            {"accuracy": float(accuracy)},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    client = BrainTumorClient(args.data)

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()


