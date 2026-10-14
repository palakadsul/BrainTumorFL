
import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.config import BATCH_SIZE, LOCAL_EPOCHS
from common.model import create_model, get_parameters, set_parameters


class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, data_dir, client_id):
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        train_dir = self.data_dir / "Training"
        test_dir = self.data_dir / "Testing"

        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}"
            )
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Testing directory not found: {test_dir}"
            )

        self.train_dataset = datasets.ImageFolder(
            str(train_dir), transform=transform
        )
        self.test_dataset = datasets.ImageFolder(
            str(test_dir), transform=transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        print(f"Client ID: {self.client_id}")
        print(f"Training images: {len(self.train_dataset)}")
        print(f"Validation images: {len(self.test_dataset)}")
        print(f"Device: {self.device}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001,
        )

        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        running_loss = 0.0
        total = 0

        for _ in range(epochs):
            self.model.train()

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                total += images.size(0)

        average_loss = running_loss / total if total else 0.0

        print(
            f"Client {self.client_id}: local training complete, "
            f"images={total}, loss={average_loss:.4f}"
        )

        return (
            get_parameters(self.model),
            len(self.train_dataset),
            {"client_id": self.client_id, "train_loss": float(average_loss)},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        if total == 0:
            print(
                f"Client {self.client_id}: no validation samples available"
            )
            return 0.0, 0, {
                "accuracy": 0.0,
                "client_id": self.client_id,
            }

        val_loss /= total
        accuracy = correct / total

        print(
            f"Client {self.client_id} validation: "
            f"loss={val_loss:.4f}, "
            f"accuracy={accuracy:.2%}, samples={total}"
        )

        return (
            float(val_loss),
            total,
            {
                "accuracy": float(accuracy),
                "client_id": self.client_id,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--client-id", required=True)
    args = parser.parse_args()

    client = BrainTumorClient(args.data, args.client_id)

    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()