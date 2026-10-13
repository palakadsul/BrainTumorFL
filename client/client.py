
import argparse
from collections import OrderedDict

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.model import create_model, get_parameters, set_parameters
from common.config import CLASSES, IMAGE_SIZE, BATCH_SIZE, LOCAL_EPOCHS


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
            f"Expected class folders {CLASSES}, "
            f"but found {dataset.classes} in {data_dir}"
        )

    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    ), len(dataset)


class BrainTumorClient(fl.client.NumPyClient):
    def __init__(self, data_dir):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = create_model().to(self.device)
        self.trainloader, self.num_examples = load_data(data_dir)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        total_loss = 0.0
        for _ in range(LOCAL_EPOCHS):
            for images, labels in self.trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)

        average_loss = total_loss / (
            self.num_examples * LOCAL_EPOCHS
        )

        print(
            f"Local training complete: "
            f"{self.num_examples} images, loss={average_loss:.4f}"
        )

        return (
            get_parameters(self.model),
            self.num_examples,
            {"train_loss": float(average_loss)},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return 0.0, self.num_examples, {}


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
