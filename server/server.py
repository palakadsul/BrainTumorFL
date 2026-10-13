
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import torch

from common.config import NUM_ROUNDS
from common.model import create_model, get_parameters, set_parameters

HOST = "0.0.0.0:8080"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "global_model.pth"
INITIAL_WEIGHTS = PROJECT_ROOT / "weights" / "resnet18.pt"


def load_initial_parameters(model):
    checkpoint = torch.load(
        INITIAL_WEIGHTS, map_location="cpu", weights_only=True
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"Initialized global model from: {INITIAL_WEIGHTS}")
    return get_parameters(model)


class SavingFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            model = create_model()
            arrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )
            set_parameters(model, arrays)

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

            print(
                f"Round {server_round}: global model saved to "
                f"{MODEL_PATH}"
            )

        return aggregated_parameters, metrics


def main():
    if not INITIAL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {INITIAL_WEIGHTS}"
        )

    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(
        load_initial_parameters(model)
    )

    strategy = SavingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    print("Starting BrainTumorFL server...")
    print("Initial model: weights/resnet18.pt")
    print("Waiting for both clients to connect...")

    fl.server.start_server(
        server_address=HOST,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print(f"Federated training finished. Model path: {MODEL_PATH}")


if __name__ == "__main__":
    main()




