from pathlib import Path

import flwr as fl
import torch

from common.config import NUM_ROUNDS
from common.model import create_model, get_parameters, set_parameters

HOST = "0.0.0.0:8080"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "global_model.pth"


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
                f"{MODEL_PATH.resolve()}"
            )

        return aggregated_parameters, metrics


def main():
    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(model)
    )

    strategy = SavingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_evaluate_clients=0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    print("Starting BrainTumorFL server...")
    print("Waiting for both clients to connect...")

    fl.server.start_server(
        server_address=HOST,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print(f"Federated training finished. Model path: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
