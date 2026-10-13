
import flwr as fl
from pathlib import Path

from common.model import create_model, get_parameters

NUM_ROUNDS = 3
HOST = "0.0.0.0:8080"


def main():
    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(
        get_parameters(model)
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    print("Starting BrainTumorFL server...")
    print("Waiting for both clients to connect.")

    fl.server.start_server(
        server_address=HOST,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
