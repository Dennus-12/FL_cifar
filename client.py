from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import cifar as cifar
import flwr as fl

import argparse

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    




def main(client_id: int, num_clients: int) -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = cifar.Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = cifar.load_data(client_id, num_clients)

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)

def parse_args():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(client_id=args.client_id, num_clients=args.num_clients)