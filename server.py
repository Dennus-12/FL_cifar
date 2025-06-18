import flwr as fl

def main():
    # Define the strategy (FedAvg is the default strategy in Flower)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,  # Minimum number of clients to participate in training
        min_evaluate_clients=3,  # Minimum number of clients to participate in evaluation
        min_available_clients=3,  # Minimum number of total clients available
    )
    



    # Start the server and wait for the clients to connect
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),  # Number of federated learning rounds
    )

if __name__ == "__main__":
    main()
