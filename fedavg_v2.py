import warnings
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_and_preprocess_data(num_clients: int = 7):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    df = pd.read_csv(url, names=column_names, na_values="?").dropna()

    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])

    X = df.drop(columns=['target']).values
    y = df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Global test set (optional)
    X_train_full, X_global_test, y_train_full, y_global_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    full_train_dataset = TensorDataset(
        torch.tensor(X_train_full, dtype=torch.float32),
        torch.tensor(y_train_full, dtype=torch.long)
    )

    # Partition training data for simulated clients
    partition_size = len(full_train_dataset) // num_clients
    client_train_sets = []
    client_test_sets = []

    for i in range(num_clients):
        start = i * partition_size
        end = (i + 1) * partition_size if i != num_clients - 1 else len(full_train_dataset)
        subset_indices = list(range(start, end))
        subset = torch.utils.data.Subset(full_train_dataset, subset_indices)

        # SECOND LEVEL SPLIT
        train_len = int(0.8 * len(subset))
        test_len = len(subset) - train_len
        train_subset, test_subset = torch.utils.data.random_split(
            subset, [train_len, test_len]
        )

        client_train_sets.append(train_subset)
        client_test_sets.append(test_subset)

    global_test_set = TensorDataset(
        torch.tensor(X_global_test, dtype=torch.float32),
        torch.tensor(y_global_test, dtype=torch.long)
    )

    return client_train_sets, client_test_sets, global_test_set


# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim=13):
        super(HeartDiseaseNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader.dataset), correct / total

# ==============================================================================
# 3. FLOWER CLIENT
# ==============================================================================

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, train_set, local_test_set):
        self.net = HeartDiseaseNet()
        self.trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
        self.testloader = DataLoader(local_test_set, batch_size=32)


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=5)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ==============================================================================
# 4. SERVER STRATEGY & AGGREGATION
# ==============================================================================

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# ==============================================================================
# 5. SIMULATION
# ==============================================================================

def main():
    # 1. Load Data
    NUM_CLIENTS = 7
    client_train_sets, client_test_sets, global_test_set = load_and_preprocess_data(NUM_CLIENTS)

    if client_train_sets is None:
        return
    
    def evaluate_global(server_round, parameters, config):
        net = HeartDiseaseNet()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        testloader = DataLoader(global_test_set, batch_size=32)
        loss, acc = test(net, testloader)
        return loss, {"global_accuracy": acc}

    # 2. Define Client Function
    def client_fn(cid: str) -> fl.client.Client:
        # cid is a string '0', '1', etc. provided by the simulator
        idx = int(cid)
        # In this simulation, every client gets a specific partition AND the global test set
        # (Realistically, clients would test on their own local validation set)
        return HeartDiseaseClient(
            client_train_sets[idx],
            client_test_sets[idx]   # each client gets their own test set
        ).to_client()

    # 3. Define Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2, # Never sample less than 2 clients for evaluation
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global,
        evaluate_metrics_aggregation_fn=weighted_average, # Aggregate accuracy
    )

    # 4. Start Simulation
    print("Starting Flower Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
        # If using Ray backend, you can specify client_resources={'num_cpus': 1}
    )

if __name__ == "__main__":
    main()