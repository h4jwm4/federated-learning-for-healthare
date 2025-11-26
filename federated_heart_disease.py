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

def load_and_preprocess_data(num_clients: int = 3):
    """
    Downloads UCI Heart Disease data, preprocesses it, and splits it for clients.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    
    print(f"Downloading data from {url}...")
    try:
        df = pd.read_csv(url, names=column_names, na_values="?")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None

    # Drop rows with missing values (ca and thal have some '?')
    df = df.dropna()

    # Convert 'num' to binary: 0 = No Disease, 1-4 = Disease
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])

    # Separate features and target
    X = df.drop(columns=['target']).values
    y = df['target'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test sets (global split for evaluation references)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    # Partition training data for simulated clients
    # We simply split the training set into 'num_clients' chunks
    partition_size = len(train_data) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start = i * partition_size
        end = (i + 1) * partition_size if i != num_clients - 1 else len(train_data)
        subset = torch.utils.data.Subset(train_data, list(range(start, end)))
        client_datasets.append(subset)

    print(f"Data loaded. Total samples: {len(df)}. Clients: {num_clients}. Partition size: ~{partition_size}")
    
    # We return the client partitions and the global test set
    return client_datasets, test_data

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
    def __init__(self, train_set, test_set):
        self.net = HeartDiseaseNet()
        self.trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
        self.testloader = DataLoader(test_set, batch_size=32)

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
    client_datasets, global_test_set = load_and_preprocess_data(NUM_CLIENTS)

    if client_datasets is None:
        return

    # 2. Define Client Function
    def client_fn(cid: str) -> fl.client.Client:
        # cid is a string '0', '1', etc. provided by the simulator
        idx = int(cid)
        # In this simulation, every client gets a specific partition AND the global test set
        # (Realistically, clients would test on their own local validation set)
        return HeartDiseaseClient(client_datasets[idx], global_test_set).to_client()

    # 3. Define Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2, # Never sample less than 2 clients for evaluation
        min_available_clients=NUM_CLIENTS,
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