import warnings
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. LOGGING SETUP (UPDATED)
# ==============================================================================
CLIENT_LOG_FILE = "client_metrics.csv"
GLOBAL_LOG_FILE = "global_metrics.csv"

def initialize_logging():
    """Initializes both CSV files with headers."""
    # 1. Setup Client Log
    if os.path.exists(CLIENT_LOG_FILE):
        os.remove(CLIENT_LOG_FILE)
    with open(CLIENT_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["round", "cid", "phase", "loss", "accuracy", "num_samples"])

    # 2. Setup Global Log
    if os.path.exists(GLOBAL_LOG_FILE):
        os.remove(GLOBAL_LOG_FILE)
    with open(GLOBAL_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["round", "loss", "accuracy"])

def log_client_metrics(rnd, cid, phase, loss, accuracy, num_samples):
    with open(CLIENT_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([rnd, cid, phase, f"{loss:.4f}", f"{accuracy:.4f}", num_samples])

def log_global_metrics(rnd, loss, accuracy):
    """Appends a row of global metrics."""
    with open(GLOBAL_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([rnd, f"{loss:.4f}", f"{accuracy:.4f}"])
    print(f">>> GLOBAL LOG: Round {rnd} | Loss: {loss:.4f} | Acc: {accuracy:.4f}")

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_and_preprocess_data(num_clients: int = 7):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    try:
        df = pd.read_csv(url, names=column_names, na_values="?").dropna()
    except Exception:
        # Fallback if URL fails, creates dummy data for testing flow
        print("Data load failed, creating dummy data...")
        df = pd.DataFrame(np.random.randn(300, 14), columns=column_names)
        df['num'] = np.random.randint(0, 2, 300)

    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])

    X = df.drop(columns=['target']).values
    y = df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train_full, X_global_test, y_train_full, y_global_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    full_train_dataset = TensorDataset(
        torch.tensor(X_train_full, dtype=torch.float32),
        torch.tensor(y_train_full, dtype=torch.long)
    )

    partition_size = len(full_train_dataset) // num_clients
    client_train_sets = []
    client_test_sets = []

    for i in range(num_clients):
        start = i * partition_size
        end = (i + 1) * partition_size if i != num_clients - 1 else len(full_train_dataset)
        subset_indices = list(range(start, end))
        subset = torch.utils.data.Subset(full_train_dataset, subset_indices)
        
        train_len = int(0.8 * len(subset))
        test_len = len(subset) - train_len
        train_subset, test_subset = torch.utils.data.random_split(subset, [train_len, test_len])

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
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / (len(trainloader) * epochs), correct / total

def test(net, testloader):
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
    
    # Handle edge case for empty dataloader (though rare in this setup)
    if total == 0: return 0.0, 0.0
    
    return loss / len(testloader.dataset), correct / total

# ==============================================================================
# 3. FLOWER CLIENT
# ==============================================================================

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, cid, train_set, local_test_set):
        self.cid = cid
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
        server_round = config.get("server_round", 0)
        
        loss, accuracy = train(self.net, self.trainloader, epochs=5)
        log_client_metrics(server_round, self.cid, "fit", loss, accuracy, len(self.trainloader.dataset))
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 0)
        
        loss, accuracy = test(self.net, self.testloader)
        log_client_metrics(server_round, self.cid, "evaluate", loss, accuracy, len(self.testloader.dataset))

        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ==============================================================================
# 4. SERVER HELPERS
# ==============================================================================

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if not examples: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config_fn():
    def fit_config_fn(server_round: int):
        return {"server_round": server_round}
    return fit_config_fn

def get_on_evaluate_config_fn():
    def evaluate_config_fn(server_round: int):
        return {"server_round": server_round}
    return evaluate_config_fn

# ==============================================================================
# 5. SIMULATION
# ==============================================================================

def main():
    initialize_logging() # Initializes client_metrics.csv AND global_metrics.csv
    print(f"Logging enabled: {CLIENT_LOG_FILE} and {GLOBAL_LOG_FILE}")

    NUM_CLIENTS = 7
    client_train_sets, client_test_sets, global_test_set = load_and_preprocess_data(NUM_CLIENTS)

    if client_train_sets is None: return

    # --- GLOBAL EVALUATION FUNCTION WITH LOGGING ---
    def evaluate_global(server_round, parameters, config):
        """
        Evaluate the aggregated model on the centralized test set
        and log the results to the CSV.
        """
        net = HeartDiseaseNet()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        testloader = DataLoader(global_test_set, batch_size=32)
        
        # Calculate Metrics
        loss, acc = test(net, testloader)
        
        # Log to Global CSV
        log_global_metrics(server_round, loss, acc)
        
        return loss, {"global_accuracy": acc}

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return HeartDiseaseClient(
            cid, 
            client_train_sets[idx],
            client_test_sets[idx]
        ).to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global, # Pass the logging-enabled eval function here
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=get_on_evaluate_config_fn()
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()