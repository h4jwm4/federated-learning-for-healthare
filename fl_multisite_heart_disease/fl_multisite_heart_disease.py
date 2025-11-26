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

# Ignore warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. LOGGING SETUP
# ==============================================================================
CLIENT_LOG_FILE = "client_metrics.csv"
GLOBAL_LOG_FILE = "global_metrics.csv"

def initialize_logging():
    if os.path.exists(CLIENT_LOG_FILE): os.remove(CLIENT_LOG_FILE)
    if os.path.exists(GLOBAL_LOG_FILE): os.remove(GLOBAL_LOG_FILE)
    
    with open(CLIENT_LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["round", "cid", "location", "phase", "loss", "accuracy", "num_samples"])
        
    with open(GLOBAL_LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["round", "loss", "accuracy"])

def log_client_metrics(rnd, cid, location, phase, loss, accuracy, num_samples):
    with open(CLIENT_LOG_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([rnd, cid, location, f"{loss:.4f}", f"{accuracy:.4f}", num_samples])

def log_global_metrics(rnd, loss, accuracy):
    with open(GLOBAL_LOG_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([rnd, f"{loss:.4f}", f"{accuracy:.4f}"])
    print(f">>> GLOBAL ROUND {rnd} | Loss: {loss:.4f} | Acc: {accuracy:.4f}")

# ==============================================================================
# 1. MULTI-SITE DATA LOADING & PARTITIONING
# ==============================================================================

def load_partitioned_data(num_clients_per_location=3):
    """
    Loads 4 datasets, cleans them, and partitions them into 
    num_clients_per_location (3) clients per dataset.
    Total Clients = 4 locations * 3 = 12.
    """
    
    # 1. Define Sources
    DATA_URLS = {
        'Cleveland': "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        'Hungarian': "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        'Switzerland': "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        'VA_Long_Beach': "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    }
    
    COLUMNS = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    
    print("Loading and processing datasets from 4 locations...")
    
    # Storage for processed splits
    location_train_data = {} # location_name -> (X_train, y_train)
    global_test_X = []
    global_test_y = []
    
    # 2. Load and Clean Each Dataset Individually
    for loc_name, url in DATA_URLS.items():
        try:
            # Load
            df = pd.read_csv(url, names=COLUMNS, na_values="?")
            
            # Clean: Impute missing values with mean (crucial for Switzerland/VA)
            # We convert to numeric first to coerce any errors
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean()) 
            
            # Target conversion (0 = Healthy, 1-4 = Disease)
            df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
            df = df.drop(columns=['num'])
            
            X = df.drop(columns=['target']).values
            y = df['target'].values
            
            # Split this location's data: 80% for its clients, 20% for global test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            location_train_data[loc_name] = (X_train, y_train)
            global_test_X.append(X_test)
            global_test_y.append(y_test)
            
            print(f"  -> {loc_name}: {len(X_train)} train samples, {len(X_test)} test samples.")
            
        except Exception as e:
            print(f"  !! Error loading {loc_name}: {e}")
            return None, None, None

    # 3. Global Standardization
    # We fit the scaler on ALL training data to ensure consistent input features
    all_train_X = np.concatenate([data[0] for data in location_train_data.values()])
    scaler = StandardScaler()
    scaler.fit(all_train_X)
    
    # 4. Create Client Datasets
    client_train_loaders = []
    client_test_loaders = []
    
    # Mapping to track which client belongs to which location
    client_location_map = {} 
    
    client_id_counter = 0
    
    # Order: Cleveland (0-2), Hungarian (3-5), Switzerland (6-8), VA (9-11)
    for loc_name in DATA_URLS.keys():
        X_loc, y_loc = location_train_data[loc_name]
        
        # Transform this location's data using the global scaler
        X_loc = scaler.transform(X_loc)
        
        # Convert to Tensor
        dataset_loc = TensorDataset(
            torch.tensor(X_loc, dtype=torch.float32), 
            torch.tensor(y_loc, dtype=torch.long)
        )
        
        # Split into 'num_clients_per_location' parts
        partition_size = len(dataset_loc) // num_clients_per_location
        lengths = [partition_size] * num_clients_per_location
        # Add remainder to the last client
        lengths[-1] += len(dataset_loc) - sum(lengths)
        
        client_splits = torch.utils.data.random_split(dataset_loc, lengths)
        
        for subset in client_splits:
            # Further split each client's data into Local Train (80%) and Local Val (20%)
            local_train_len = int(0.8 * len(subset))
            local_val_len = len(subset) - local_train_len
            
            # Handle edge case for very small datasets (Switzerland)
            if local_train_len == 0: local_train_len = 1; local_val_len = 0
                
            local_train, local_val = torch.utils.data.random_split(subset, [local_train_len, local_val_len])
            
            client_train_loaders.append(local_train)
            client_test_loaders.append(local_val)
            client_location_map[str(client_id_counter)] = loc_name
            client_id_counter += 1

    # 5. Prepare Global Test Set
    X_global = np.concatenate(global_test_X)
    y_global = np.concatenate(global_test_y)
    X_global = scaler.transform(X_global) # Don't forget to scale test set!
    
    global_test_dataset = TensorDataset(
        torch.tensor(X_global, dtype=torch.float32), 
        torch.tensor(y_global, dtype=torch.long)
    )
    
    return client_train_loaders, client_test_loaders, global_test_dataset, client_location_map

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
    total_loss, correct, total = 0.0, 0, 0
    
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
            
    if total == 0: return 0.0, 0.0
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
            
    if total == 0: return 0.0, 0.0
    return loss / len(testloader.dataset), correct / total

# ==============================================================================
# 3. FLOWER CLIENT
# ==============================================================================

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, cid, location, train_set, local_test_set):
        self.cid = cid
        self.location = location
        self.net = HeartDiseaseNet()
        self.trainloader = DataLoader(train_set, batch_size=16, shuffle=True) # Reduced batch size for small data
        self.testloader = DataLoader(local_test_set, batch_size=16)

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
        
        # Log metric with Location info
        log_client_metrics(server_round, self.cid, self.location, "fit", loss, accuracy, len(self.trainloader.dataset))
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 0)
        
        loss, accuracy = test(self.net, self.testloader)
        log_client_metrics(server_round, self.cid, self.location, "evaluate", loss, accuracy, len(self.testloader.dataset))

        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ==============================================================================
# 4. SERVER STRATEGY
# ==============================================================================

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if not examples: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config_fn():
    def fit_config_fn(server_round: int): return {"server_round": server_round}
    return fit_config_fn

def get_on_evaluate_config_fn():
    def evaluate_config_fn(server_round: int): return {"server_round": server_round}
    return evaluate_config_fn

# ==============================================================================
# 5. MAIN SIMULATION
# ==============================================================================

def main():
    initialize_logging()
    
    # 1. Load Data (12 clients total, 3 per location)
    NUM_CLIENTS = 12
    client_train_sets, client_test_sets, global_test_set, loc_map = load_partitioned_data(num_clients_per_location=3)

    if client_train_sets is None: return
    
    # Global Eval Function
    def evaluate_global(server_round, parameters, config):
        net = HeartDiseaseNet()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        testloader = DataLoader(global_test_set, batch_size=32)
        loss, acc = test(net, testloader)
        log_global_metrics(server_round, loss, acc)
        return loss, {"global_accuracy": acc}

    # Client Function
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return HeartDiseaseClient(
            cid,
            loc_map[cid], # Pass location name for logging
            client_train_sets[idx],
            client_test_sets[idx]
        ).to_client()

    # Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Sample all clients
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=get_on_evaluate_config_fn()
    )

    print(f"\nStarting Simulation with {NUM_CLIENTS} clients across 4 locations.")
    print("Partitioning: 0-2 (Cleveland), 3-5 (Hungarian), 6-8 (Switzerland), 9-11 (VA Long Beach)\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()