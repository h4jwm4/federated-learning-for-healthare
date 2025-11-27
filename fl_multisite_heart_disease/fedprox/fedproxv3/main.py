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
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
        csv.writer(f).writerow([rnd, cid, location, phase, f"{loss:.4f}", f"{accuracy:.4f}", num_samples])

def log_global_metrics(rnd, loss, accuracy):
    with open(GLOBAL_LOG_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([rnd, f"{loss:.4f}", f"{accuracy:.4f}"])
    print(f">>> GLOBAL ROUND {rnd} | Loss: {loss:.4f} | Acc: {accuracy:.4f}")

# ==============================================================================
# 1. CUSTOM DATASET & MULTI-SITE DATA LOADING
# ==============================================================================

class AugmentedHeartDiseaseDataset(Dataset):
    """
    Custom Dataset that applies Gaussian Noise Injection to continuous features
    during training to prevent overfitting (Data Augmentation).
    """
    def __init__(self, X, y, augment=False, noise_level=0.05):
        # Ensure input is tensor
        self.X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y
        self.augment = augment
        self.noise_level = noise_level
        
        # Indices of continuous features in the dataset columns:
        # 0: age, 3: trestbps, 4: chol, 7: thalach, 9: oldpeak
        # We only apply noise to these. Categorical vars (sex, cp, etc.) are left alone.
        self.cont_indices = [0, 3, 4, 7, 9]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx].clone() # Clone to ensure we don't overwrite original data
        y_sample = self.y[idx]

        if self.augment:
            # Generate Gaussian noise
            noise = torch.randn(len(self.cont_indices)) * self.noise_level
            # Add noise only to continuous indices
            x_sample[self.cont_indices] += noise

        return x_sample, y_sample

def load_partitioned_data(num_clients_per_location=1):
    """
    Loads 4 datasets, cleans them, and partitions them.
    Applies Data Augmentation to the training sets.
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
    
    location_train_data = {} 
    global_test_X = []
    global_test_y = []
    
    for loc_name, url in DATA_URLS.items():
        try:
            df = pd.read_csv(url, names=COLUMNS, na_values="?")
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean()) 
            df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
            df = df.drop(columns=['num'])
            
            X = df.drop(columns=['target']).values
            y = df['target'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            location_train_data[loc_name] = (X_train, y_train)
            global_test_X.append(X_test)
            global_test_y.append(y_test)
            
            print(f"  -> {loc_name}: {len(X_train)} train samples, {len(X_test)} test samples.")
            
        except Exception as e:
            print(f"  !! Error loading {loc_name}: {e}")
            return None, None, None

    all_train_X = np.concatenate([data[0] for data in location_train_data.values()])
    scaler = StandardScaler()
    scaler.fit(all_train_X)
    
    client_train_loaders = []
    client_test_loaders = []
    client_location_map = {} 
    client_id_counter = 0
    
    for loc_name in DATA_URLS.keys():
        X_loc, y_loc = location_train_data[loc_name]
        X_loc = scaler.transform(X_loc)
        
        # We need indices to split, but we want distinct Dataset objects 
        # (one with augment=True, one with augment=False)
        total_len = len(X_loc)
        indices = list(range(total_len))
        
        # Calculate partition sizes
        partition_size = total_len // num_clients_per_location
        lengths = [partition_size] * num_clients_per_location
        lengths[-1] += total_len - sum(lengths)
        
        # Helper to slice array based on lengths
        current_idx = 0
        for i in range(num_clients_per_location):
            end_idx = current_idx + lengths[i]
            client_indices = indices[current_idx:end_idx]
            current_idx = end_idx
            
            # Get data for this client
            X_client = X_loc[client_indices]
            y_client = y_loc[client_indices]
            
            # Split client data into Train (80%) and Val (20%)
            # We do this manually to assign different Datasets
            split_idx = int(0.8 * len(X_client))
            if split_idx == 0: split_idx = 1 # Edge case
            
            X_c_train = X_client[:split_idx]
            y_c_train = y_client[:split_idx]
            
            X_c_val = X_client[split_idx:]
            y_c_val = y_client[split_idx:]
            
            # Create Custom Datasets
            # Train gets Augmentation = True
            train_dataset = AugmentedHeartDiseaseDataset(X_c_train, y_c_train, augment=True)
            # Validation gets Augmentation = False
            val_dataset = AugmentedHeartDiseaseDataset(X_c_val, y_c_val, augment=False)
            
            client_train_loaders.append(train_dataset)
            client_test_loaders.append(val_dataset)
            client_location_map[str(client_id_counter)] = loc_name
            client_id_counter += 1

    X_global = np.concatenate(global_test_X)
    y_global = np.concatenate(global_test_y)
    X_global = scaler.transform(X_global) 
    
    # Global test set (No augmentation)
    global_test_dataset = AugmentedHeartDiseaseDataset(X_global, y_global, augment=False)
    
    return client_train_loaders, client_test_loaders, global_test_dataset, client_location_map

# ==============================================================================
# 2. MODEL DEFINITION & MODIFIED TRAIN FUNCTION (FedProx Core) - UNCHANGED
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

def train(net: HeartDiseaseNet, trainloader: DataLoader, epochs: int, 
          global_weights_snapshot: Optional[List[torch.Tensor]] = None, mu: float = 0.0):
    """
    Train the network using FedProx by adding a proximal term to the loss.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            # --- FedProx Modification: Add Proximal Term ---
            if mu > 0.0 and global_weights_snapshot is not None:
                prox_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_weights_snapshot):
                    prox_term += torch.sum(torch.square(local_param - global_param))
                loss += (mu / 2.0) * prox_term
            # --- End FedProx Modification ---

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
# 3. FLOWER CLIENT (Modified for FedProx) - UNCHANGED
# ==============================================================================

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, cid, location, train_set, local_test_set):
        self.cid = cid
        self.location = location
        self.net = HeartDiseaseNet()
        self.global_weights_snapshot: Optional[List[torch.Tensor]] = None 
        # Pass the Augmented Datasets directly to DataLoader
        self.trainloader = DataLoader(train_set, batch_size=16, shuffle=True)
        self.testloader = DataLoader(local_test_set, batch_size=16)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        self.global_weights_snapshot = [torch.tensor(v).clone().detach() for v in parameters] 
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 0)
        mu = config.get("mu", 0.0) 
        
        loss, accuracy = train(self.net, self.trainloader, epochs=5, 
                               global_weights_snapshot=self.global_weights_snapshot, 
                               mu=mu)
        
        log_client_metrics(server_round, self.cid, self.location, "fit", loss, accuracy, len(self.trainloader.dataset))
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = config.get("server_round", 0)
        
        loss, accuracy = test(self.net, self.testloader) 
        log_client_metrics(server_round, self.cid, self.location, "evaluate", loss, accuracy, len(self.testloader.dataset))

        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ==============================================================================
# 4. SERVER STRATEGY (Configuration for FedProx) - UNCHANGED
# ==============================================================================

FEDPROX_MU = 1.0

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if not examples: return {"accuracy": 0}
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config_fn():
    def fit_config_fn(server_round: int): 
        return {"server_round": server_round, "mu": FEDPROX_MU}
    return fit_config_fn

def get_on_evaluate_config_fn():
    def evaluate_config_fn(server_round: int): return {"server_round": server_round}
    return evaluate_config_fn

# ==============================================================================
# 5. MAIN SIMULATION
# ==============================================================================

def main():
    initialize_logging()
    
    NUM_CLIENTS = 4 
    
    # client_train_sets will now contain AugmentedHeartDiseaseDataset objects
    client_train_sets, client_test_sets, global_test_set, loc_map = load_partitioned_data(num_clients_per_location=1)

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
            loc_map[cid], 
            client_train_sets[idx],
            client_test_sets[idx]
        ).to_client()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, 
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config_fn(),
        on_evaluate_config_fn=get_on_evaluate_config_fn()
    )

    print(f"\nStarting FedProx Simulation (mu={FEDPROX_MU}) with Data Augmentation (Gaussian Noise).")
    print(f"Clients: {NUM_CLIENTS} across 4 locations.\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()