import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from collections import OrderedDict

# Ignore warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# 0. LOGGING SETUP
# ==============================================================================
CENTRAL_LOG_FILE = "centralized_4site_metrics.csv"

def initialize_central_logging():
    """Initializes the CSV file for centralized metrics."""
    if os.path.exists(CENTRAL_LOG_FILE): os.remove(CENTRAL_LOG_FILE)
    with open(CENTRAL_LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])

def log_centralized_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy):
    """Logs the training and testing metrics for one epoch."""
    with open(CENTRAL_LOG_FILE, mode='a', newline='') as f:
        csv.writer(f).writerow([
            epoch, 
            f"{train_loss:.4f}", 
            f"{train_accuracy:.4f}", 
            f"{test_loss:.4f}", 
            f"{test_accuracy:.4f}"
        ])
    print(f">>> CENTRAL EPOCH {epoch} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

# ==============================================================================
# 1. MULTI-SITE DATA LOADING (REUSED FROM FL SCRIPT)
# ==============================================================================

def load_partitioned_data(num_clients_per_location=3):
    """
    Loads 4 datasets, cleans them, and partitions them into 
    num_clients_per_location (3) clients per dataset.
    This function's output is adapted for centralized training below.
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
    
    location_train_data = {} # location_name -> (X_train, y_train)
    global_test_X = []
    global_test_y = []
    
    # 2. Load and Clean Each Dataset Individually
    for loc_name, url in DATA_URLS.items():
        try:
            # Load
            df = pd.read_csv(url, names=COLUMNS, na_values="?")
            
            # Clean: Impute missing values with mean 
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean()) 
            
            # Target conversion (0 = Healthy, 1-4 = Disease)
            df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
            df = df.drop(columns=['num'])
            
            X = df.drop(columns=['target']).values
            y = df['target'].values
            
            # Split this location's data: 80% for training pool, 20% for global test
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
    
    # 4. Create Client Datasets (Used to combine all training data into one pool)
    client_train_sets = []
    
    for loc_name in DATA_URLS.keys():
        X_loc, y_loc = location_train_data[loc_name]
        
        # Transform this location's data using the global scaler
        X_loc = scaler.transform(X_loc)
        
        # Convert to Tensor
        dataset_loc = TensorDataset(
            torch.tensor(X_loc, dtype=torch.float32), 
            torch.tensor(y_loc, dtype=torch.long)
        )
        
        # We split the location data into client subsets (but only collect the training part)
        partition_size = len(dataset_loc) // num_clients_per_location
        lengths = [partition_size] * num_clients_per_location
        lengths[-1] += len(dataset_loc) - sum(lengths)
        
        client_splits = torch.utils.data.random_split(dataset_loc, lengths)
        
        for subset in client_splits:
            # Split into Local Train (80%) and Local Val (20%)
            local_train_len = int(0.8 * len(subset))
            local_val_len = len(subset) - local_train_len
            
            if local_train_len == 0: local_train_len = 1; local_val_len = 0
                
            local_train, _ = torch.utils.data.random_split(subset, [local_train_len, local_val_len])
            
            # Add ONLY the training part to the pool
            client_train_sets.append(local_train)

    # 5. Prepare Global Test Set
    X_global = np.concatenate(global_test_X)
    y_global = np.concatenate(global_test_y)
    X_global = scaler.transform(X_global) # Scale test set!
    
    global_test_dataset = TensorDataset(
        torch.tensor(X_global, dtype=torch.float32), 
        torch.tensor(y_global, dtype=torch.long)
    )
    
    return client_train_sets, global_test_dataset

# ==============================================================================
# 2. MODEL DEFINITION (REUSED FROM FL SCRIPT)
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
    """Standard PyTorch training loop for one or more epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # Run only for the specified number of epochs (usually 1 per main loop call)
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
    # Calculate average loss and accuracy over the entire training epoch(s)
    return total_loss / (len(trainloader) * epochs), correct / total

def test(net, testloader):
    """Standard PyTorch testing/evaluation loop."""
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
# 5. MAIN CENTRALIZED TRAINING
# ==============================================================================

def main_centralized(num_epochs=30):
    """
    Main function to run the centralized training baseline.
    """
    initialize_central_logging()
    
    # 1. Load and prepare all data
    client_train_sets, global_test_set = load_partitioned_data(num_clients_per_location=3)

    if client_train_sets is None: return

    # 2. Combine all client training subsets into a single large dataset
    full_train_dataset = ConcatDataset(client_train_sets)
    
    # 3. Create DataLoaders
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    trainloader = DataLoader(full_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    testloader = DataLoader(global_test_set, batch_size=TEST_BATCH_SIZE)
    
    print(f"\nTotal Centralized Training Samples: {len(full_train_dataset)}")
    print(f"Total Global Test Samples: {len(global_test_set)}")

    # 4. Initialize Model
    net = HeartDiseaseNet()
    
    # 5. Training Loop
    print(f"\nStarting Centralized Training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss, train_accuracy = train(net, trainloader, epochs=1) 
        
        # Evaluate on the combined global test set
        test_loss, test_accuracy = test(net, testloader)
        
        # Log the metrics
        log_centralized_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy)

    print("\nCentralized Training Complete.")
    print(f"Results saved to '{CENTRAL_LOG_FILE}'")

if __name__ == "__main__":
    main_centralized(num_epochs=30)