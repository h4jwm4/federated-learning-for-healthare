import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
from collections import OrderedDict

# --- Configuration ---
CENTRAL_LOG_FILE = "centralized_baseline_metrics.csv"
EPOCHS = 50  # Chosen as a balance point for convergence on small data

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING (REUSED)
# ==============================================================================

def load_and_preprocess_centralized():
    """Loads data and prepares the full training and global test sets."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    try:
        df = pd.read_csv(url, names=column_names, na_values="?").dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num'])

    X = df.drop(columns=['target']).values
    y = df['target'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Global test set (20% of the data)
    X_train_full, X_global_test, y_train_full, y_global_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Centralized DataLoader uses the entire 80% for training
    train_dataset = TensorDataset(
        torch.tensor(X_train_full, dtype=torch.float32),
        torch.tensor(y_train_full, dtype=torch.long)
    )
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Global Test DataLoader
    test_dataset = TensorDataset(
        torch.tensor(X_global_test, dtype=torch.float32),
        torch.tensor(y_global_test, dtype=torch.long)
    )
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader

# ==============================================================================
# 2. MODEL DEFINITION (REUSED)
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

# ==============================================================================
# 3. UTILITY FUNCTIONS (REUSED)
# ==============================================================================

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

def initialize_log():
    """Initializes the CSV file with headers."""
    if os.path.exists(CENTRAL_LOG_FILE):
        os.remove(CENTRAL_LOG_FILE)
    
    with open(CENTRAL_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])

# ==============================================================================
# 4. CENTRALIZED TRAINING LOGIC
# ==============================================================================

def train_and_log(net, trainloader, testloader, epochs):
    """Trains the network centrally and logs metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    initialize_log()
    print(f"Starting Centralized Training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # Calculate training metrics
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total

        # Evaluate on global test set
        test_loss_avg, test_acc = test(net, testloader)
        
        # Log results
        with open(CENTRAL_LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, f"{train_loss_avg:.4f}", f"{train_acc:.4f}", f"{test_loss_avg:.4f}", f"{test_acc:.4f}"])

        print(f"Epoch {epoch:2d}/{epochs}: Train Loss={train_loss_avg:.4f}, Test Acc={test_acc:.4f}")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

def main():
    trainloader, testloader = load_and_preprocess_centralized()
    if trainloader is None:
        return
        
    net = HeartDiseaseNet()
    
    train_and_log(net, trainloader, testloader, epochs=EPOCHS)
    
    print(f"\nCentralized training complete. Results saved to {CENTRAL_LOG_FILE}")
    final_acc = pd.read_csv(CENTRAL_LOG_FILE)['test_accuracy'].iloc[-1]
    print(f"Final Centralized Test Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()