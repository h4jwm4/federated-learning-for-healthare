import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
global_file = './global_metrics.csv'
centralized_file = './centralized_4site_metrics.csv'
accuracy_plot_path = '4site_accuracy_comparison.png'
loss_plot_path = '4site_loss_comparison.png'

# --- Data Loading and Preprocessing ---
# Load data
df_global = pd.read_csv(global_file)
df_centralized = pd.read_csv(centralized_file)

# 1. Prepare Global Metrics (FL)
df_fl = df_global.rename(columns={'round': 'iteration', 'loss': 'test_loss', 'accuracy': 'test_accuracy'})
df_fl['model'] = 'Federated (Global)'
df_fl = df_fl[['iteration', 'test_loss', 'test_accuracy', 'model']]

# 2. Prepare Centralized Metrics
df_central = df_centralized.rename(columns={'epoch': 'iteration'})
df_central = df_central[['iteration', 'test_loss', 'test_accuracy']]
df_central['model'] = 'Centralized (4 Sites)'

# Combine for plotting
df_combined = pd.concat([df_fl, df_central], ignore_index=True)

# Ensure numeric types
df_combined['test_accuracy'] = pd.to_numeric(df_combined['test_accuracy'], errors='coerce')
df_combined['test_loss'] = pd.to_numeric(df_combined['test_loss'], errors='coerce')

# --- Plotting Functions ---

# 1. Accuracy Plot
plt.figure(figsize=(10, 6))
for name, group in df_combined.groupby('model'):
    plt.plot(group['iteration'], group['test_accuracy'], label=name, marker='o', markersize=4)

plt.title('Comparison of Test Accuracy (4 Sites: Centralized vs. Federated)')
plt.xlabel('Round (FL) / Epoch (Centralized)')
plt.ylabel('Test Accuracy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Model Type')
# Set y-limit based on data minimum
min_acc = df_combined['test_accuracy'].min()
plt.ylim(min_acc * 0.9 if min_acc > 0 else 0, 1.05)
plt.xticks(df_combined['iteration'].unique()[::2])
plt.tight_layout()
plt.savefig(accuracy_plot_path)
plt.close()
print(f"Accuracy plot saved to {accuracy_plot_path}")

# 2. Loss Plot
plt.figure(figsize=(10, 6))
for name, group in df_combined.groupby('model'):
    plt.plot(group['iteration'], group['test_loss'], label=name, marker='o', markersize=4)

plt.title('Comparison of Test Loss (4 Sites: Centralized vs. Federated)')
plt.xlabel('Round (FL) / Epoch (Centralized)')
plt.ylabel('Test Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Model Type')
plt.ylim(0, df_combined['test_loss'].max() * 1.05)
plt.xticks(df_combined['iteration'].unique()[::2])
plt.tight_layout()
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss plot saved to {loss_plot_path}")