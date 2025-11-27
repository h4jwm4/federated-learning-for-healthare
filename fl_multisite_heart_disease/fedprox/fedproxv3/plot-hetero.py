import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Added os import for file existence check

# --- Configuration ---
GLOBAL_METRICS_FILE = "./global_metrics.csv"
CLIENT_METRICS_FILE = "./client_metrics.csv"
OUTPUT_FILE = 'client_heterogeneity_plot.png'

# Check if files exist before proceeding
if not os.path.exists(GLOBAL_METRICS_FILE) or not os.path.exists(CLIENT_METRICS_FILE):
    print(f"Error: Required files not found. Ensure {GLOBAL_METRICS_FILE} and {CLIENT_METRICS_FILE} exist.")
    exit()

# 1. Load Data
df_global = pd.read_csv(GLOBAL_METRICS_FILE)
df_client = pd.read_csv(CLIENT_METRICS_FILE)

# --- Data Cleaning and Aggregation ---
df_client_eval = df_client[df_client['phase'] == 'evaluate'].copy()
max_round = df_global['round'].max()

# 2. Filter data for the final evaluation round
df_final_round = df_client_eval[df_client_eval['round'] == max_round].sort_values(by='cid')

# Convert cid to integer for cleaner plotting
df_final_round['cid'] = df_final_round['cid'].astype(int)

# 3. Define the custom color mapping function for 4 clients (C0 to C3)
def get_client_color(cid):
    """Maps the 4 client IDs to their respective locations and colors."""
    if cid == 0:
        return 'darkblue'  # Client 0: Cleveland
    elif cid == 1:
        return 'darkgreen'  # Client 1: Hungarian
    elif cid == 2:
        return 'orange'  # Client 2: Switzerland
    elif cid == 3:
        return 'purple'  # Client 3: VA Long Beach
    else:
        return 'gray' # Fallback for unexpected CIDs

# Apply the color mapping to the DataFrame
colors = df_final_round['cid'].apply(get_client_color)

# 4. Create the Bar Plot
plt.figure(figsize=(10, 6)) # Reduced figure size for 4 clients
    
# Create the Bar Plot
bars = plt.bar(df_final_round['cid'].apply(lambda x: f'Client {x}'), 
               df_final_round['accuracy'], 
               color=colors)

# Add the Global Model Accuracy line for comparison
global_acc_final = df_global[df_global['round'] == max_round]['accuracy'].iloc[0]
plt.axhline(global_acc_final, color='red', linestyle='--', 
            label=f'Global Accuracy ({global_acc_final:.4f})', linewidth=2)

# Label the bars with their accuracy values (Y-offset added to prevent overlap)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, 
             f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# Create custom legend handles for the color groups (4 clients, 4 unique colors)
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc='darkblue'),
    plt.Rectangle((0, 0), 1, 1, fc='darkgreen'),
    plt.Rectangle((0, 0), 1, 1, fc='orange'),
    plt.Rectangle((0, 0), 1, 1, fc='purple'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)
]
legend_labels = [
    'Client 0 (Cleveland)', 
    'Client 1 (Hungarian)', 
    'Client 2 (Switzerland)', 
    'Client 3 (VA Long Beach)',
    f'Global Accuracy ({global_acc_final:.4f})'
]
    
plt.title(f'Client Performance Heterogeneity (Round {max_round})', fontsize=14)
plt.xlabel('Client ID', fontsize=12)
plt.ylabel('Local Validation Accuracy', fontsize=12)
# Set Y-axis up to 1.1 to accommodate the highest label
plt.ylim(0, 1.1) 
# Move legend outside the plot to the right
plt.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(1.05, 0.5)) 
plt.grid(axis='y', linestyle='--', alpha=0.6)
# Adjust layout to fit the external legend
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.savefig(OUTPUT_FILE)
plt.close()

print(f"Plot saved to {OUTPUT_FILE}")