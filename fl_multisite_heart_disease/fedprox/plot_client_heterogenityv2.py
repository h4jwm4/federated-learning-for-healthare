import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
GLOBAL_METRICS_FILE = "./global_metrics.csv"
CLIENT_METRICS_FILE = "./client_metrics.csv"
OUTPUT_FILE = 'client_heterogeneity_plotv2.png'

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

# 3. Define the custom color mapping function
def get_client_color(cid):
    # Mapping based on client ranges (which correspond to locations)
    if 0 <= cid <= 2:
        return 'darkblue'  # Cleveland
    elif 3 <= cid <= 5:
        return 'darkgreen'  # Hungarian
    elif 6 <= cid <= 8:
        return 'orange'  # Switzerland
    elif 9 <= cid <= 11:
        return 'purple'  # VA Long Beach
    else:
        return 'gray'

# Apply the color mapping to the DataFrame
colors = df_final_round['cid'].apply(get_client_color)

# 4. Create the Bar Plot
plt.figure(figsize=(15, 6))
    
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
             f'{yval:.4f}', ha='center', va='bottom', fontsize=9)

# Create custom legend handles for the color groups
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, fc='darkblue'),
    plt.Rectangle((0, 0), 1, 1, fc='darkgreen'),
    plt.Rectangle((0, 0), 1, 1, fc='orange'),
    plt.Rectangle((0, 0), 1, 1, fc='purple'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2)
]
legend_labels = [
    'Cleveland (C0-C2)', 
    'Hungarian (C3-C5)', 
    'Switzerland (C6-C8)', 
    'VA Long Beach (C9-C11)',
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