import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
GLOBAL_METRICS_FILE = "./global_metrics.csv"
CLIENT_METRICS_FILE = "./client_metrics.csv"
FINAL_ROUND_CLIENT_COUNT = 12 # Adjust if your client count changes

def generate_fl_plots():
    # 1. Load Data
    try:
        df_global = pd.read_csv(GLOBAL_METRICS_FILE)
        df_client = pd.read_csv(CLIENT_METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure both CSV files are in the current directory.")
        return

    # --- Data Cleaning and Aggregation ---
    
    # Filter client data to only include evaluation metrics (local test set)
    df_client_eval = df_client[df_client['phase'] == 'evaluate'].copy()
    
    # Calculate the average client accuracy for each round
    df_avg_client_acc = df_client_eval.groupby('round')['accuracy'].mean().reset_index()
    df_avg_client_acc.rename(columns={'accuracy': 'avg_client_accuracy'}, inplace=True)
    
    # Find the maximum round for final round analysis
    max_round = df_global['round'].max()
    print(f"Data loaded successfully. Max round found: {max_round}")

    # ====================================================================
    # PLOT 1: Global Convergence vs. Average Client Performance
    # ====================================================================
    
    # Merge global metrics and average client metrics
    df_combined = pd.merge(df_global, df_avg_client_acc, on='round', how='inner')
    
    plt.figure(figsize=(10, 6))
    
    # Plot Global Accuracy
    plt.plot(df_combined['round'], df_combined['accuracy'], 
             label='Global Model Accuracy (FedProx)', 
             marker='o', linestyle='-', color='blue')
    
    # Plot Average Client Accuracy
    plt.plot(df_combined['round'], df_combined['avg_client_accuracy'], 
             label='Average Local Client Accuracy', 
             marker='s', linestyle='--', color='red', alpha=0.7)
    
    plt.title('FL Convergence: Global Model vs. Average Client Accuracy Over Rounds', fontsize=14)
    plt.xlabel('Server Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.xticks(np.arange(0, max_round + 1, 2)) # Show every 2nd round
    plt.tight_layout()
    plt.savefig('global_convergence_plotv2.png')
    plt.close()
    
    print("Generated plot: global_convergence_plotv2.png")


if __name__ == '__main__':
    generate_fl_plots()