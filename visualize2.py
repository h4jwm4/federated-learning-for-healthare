import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
GLOBAL_METRICS_FILE = "./global_metrics.csv"
CENTRALIZED_METRICS_FILE = "./centralized_baseline_metrics.csv"
OUTPUT_PLOT_FILE = "./final_accuracy_comparison_plot.png"

def generate_comparison_plot():
    # Load Data
    try:
        df_global = pd.read_csv(GLOBAL_METRICS_FILE)
        df_central = pd.read_csv(CENTRALIZED_METRICS_FILE)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please upload {e.filename}.")
        return

    # Extract Final FL Accuracy (last row of FL logs)
    final_fl_acc = df_global['accuracy'].iloc[-1]
    
    # Extract Final Centralized Accuracy (last row of centralized logs)
    final_central_acc = df_central['test_accuracy'].iloc[-1]

    # Prepare data for plotting
    models = ['Centralized Baseline', 'Federated Learning (FL)']
    accuracies = [final_central_acc, final_fl_acc]
    colors = ['#4CAF50', '#2196F3'] 

    # Create the Comparison Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=colors, width=0.5)

    # Label the bars with their accuracy values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, 
                 f'{yval:.4f}', ha='center', va='bottom', fontsize=12)

    plt.title('Performance Comparison: Centralized vs. Federated Model', fontsize=14)
    plt.xlabel('Training Methodology', fontsize=12)
    plt.ylabel('Final Test Accuracy', fontsize=12)
    plt.ylim(0, 1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the key takeaway
    difference = final_central_acc - final_fl_acc
    plt.text(0.5, 0.95, 
             f"Difference: {difference:.4f}", 
             fontsize=12, color='darkred', ha='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE)
    plt.close()

    print(f"Comparison plot saved as: {OUTPUT_PLOT_FILE}")

if __name__ == '__main__':
    generate_comparison_plot()