import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("output/viz", exist_ok=True)

try:
    logs = pd.read_csv("output/training_log.csv")
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot Gradient Norm (Training Proxy)
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Proxy (Gradient Norm)', color=color1)
    ax1.plot(logs['epoch'], logs['gradient_norm'], color=color1, linewidth=2, label='Train')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Plot Validation MSE
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Validation Loss (MSE)', color=color2)
    ax2.plot(logs['epoch'], logs['val_mse'], color=color2, linewidth=2, linestyle='--', label='Val MSE')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("Model Convergence: Training vs. Validation")
    fig.tight_layout()
    plt.savefig("output/viz/convergence_curve.png", dpi=300)
    print("Graph successfully saved to output/viz/convergence_curve.png!")

except Exception as e:
    print(f"Error generating graph: {e}")
    print("Check if output/training_log.csv exists and has 'epoch', 'gradient_norm', and 'val_mse' columns.")