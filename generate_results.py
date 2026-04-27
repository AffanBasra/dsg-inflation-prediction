import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("output/viz", exist_ok=True)
print("--- Generating Python Analytics from Java Outputs ---")

try:
    preds = pd.read_csv("output/predictions.csv")
    logs = pd.read_csv("output/training_log.csv")
except FileNotFoundError:
    print("Error: Run the Java Master first to generate CSV outputs!")
    exit()

THRESHOLD = preds['actual'].mean()
preds['actual_class'] = (preds['actual'] > THRESHOLD).astype(int)
preds['pred_class'] = (preds['predicted'] > THRESHOLD).astype(int)

acc = accuracy_score(preds['actual_class'], preds['pred_class'])
prec = precision_score(preds['actual_class'], preds['pred_class'], zero_division=0)
rec = recall_score(preds['actual_class'], preds['pred_class'], zero_division=0)
f1 = f1_score(preds['actual_class'], preds['pred_class'], zero_division=0)

pd.DataFrame([{"Accuracy": acc, "Precision": prec, "Recall": rec, "F1_Score": f1, "Threshold": THRESHOLD}]).to_csv("output/viz/final_metrics.csv", index=False)

sns.set_theme(style="darkgrid")

# Graph 1: Training vs Validation Curve (Dual Axis)
fig, ax1 = plt.subplots(figsize=(8, 5))
color1 = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Proxy (Gradient Norm)', color=color1)
ax1.plot(logs['epoch'], logs['gradient_norm'], color=color1, linewidth=2, label='Train (Grad Norm)')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel('Validation Loss (MSE)', color=color2)
ax2.plot(logs['epoch'], logs['val_mse'], color=color2, linewidth=2, linestyle='--', label='Val MSE')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title("Model Convergence: Training vs. Validation Loss")
fig.tight_layout()
plt.savefig("output/viz/convergence_curve.png", dpi=300)

# Graph 2: Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(preds['actual'], preds['predicted'], alpha=0.5, color='purple')
min_val = min(preds['actual'].min(), preds['predicted'].min())
max_val = max(preds['actual'].max(), preds['predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title("Actual vs Predicted Inflation (Normalized)")
plt.xlabel("Actual Inflation")
plt.ylabel("Predicted Inflation")
plt.tight_layout()
plt.savefig("output/viz/actual_vs_predicted.png", dpi=300)

print("\nVisualizations saved to 'output/viz/' folder.")