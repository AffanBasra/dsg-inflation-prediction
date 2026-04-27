import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("output/viz", exist_ok=True)

# 1. Load the Thread Scaling Data
try:
    df = pd.read_csv("output/viz/thread_efficiency.csv")
except FileNotFoundError:
    print("Error: Run benchmark_suite.py first!")
    exit()

# 2. Calculate Measured Speedup
t1 = df.loc[df['Threads'] == 1, 'Execution_Time_sec'].values[0]
df['Measured_Speedup'] = t1 / df['Execution_Time_sec']

# 3. Calculate Parallel Fraction (f) using Amdahl's Law formula for p=8
# Amdahl's Law: Speedup = 1 / ((1 - f) + (f / p))
# Solving for f: f = (p / (p - 1)) * (1 - (1 / Speedup))
p_max = 8
s_max = df.loc[df['Threads'] == p_max, 'Measured_Speedup'].values[0]
f = (p_max / (p_max - 1)) * (1 - (1 / s_max))
print(f"Calculated Parallel Fraction (f): {f:.4f} ({f*100:.2f}%)")
print(f"Sequential Fraction (1 - f): {1-f:.4f} ({(1-f)*100:.2f}%)")

# 4. Generate Theoretical Data
p_values = np.linspace(1, 8, 100)
ideal_speedup = p_values
amdahls_speedup = 1 / ((1 - f) + (f / p_values))

# 5. Plot the Graph
plt.figure(figsize=(9, 6))
sns.set_theme(style="whitegrid")

# Plot Ideal Linear Speedup
plt.plot(p_values, ideal_speedup, 'k--', label='Ideal Speedup (Linear)', alpha=0.6)

# Plot Amdahl's Theoretical Limit
plt.plot(p_values, amdahls_speedup, 'b-', label=f"Amdahl's Limit (f = {f:.2f})", linewidth=2)

# Plot Actual Measured Speedup
plt.plot(df['Threads'], df['Measured_Speedup'], 'ro', markersize=8, label='Measured Speedup')
plt.plot(df['Threads'], df['Measured_Speedup'], 'r-', alpha=0.5)

plt.title("Amdahl's Law & Scaling Analysis\nTheoretical vs. Measured Speedup")
plt.xlabel("Number of Threads (P)")
plt.ylabel("Speedup Factor (S)")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig("output/viz/amdahls_law_scaling.png", dpi=300)
print("Saved theoretical graph to output/viz/amdahls_law_scaling.png")