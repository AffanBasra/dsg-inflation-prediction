import os
import re
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File Paths
CONFIG_PATH = "src/dsg/Config.java"
OUT_DIR = "output/viz"
os.makedirs(OUT_DIR, exist_ok=True)

def update_config(param, value):
    """Dynamically updates Config.java parameters using regex."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Matches: public static final int PARAM_NAME = 123;
    pattern = rf"(public\s+static\s+final\s+int\s+{param}\s*=\s*)\d+;"
    new_content = re.sub(pattern, rf"\g<1>{value};", content)
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

def compile_java():
    """Recompiles the Java project."""
    print("   -> Compiling Java source...")
    result = subprocess.run(["javac", "-d", "bin", "src/dsg/*.java"], capture_output=True, text=True)
    if result.returncode != 0:
        print("COMPILATION ERROR:\n", result.stderr)
        exit(1)

def run_java_process(main_class):
    """Runs a Java class and returns execution time in seconds."""
    start_time = time.time()
    # Run the process and wait for it to finish
    subprocess.run(["java", "-cp", "bin", main_class], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - start_time

print("======================================================")
print("   DSG Benchmarking Suite & Graph Generator")
print("======================================================")

# ---------------------------------------------------------
# EXPERIMENT 1: Intra-Node Thread Scaling
# ---------------------------------------------------------
print("\n[1/2] Running Thread Scaling Experiment...")
update_config("MIN_RESPONSES", 3) # Ensure normal GC-DC is on
thread_counts = [1, 2, 4, 8]
thread_results = []

for t in thread_counts:
    print(f"   Testing with {t} Threads per Worker...")
    update_config("THREADS_PER_WORKER", t)
    compile_java()
    
    exec_time = run_java_process("dsg.IntegrationLauncher")
    thread_results.append({"Threads": t, "Execution_Time_sec": round(exec_time, 2)})

# Save CSV
df_threads = pd.DataFrame(thread_results)
df_threads.to_csv(f"{OUT_DIR}/thread_efficiency.csv", index=False)
print(f"   -> Saved {OUT_DIR}/thread_efficiency.csv")

# Plot Thread Scaling
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_threads, x="Threads", y="Execution_Time_sec", marker='o', color='green', linewidth=2)
plt.title("Intra-Node Parallelism: Execution Time vs Threads")
plt.xlabel("Threads per Worker")
plt.ylabel("Execution Time (Seconds)")
plt.ylim(0, max(df_threads["Execution_Time_sec"]) * 1.2)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/thread_scaling_graph.png", dpi=300)

# ---------------------------------------------------------
# EXPERIMENT 2: Optimization (Sequential vs Distributed vs GC-DC)
# ---------------------------------------------------------
print("\n[2/2] Running System Architecture Comparison...")
arch_results = []

# 1. Sequential Baseline
print("   Testing Sequential Baseline...")
compile_java() # Compile current state
seq_time = run_java_process("dsg.Baseline")
arch_results.append({"Architecture": "Sequential\n(1 Process, 1 Thread)", "Time_sec": round(seq_time, 2)})

# 2. Distributed NAIVE (No GC-DC, waits for all 4 workers)
# Setting MIN_RESPONSES = 4 forces the Master to wait for the 5-second straggler
print("   Testing Distributed NAIVE (No Gradient Coding)...")
update_config("THREADS_PER_WORKER", 8)
update_config("MIN_RESPONSES", 4) 
compile_java()
naive_time = run_java_process("dsg.IntegrationLauncher")
arch_results.append({"Architecture": "Distributed Naive\n(Waits for Straggler)", "Time_sec": round(naive_time, 2)})

# 3. Distributed GC-DC (Optimized, requires only 3 workers)
print("   Testing Distributed GC-DC (Straggler Tolerant)...")
update_config("MIN_RESPONSES", 3)
compile_java()
gcdc_time = run_java_process("dsg.IntegrationLauncher")
arch_results.append({"Architecture": "Distributed GC-DC\n(Ignores Straggler)", "Time_sec": round(gcdc_time, 2)})

# Save CSV
df_arch = pd.DataFrame(arch_results)
df_arch.to_csv(f"{OUT_DIR}/gcdc_optimization_impact.csv", index=False)
print(f"   -> Saved {OUT_DIR}/gcdc_optimization_impact.csv")

# Plot Architecture Comparison Bar Chart
plt.figure(figsize=(9, 6))
colors = ['#e74c3c', '#f39c12', '#2ecc71'] # Red, Orange, Green
ax = sns.barplot(data=df_arch, x="Architecture", y="Time_sec", palette=colors)
plt.title("System Performance Comparison (100 Epochs)\nImpact of Distributed Execution & Gradient Coding")
plt.ylabel("Total Execution Time (Seconds)")

# Add text labels on top of bars
for i, v in enumerate(df_arch["Time_sec"]):
    ax.text(i, v + (max(df_arch["Time_sec"])*0.02), f"{v}s", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/architecture_comparison.png", dpi=300)

# ---------------------------------------------------------
# Cleanup & Restore
# ---------------------------------------------------------
# Restore Config.java to default expected state
update_config("THREADS_PER_WORKER", 8)
update_config("MIN_RESPONSES", 3)
compile_java()

print("\n======================================================")
print("Benchmarking Complete! All CSVs and Graphs are in 'output/viz/'")
print("======================================================")