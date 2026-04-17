# DSG Inflation Prediction — Distributed Subgradient with Gradient Coding

A parallel and distributed computing project that uses **Distributed Subgradient (DSG)** optimization with **Gradient Coding for Distributed Computing (GC-DC)** to predict Pakistan's inflation rate from 11 macroeconomic indicators.

## Team

| Member | ID | Role | Deliverables |
|--------|-----|------|-------------|
| Affan Ahmed Basra | 476173 | Networking, Protocol & Integration | `Config.java`, `MessageProtocol.java`, `SocketManager.java`, `IntegrationLauncher.java`, `StragglerInjector.java` |
| Rimsha Mahmood | 455080 | Master, GC-DC Math & Evaluation | `Master.java`, `GaussianElimination.java`, `Baseline.java`, `Evaluation.java` |
| Fizza Kashif | 466184 | Workers, Hybrid Parallelism & Core Logic | `Worker.java`, `GradientComputer.java`, `ThreadedGradient.java` |

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    IntegrationLauncher                         │
│               (spawns 1 Master + 4 Worker JVMs)               │
└──────┬───────────────────────────────────────────┬────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐        TCP/Serialization    ┌──────────────┐
│    Master    │◄───────────────────────────►│   Worker 0   │
│              │                             │  (shard 0+1) │
│  - Barrier   │  InitMsg (shard data)       ├──────────────┤
│    Sync      │  BroadcastMsg (weights)     │   Worker 1   │
│              │  GradReturnMsg (gradient)   │  (shard 1+2) │
│  - GC-DC     │                             ├──────────────┤
│    Recovery  │  Timeout: 5 seconds         │   Worker 2   │
│              │  Min responses: 3/4         │  (shard 2+3) │
│  - Weight    │                             ├──────────────┤
│    Update    │                             │   Worker 3   │
│              │                             │  (shard 3+0) │
│              │                             │  ⚠ STRAGGLER │
└──────────────┘                             └──────────────┘
```

## Dataset

- **Source**: Pakistan Economic Indicators (1986–2021)
- **Features (11)**: `exchange_rate`, `gdp_growth`, `unemployment`, `broad_money`, `exports`, `imports`, `oil_rents`, `remittances`, `inflation_lag1`, `inflation_lag2`, `inflation_lag3`
- **Target**: `inflation` (CPI)
- **Scale**: ~50,000 synthetic samples (36 annual → 421 monthly → 50K via Gaussian jittering)
- **Shards**: 4 × ~9,854 rows (Z-score normalized, temporal train/test split)

## GC-DC Coding Scheme

Cyclic shift code with **s = 1** straggler tolerance:

| Worker | Assigned Shards | Coded Gradient |
|--------|----------------|----------------|
| 0 | shard 0 + shard 1 | g₀ + g₁ |
| 1 | shard 1 + shard 2 | g₁ + g₂ |
| 2 | shard 2 + shard 3 | g₂ + g₃ |
| 3 ⚠ | shard 3 + shard 0 | g₃ + g₀ |

**Recovery examples** (when 1 worker times out):
- Worker 3 drops → `coded₀ + coded₂ = (g₀+g₁) + (g₂+g₃) = g_full` ✓
- Worker 0 drops → `coded₁ + coded₃ = (g₁+g₂) + (g₃+g₀) = g_full` ✓

## Quick Start

### Prerequisites

- **Java 11+** (JDK) — for the distributed system
- **Python 3.x** with `polars`, `scipy`, `numpy` — for the data pipeline only

### 1. Generate Data (already done)

```bash
python data_pipeline.py
```

This produces `output/shards/shard_0..3.csv`, `output/X_test.csv`, `output/y_test.csv`, and `output/scaler_params.json`.

### 2. Compile

```bash
compile.bat
```

### 3. Run

```bash
run.bat
```

This launches 1 Master JVM + 4 Worker JVMs via `IntegrationLauncher`. Workers connect to the Master via TCP, receive shard data, and begin distributed training.

## Project Structure

```
├── data_pipeline.py               # Python data pipeline (generates shards)
├── output/                        # Pipeline outputs
│   ├── shards/shard_0..3.csv      # 4 worker shards (~9,854 rows each)
│   ├── X_test.csv, y_test.csv     # Test data (9,912 rows)
│   └── scaler_params.json         # Z-score normalization parameters
├── src/dsg/                       # Java source code
│   ├── Config.java                # Shared constants & coding matrix
│   ├── MessageProtocol.java       # Serializable message definitions
│   ├── SocketManager.java         # TCP socket management
│   ├── CsvLoader.java             # CSV parsing utility
│   ├── IntegrationLauncher.java   # One-click multi-JVM launcher
│   ├── StragglerInjector.java     # Straggler delay simulation
│   ├── Master.java                # Master node (Rimsha)
│   ├── Worker.java                # Worker node (Fizza)
│   ├── GaussianElimination.java   # GC-DC gradient recovery (Rimsha)
│   ├── GradientComputer.java      # MSE gradient computation (Fizza)
│   ├── ThreadedGradient.java      # Multi-threaded gradient (Fizza)
│   ├── Baseline.java              # Sequential baseline (Rimsha)
│   └── Evaluation.java            # Metrics: MSE, MAE, R² (Rimsha)
├── compile.bat                    # One-click compile
├── run.bat                        # One-click run
└── README.md
```

## Git Workflow

| Branch | Owner | Purpose | Merge Deadline |
|--------|-------|---------|---------------|
| `main` | Affan | Protocol contract + interfaces | Apr 18 |
| `networking-protocol` | Affan | Sockets, integration, mocks | Apr 23 |
| `workers-threading` | Fizza | Worker logic, gradient computation, threading | Apr 24 |
| `master-eval` | Rimsha | Master logic, GC-DC math, evaluation | Apr 25 |

### Timeline

1. **Day 1 (Apr 18)**: Affan pushes contract (`Config`, `MessageProtocol`, stubs) to `main`. Everyone pulls.
2. **Days 2–5 (Apr 19–23)**: Parallel development on separate branches.
3. **Day 6 (Apr 24)**: Merge `workers-threading` + `networking-protocol`. Test Master → Workers.
4. **Day 7 (Apr 25)**: Merge `master-eval`. Full GC-DC integration test.
5. **Days 8–9 (Apr 26–27)**: Report, complexity analysis, demo prep.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Pipeline | Python (Polars, SciPy, NumPy) |
| Distributed System | Java 11+ (plain JDK, no frameworks) |
| Networking | TCP sockets (`java.net.ServerSocket/Socket`) |
| Serialization | Java `ObjectOutputStream/ObjectInputStream` |
| Intra-Node Parallelism | `java.util.concurrent.ExecutorService` |
| Build | `javac` (no Maven/Gradle) |

## License

Academic project — FAST-NUCES, Spring 2026.
