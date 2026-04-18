package dsg;

/**
 * Shared constants for the DSG Inflation Forecasting system.
 * 
 * This file is the single source of truth for all configuration values.
 * All team members import from here — no hardcoded magic numbers elsewhere.
 * 
 * @author Affan Ahmed Basra (476173)
 */
public final class Config {

    private Config() {} // prevent instantiation

      // NETWORK
    

    /** Hostname for the master node (localhost for single-machine simulation). */
    public static final String MASTER_HOST = "localhost";

    /** TCP port the master's ServerSocket binds to. */
    public static final int MASTER_PORT = 5000;

  
    // CLUSTER

    /** Number of worker nodes in the distributed system. */
    public static final int NUM_WORKERS = 4;

    /** Number of data shards (matches pipeline output). */
    public static final int NUM_SHARDS = 4;

    /** Socket read timeout in milliseconds for straggler detection. */
    public static final int STRAGGLER_TIMEOUT_MS = 5000;

    /** Minimum number of gradient responses needed for GC-DC recovery. */
    public static final int MIN_RESPONSES = 3;


    // MODEL
    /** Number of input features (8 economic + 3 lag). */
    public static final int NUM_FEATURES = 11;

    /** Learning rate (η) for gradient descent weight updates. */
    public static final double LEARNING_RATE = 0.01;

    /** Maximum number of training epochs (communication rounds). */
    public static final int MAX_EPOCHS = 100;

    // DATA PATHS (relative to project root)

    /** Directory containing the shard CSV files. */
    public static final String SHARD_DIR = "output/shards/";

    /** Filename prefix for shard files (shard_0.csv, shard_1.csv, ...). */
    public static final String SHARD_PREFIX = "shard_";

    /** Path to the test features CSV. */
    public static final String TEST_X_PATH = "output/X_test.csv";

    /** Path to the test target CSV. */
    public static final String TEST_Y_PATH = "output/y_test.csv";

    /** Path to the scaler parameters JSON (for de-normalization). */
    public static final String SCALER_PATH = "output/scaler_params.json";

    
    // FEATURE NAMES (must match shard CSV column order exactly)

    /** Ordered feature column names as they appear in the shard CSVs. */
    public static final String[] FEATURE_NAMES = {
        "exchange_rate", "gdp_growth", "unemployment", "broad_money",
        "exports", "imports", "oil_rents", "remittances",
        "inflation_lag1", "inflation_lag2", "inflation_lag3"
    };

    /** Target column name in the shard CSVs (always the last column). */
    public static final String TARGET_NAME = "inflation";

    // GC-DC CODING MATRIX — Cyclic Shift Code (s = 1 straggler tolerance)
    //
    // Each row i gives the coding coefficients for Worker i across shards 0..3.
    // Worker i computes: coded_gradient = Σ CODING_MATRIX[i][j] * grad(shard_j)
    //
    // Recovery guarantee: when any 1 worker times out, the remaining 3
    // coded gradients are sufficient to recover the exact full gradient
    // g_full = g₀ + g₁ + g₂ + g₃ via Gaussian elimination.
    //
    // Example: if Worker 3 drops out →
    //   coded_grad₀ + coded_grad₂ = (g₀+g₁) + (g₂+g₃) = g_full  ✓
    //

    /** Coding matrix: CODING_MATRIX[worker][shard] = coefficient. */
    public static final double[][] CODING_MATRIX = {
        {1, 1, 0, 0},   // Worker 0: shard 0 + shard 1
        {0, 1, 1, 0},   // Worker 1: shard 1 + shard 2
        {0, 0, 1, 1},   // Worker 2: shard 2 + shard 3
        {1, 0, 0, 1},   // Worker 3: shard 3 + shard 0
    };

    /**
     * Shard assignments: SHARD_ASSIGNMENTS[worker] = {shardA, shardB}.
     * Derived from CODING_MATRIX (indices where coefficient != 0).
     */
    public static final int[][] SHARD_ASSIGNMENTS = {
        {0, 1},   // Worker 0
        {1, 2},   // Worker 1
        {2, 3},   // Worker 2
        {3, 0},   // Worker 3
    };

    
    // HYBRID THREADING (for Fizza's ThreadedGradient)
   
    /** Number of threads per worker for intra-node parallelism. */
    public static final int THREADS_PER_WORKER = 8;

    // UTILITY

    /**
     * Returns the path to a shard CSV file.
     * @param shardIndex shard number (0..NUM_SHARDS-1)
     * @return relative file path, e.g. "output/shards/shard_0.csv"
     */
    public static String shardPath(int shardIndex) {
        return SHARD_DIR + SHARD_PREFIX + shardIndex + ".csv";
    }
}
