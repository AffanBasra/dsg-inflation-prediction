package dsg;

/**
 * Sequential (single-threaded) baseline for comparison.
 * 
 * Responsibilities (Rimsha Mahmood — 455080):
 * 
 * Trains linear regression on the FULL dataset (all 4 shards combined)
 * using single-threaded gradient descent. No distribution, no coding.
 * 
 * This establishes the accuracy benchmark:
 *   - If DSG matches Baseline's MSE/MAE/R², the distributed system is correct.
 *   - The speedup = Baseline_time / DSG_time demonstrates parallelism benefit.
 */
public class Baseline {

    public static void main(String[] args) {
        System.out.println("[Baseline] Starting sequential training...");

        // TODO: Rimsha — implement sequential baseline
        //
        // 1. Load all 4 shards and concatenate:
        //    CsvLoader.CsvData shard0 = CsvLoader.loadShard(Config.shardPath(0));
        //    ... concatenate X and y from all shards ...
        //
        // 2. Initialize weights = zeros(NUM_FEATURES + 1)
        //
        // 3. Gradient descent loop (MAX_EPOCHS):
        //    gradient = GradientComputer.computeGradient(X_all, y_all, weights);
        //    weights[j] -= LEARNING_RATE * gradient[j];
        //
        // 4. Load test set:
        //    double[][] X_test = CsvLoader.loadFeatures(Config.TEST_X_PATH);
        //    double[] y_test = CsvLoader.loadTargets(Config.TEST_Y_PATH);
        //
        // 5. Evaluate and print MSE, MAE, R²

        System.out.println("[Baseline] Not yet implemented — awaiting Rimsha's code.");
    }
}
