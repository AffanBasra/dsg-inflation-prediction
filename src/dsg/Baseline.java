package dsg;

import java.io.IOException;

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
        try {
            CsvLoader.CsvData[] shards = new CsvLoader.CsvData[Config.NUM_SHARDS];
            int totalRows = 0;
            for (int shardIndex = 0; shardIndex < Config.NUM_SHARDS; shardIndex++) {
                shards[shardIndex] = CsvLoader.loadShard(Config.shardPath(shardIndex));
                totalRows += shards[shardIndex].getNumRows();
            }

            double[][] X = new double[totalRows][Config.NUM_FEATURES];
            double[] y = new double[totalRows];
            int index = 0;
            for (CsvLoader.CsvData shard : shards) {
                double[][] shardX = shard.getX();
                double[] shardY = shard.getY();
                if (shardX.length != shardY.length) {
                    throw new IllegalStateException("Shard row count and target count do not match");
                }
                for (int row = 0; row < shardX.length; row++, index++) {
                    if (shardX[row].length != Config.NUM_FEATURES) {
                        throw new IllegalStateException("Unexpected feature count in shard row: "
                            + shardX[row].length);
                    }
                    X[index] = shardX[row];
                    y[index] = shardY[row];
                }
            }

            double[] weights = new double[Config.NUM_FEATURES + 1];
            long startTime = System.currentTimeMillis();
            for (int epoch = 0; epoch < Config.MAX_EPOCHS; epoch++) {
                double[] gradient = GradientComputer.computeGradient(X, y, weights);
                for (int j = 0; j < weights.length; j++) {
                    weights[j] -= Config.LEARNING_RATE * gradient[j];
                }
            }
            long durationMs = System.currentTimeMillis() - startTime;

            double[][] xTest = CsvLoader.loadFeatures(Config.TEST_X_PATH);
            double[] yTest = CsvLoader.loadTargets(Config.TEST_Y_PATH);
            if (xTest.length != yTest.length) {
                throw new IllegalStateException("Test feature count and target count do not match");
            }

            double[] predictions = predict(xTest, weights);
            double mse = Evaluation.mse(predictions, yTest);
            double mae = Evaluation.mae(predictions, yTest);
            double r2 = Evaluation.r2(predictions, yTest);

            System.out.println("[Baseline] Training complete.");
            System.out.println("[Baseline] Runtime: " + durationMs + " ms");
            System.out.printf("[Baseline] Test MSE = %.6f%n", mse);
            System.out.printf("[Baseline] Test MAE = %.6f%n", mae);
            System.out.printf("[Baseline] Test R² = %.6f%n", r2);
        } catch (IOException e) {
            System.err.println("[Baseline] I/O error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static double[] predict(double[][] X, double[] weights) {
        double[] predictions = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double value = weights[Config.NUM_FEATURES];
            for (int j = 0; j < Config.NUM_FEATURES; j++) {
                value += X[i][j] * weights[j];
            }
            predictions[i] = value;
        }
        return predictions;
    }
}
