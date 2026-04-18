package dsg;

import java.io.IOException;
import java.util.List;

/**
 * Master Node — Central coordinator for DSG optimization.
 * 
 * Responsibilities (Rimsha Mahmood — 455080):
 * 
 * 1. Load all 4 shard CSV files via CsvLoader.loadShard()
 * 2. Create SocketManager.MasterServer and accept worker connections
 * 3. Send InitMsg to each worker with their assigned shard data + coding coefficients
 * 4. Training loop (MAX_EPOCHS iterations):
 *    a. Broadcast current weights via BroadcastMsg
 *    b. Collect coded gradients via collectGradients(STRAGGLER_TIMEOUT_MS)
 *    c. If fewer than MIN_RESPONSES received → handle error
 *    d. Recover full gradient via GaussianElimination.recoverFullGradient()
 *    e. Update weights: θ ← θ - LEARNING_RATE · ∇
 * 5. Evaluate final model on test set using Evaluation class
 * 
 * Available networking API (from Affan's SocketManager):
 *   - server.acceptWorkers()
 *   - server.sendInit(workerId, initMsg)
 *   - server.broadcast(broadcastMsg)
 *   - server.collectGradients(timeoutMs)  → List<GradReturnMsg>
 *   - server.close()
 */
public class Master {

    public static void main(String[] args) {
        System.out.println("[Master] Starting...");
        SocketManager.MasterServer server = null;

        try {
            CsvLoader.CsvData[] shards = new CsvLoader.CsvData[Config.NUM_SHARDS];
            for (int shardIndex = 0; shardIndex < Config.NUM_SHARDS; shardIndex++) {
                shards[shardIndex] = CsvLoader.loadShard(Config.shardPath(shardIndex));
            }

            server = new SocketManager.MasterServer();
            server.acceptWorkers();

            double[] weights = new double[Config.NUM_FEATURES + 1];
            initializeWorkers(server, shards, weights);

            for (int epoch = 0; epoch < Config.MAX_EPOCHS; epoch++) {
                System.out.println("[Master] Epoch " + epoch + " broadcasting weights...");
                server.broadcast(new MessageProtocol.BroadcastMsg(weights, epoch));

                List<MessageProtocol.GradReturnMsg> responses =
                    server.collectGradients(Config.STRAGGLER_TIMEOUT_MS);
                System.out.println("[Master] Epoch " + epoch + " received "
                    + responses.size() + " gradient response(s).");

                if (responses.size() < Config.MIN_RESPONSES) {
                    System.err.println("[Master] Error: insufficient gradient responses ("
                        + responses.size() + ") — required at least " + Config.MIN_RESPONSES);
                    break;
                }

                int[] respondedWorkerIds = new int[responses.size()];
                double[][] codedGradients = new double[responses.size()][];
                boolean[] seenWorker = new boolean[Config.NUM_WORKERS];

                for (int i = 0; i < responses.size(); i++) {
                    MessageProtocol.GradReturnMsg response = responses.get(i);
                    int workerId = response.getWorkerId();
                    if (workerId < 0 || workerId >= Config.NUM_WORKERS) {
                        throw new IllegalStateException("Received invalid worker ID: " + workerId);
                    }
                    if (seenWorker[workerId]) {
                        throw new IllegalStateException("Duplicate gradient response from worker " + workerId);
                    }
                    seenWorker[workerId] = true;
                    respondedWorkerIds[i] = workerId;
                    codedGradients[i] = response.getCodedGradient();
                }

                double[] fullGradient = GaussianElimination.recoverFullGradient(
                    respondedWorkerIds, codedGradients);

                double gradientNorm = 0.0;
                for (int j = 0; j < weights.length; j++) {
                    double update = Config.LEARNING_RATE * fullGradient[j];
                    weights[j] -= update;
                    gradientNorm += fullGradient[j] * fullGradient[j];
                }
                gradientNorm = Math.sqrt(gradientNorm);

                if (gradientNorm < 1e-6) {
                    System.out.println("[Master] Convergence threshold reached at epoch "
                        + epoch + ", gradient norm = " + gradientNorm);
                    break;
                }
            }

            evaluateModel(weights);
        } catch (IOException e) {
            System.err.println("[Master] I/O error: " + e.getMessage());
            e.printStackTrace();
        } catch (RuntimeException e) {
            System.err.println("[Master] Runtime error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (server != null) {
                try {
                    server.close();
                } catch (IOException e) {
                    System.err.println("[Master] Failed to close server: " + e.getMessage());
                }
            }
        }
    }

    private static void initializeWorkers(SocketManager.MasterServer server,
                                          CsvLoader.CsvData[] shards,
                                          double[] initialWeights) throws IOException {
        for (int workerId = 0; workerId < Config.NUM_WORKERS; workerId++) {
            int[] assignment = Config.SHARD_ASSIGNMENTS[workerId];
            if (assignment == null || assignment.length != 2) {
                throw new IllegalStateException("Invalid shard assignment for worker " + workerId);
            }

            int shardA = assignment[0];
            int shardB = assignment[1];
            double shardACoeff = Config.CODING_MATRIX[workerId][shardA];
            double shardBCoeff = Config.CODING_MATRIX[workerId][shardB];

            MessageProtocol.InitMsg initMsg = new MessageProtocol.InitMsg(
                workerId,
                shards[shardA].getX(), shards[shardA].getY(),
                shards[shardB].getX(), shards[shardB].getY(),
                shardACoeff, shardBCoeff,
                initialWeights);

            System.out.println("[Master] Sending InitMsg to worker " + workerId
                + " (shards=" + shardA + "," + shardB + ")");
            server.sendInit(workerId, initMsg);
        }
    }

    private static void evaluateModel(double[] weights) throws IOException {
        System.out.println("[Master] Evaluating final model...");
        double[][] xTest = CsvLoader.loadFeatures(Config.TEST_X_PATH);
        double[] yTest = CsvLoader.loadTargets(Config.TEST_Y_PATH);

        if (xTest.length != yTest.length) {
            throw new IllegalStateException("Test feature count and target count do not match");
        }

        double[] predictions = predict(xTest, weights);
        double mse = Evaluation.mse(predictions, yTest);
        double mae = Evaluation.mae(predictions, yTest);
        double r2 = Evaluation.r2(predictions, yTest);

        System.out.println("[Master] Final evaluation complete.");
        System.out.printf("[Master] Test MSE = %.6f%n", mse);
        System.out.printf("[Master] Test MAE = %.6f%n", mae);
        System.out.printf("[Master] Test R² = %.6f%n", r2);
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
