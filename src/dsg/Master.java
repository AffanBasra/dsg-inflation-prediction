package dsg;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

public class Master {

    public static void main(String[] args) {
        System.out.println("[Master] Starting...");
        SocketManager.MasterServer server = null;

        try {
            // 1. OPEN THE PORT FIRST (So workers don't get "Connection refused")
            server = new SocketManager.MasterServer();

            // 2. THEN LOAD THE MASSIVE DATASET
            System.out.println("[Master] Loading 2 million rows into memory (this will take a moment)...");
            CsvLoader.CsvData[] shards = new CsvLoader.CsvData[Config.NUM_SHARDS];
            for (int shardIndex = 0; shardIndex < Config.NUM_SHARDS; shardIndex++) {
                shards[shardIndex] = CsvLoader.loadShard(Config.shardPath(shardIndex));
            }

            // 3. ACCEPT THE WAITING WORKERS
            server.acceptWorkers();

            double[] weights = new double[Config.NUM_FEATURES + 1];
            initializeWorkers(server, shards, weights);

            // Setup for Early Stopping & Regularization
            double[] bestWeights = new double[weights.length];
            double bestLoss = Double.MAX_VALUE;
            int patienceCounter = 0;
            final int PATIENCE = 20;
            final double LAMBDA = 0.01;

            // --- LOAD TEST DATA HERE TO CALCULATE VAL_MSE DURING TRAINING ---
            double[][] xTest = CsvLoader.loadFeatures(Config.TEST_X_PATH);
            double[] yTest = CsvLoader.loadTargets(Config.TEST_Y_PATH);

            try (PrintWriter logWriter = new PrintWriter(new FileWriter("output/training_log.csv"))) {
                logWriter.println("epoch,gradient_norm,val_mse"); // Added val_mse

                for (int epoch = 0; epoch < Config.MAX_EPOCHS; epoch++) {
                    server.broadcast(new MessageProtocol.BroadcastMsg(weights, epoch));

                    List<MessageProtocol.GradReturnMsg> responses = server.collectGradients(Config.STRAGGLER_TIMEOUT_MS);

                    if (responses.size() < Config.MIN_RESPONSES) {
                        System.err.println("[Master] Error: insufficient gradient responses.");
                        break;
                    }

                    int[] respondedWorkerIds = new int[responses.size()];
                    double[][] codedGradients = new double[responses.size()][];
                    boolean[] seenWorker = new boolean[Config.NUM_WORKERS];

                    for (int i = 0; i < responses.size(); i++) {
                        MessageProtocol.GradReturnMsg response = responses.get(i);
                        int workerId = response.getWorkerId();
                        seenWorker[workerId] = true;
                        respondedWorkerIds[i] = workerId;
                        codedGradients[i] = response.getCodedGradient();
                    }

                    double[] fullGradient = GaussianElimination.recoverFullGradient(respondedWorkerIds, codedGradients);

                    double gradientNorm = 0.0;
                    for (int j = 0; j < weights.length; j++) {
                        double regularizationTerm = (j < Config.NUM_FEATURES) ? (LAMBDA * weights[j]) : 0.0;
                        double update = Config.LEARNING_RATE * (fullGradient[j] + regularizationTerm);
                        weights[j] -= update;
                        gradientNorm += fullGradient[j] * fullGradient[j];
                    }
                    gradientNorm = Math.sqrt(gradientNorm);

                    // Calculate current Validation MSE
                    double[] currentPreds = predict(xTest, weights);
                    double valMse = Evaluation.mse(currentPreds, yTest);

                    // Log to CSV
                    logWriter.println(epoch + "," + gradientNorm + "," + valMse);

                    // Early Stopping Check
                    if (gradientNorm < bestLoss) {
                        bestLoss = gradientNorm;
                        System.arraycopy(weights, 0, bestWeights, 0, weights.length);
                        patienceCounter = 0;
                    } else {
                        patienceCounter++;
                        if (patienceCounter >= PATIENCE) {
                            System.out.println("[Master] Early stopping triggered at epoch " + epoch);
                            break;
                        }
                    }

                    if (gradientNorm < 1e-6) {
                        break;
                    }
                }
            }

            evaluateAndExportModel(bestWeights, xTest, yTest);

        } catch (Exception e) {
            System.err.println("[Master] Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (server != null) {
                try {
                    server.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    private static void initializeWorkers(SocketManager.MasterServer server, CsvLoader.CsvData[] shards, double[] initialWeights) throws IOException {
        for (int workerId = 0; workerId < Config.NUM_WORKERS; workerId++) {
            int shardA = Config.SHARD_ASSIGNMENTS[workerId][0];
            int shardB = Config.SHARD_ASSIGNMENTS[workerId][1];
            MessageProtocol.InitMsg initMsg = new MessageProtocol.InitMsg(
                    workerId, shards[shardA].getX(), shards[shardA].getY(),
                    shards[shardB].getX(), shards[shardB].getY(),
                    Config.CODING_MATRIX[workerId][shardA], Config.CODING_MATRIX[workerId][shardB],
                    initialWeights);
            server.sendInit(workerId, initMsg);
        }
    }

    private static void evaluateAndExportModel(double[] weights, double[][] xTest, double[] yTest) throws IOException {
        double[] predictions = predict(xTest, weights);
        System.out.printf("[Master] Final Test MSE = %.6f%n", Evaluation.mse(predictions, yTest));

        try (PrintWriter pw = new PrintWriter(new FileWriter("output/predictions.csv"))) {
            pw.println("actual,predicted");
            for (int i = 0; i < yTest.length; i++) {
                pw.println(yTest[i] + "," + predictions[i]);
            }
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
