package dsg;

/**
 * GC-DC Recovery Engine — Recovers the full gradient from coded responses.
 *
 * Responsibilities (Rimsha Mahmood — 455080):
 *
 * When a straggler worker times out, only MIN_RESPONSES (3 out of 4)
 * coded gradients are available. This class uses the coding matrix
 * structure to recover the exact full gradient g₀ + g₁ + g₂ + g₃.
 *
 * Mathematical approach:
 *   1. Extract the responding workers' rows from Config.CODING_MATRIX → C_sub (3×4)
 *   2. Find recovery weights w such that wᵀ · C_sub = [1,1,1,1]
 *      (i.e., the linear combination that sums all shard gradients)
 *   3. Full gradient = w₁·coded_grad₁ + w₂·coded_grad₂ + w₃·coded_grad₃
 *
 * For the cyclic shift code, the recovery is particularly simple:
 *   - Drop Worker 0: full = coded₁ + coded₃ = (g₁+g₂)+(g₃+g₀)
 *   - Drop Worker 1: full = coded₀ + coded₂ = (g₀+g₁)+(g₂+g₃)
 *   - Drop Worker 2: full = coded₁ + coded₃ = (g₁+g₂)+(g₃+g₀)
 *   - Drop Worker 3: full = coded₀ + coded₂ = (g₀+g₁)+(g₂+g₃)
 *
 * But the implementation should be GENERAL (Gaussian elimination on C_sub)
 * so it works with any valid coding matrix.
 */
public class GaussianElimination {

    /**
     * Recover the full gradient from the coded gradient responses.
     *
     * @param respondedWorkerIds  worker IDs that responded in time (length >= MIN_RESPONSES)
     * @param codedGradients      their coded gradient vectors [numResponded][gradientDim]
     * @return the recovered full gradient vector [gradientDim]
     * @throws IllegalArgumentException if the input is invalid or recovery is impossible
     */
    public static double[] recoverFullGradient(int[] respondedWorkerIds, double[][] codedGradients) {
        if (respondedWorkerIds == null) {
            throw new IllegalArgumentException("respondedWorkerIds must not be null");
        }
        if (codedGradients == null) {
            throw new IllegalArgumentException("codedGradients must not be null");
        }
        int numResponded = respondedWorkerIds.length;
        if (numResponded < Config.MIN_RESPONSES) {
            throw new IllegalArgumentException(String.format(
                "At least %d worker responses are required, but got %d",
                Config.MIN_RESPONSES, numResponded));
        }
        if (numResponded != codedGradients.length) {
            throw new IllegalArgumentException(String.format(
                "respondedWorkerIds length (%d) must match codedGradients length (%d)",
                numResponded, codedGradients.length));
        }

        int gradientDim = -1;
        boolean[] seenWorker = new boolean[Config.NUM_WORKERS];
        for (int i = 0; i < numResponded; i++) {
            int workerId = respondedWorkerIds[i];
            if (workerId < 0 || workerId >= Config.NUM_WORKERS) {
                throw new IllegalArgumentException("Invalid worker ID: " + workerId);
            }
            if (seenWorker[workerId]) {
                throw new IllegalArgumentException("Duplicate worker response for worker ID: " + workerId);
            }
            seenWorker[workerId] = true;

            if (codedGradients[i] == null) {
                throw new IllegalArgumentException("codedGradients[" + i + "] must not be null");
            }
            if (gradientDim < 0) {
                gradientDim = codedGradients[i].length;
                if (gradientDim == 0) {
                    throw new IllegalArgumentException("Gradient vectors must not be empty");
                }
            } else if (codedGradients[i].length != gradientDim) {
                throw new IllegalArgumentException(String.format(
                    "All coded gradients must have the same length: expected %d but got %d",
                    gradientDim, codedGradients[i].length));
            }
        }

        int numShards = Config.NUM_SHARDS;
        double[][] codingSubmatrix = new double[numResponded][numShards];
        for (int i = 0; i < numResponded; i++) {
            int workerId = respondedWorkerIds[i];
            double[] codingRow = Config.CODING_MATRIX[workerId];
            if (codingRow == null || codingRow.length != numShards) {
                throw new IllegalArgumentException("Invalid coding matrix row for worker " + workerId);
            }
            System.arraycopy(codingRow, 0, codingSubmatrix[i], 0, numShards);
        }

        if (numResponded == Config.NUM_WORKERS) {
            double[] fullGradient = new double[gradientDim];
            for (int i = 0; i < numResponded; i++) {
                for (int j = 0; j < gradientDim; j++) {
                    fullGradient[j] += codedGradients[i][j];
                }
            }
            for (int j = 0; j < gradientDim; j++) {
                fullGradient[j] *= 0.5;
            }
            return fullGradient;
        }

        double[][] normalMatrix = new double[numResponded][numResponded];
        double[] rhs = new double[numResponded];
        for (int i = 0; i < numResponded; i++) {
            for (int j = 0; j < numResponded; j++) {
                normalMatrix[i][j] = dotProduct(codingSubmatrix[i], codingSubmatrix[j]);
            }
            rhs[i] = rowSum(codingSubmatrix[i]);
        }

        double[] recoveryWeights = solveLinearSystem(normalMatrix, rhs);
        verifyRecoveryWeights(codingSubmatrix, recoveryWeights);

        double[] fullGradient = new double[gradientDim];
        for (int i = 0; i < numResponded; i++) {
            for (int j = 0; j < gradientDim; j++) {
                fullGradient[j] += recoveryWeights[i] * codedGradients[i][j];
            }
        }
        return fullGradient;
    }

    private static void verifyRecoveryWeights(double[][] codingSubmatrix, double[] weights) {
        int numShards = codingSubmatrix[0].length;
        for (int shard = 0; shard < numShards; shard++) {
            double coefficientSum = 0.0;
            for (int i = 0; i < weights.length; i++) {
                coefficientSum += weights[i] * codingSubmatrix[i][shard];
            }
            if (Math.abs(coefficientSum - 1.0) > 1e-9) {
                throw new IllegalArgumentException("Recovery weights do not reproduce full gradient for shard "
                    + shard + ". Expected 1.0, got " + coefficientSum);
            }
        }
    }

    private static double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double rowSum(double[] row) {
        double sum = 0.0;
        for (double value : row) {
            sum += value;
        }
        return sum;
    }

    private static double[] solveLinearSystem(double[][] matrix, double[] rhs) {
        int n = matrix.length;
        if (rhs == null || rhs.length != n) {
            throw new IllegalArgumentException("Invalid linear system dimensions");
        }

        double[][] a = new double[n][n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
            b[i] = rhs[i];
        }

        for (int pivot = 0; pivot < n; pivot++) {
            int bestRow = pivot;
            double maxAbs = Math.abs(a[pivot][pivot]);
            for (int row = pivot + 1; row < n; row++) {
                double value = Math.abs(a[row][pivot]);
                if (value > maxAbs) {
                    maxAbs = value;
                    bestRow = row;
                }
            }
            if (maxAbs < 1e-12) {
                throw new IllegalArgumentException("Cannot solve linear system: matrix is singular or nearly singular");
            }
            if (bestRow != pivot) {
                double[] tempRow = a[pivot];
                a[pivot] = a[bestRow];
                a[bestRow] = tempRow;
                double tempVal = b[pivot];
                b[pivot] = b[bestRow];
                b[bestRow] = tempVal;
            }

            double diag = a[pivot][pivot];
            for (int col = pivot; col < n; col++) {
                a[pivot][col] /= diag;
            }
            b[pivot] /= diag;

            for (int row = 0; row < n; row++) {
                if (row == pivot) {
                    continue;
                }
                double factor = a[row][pivot];
                if (factor == 0.0) {
                    continue;
                }
                for (int col = pivot; col < n; col++) {
                    a[row][col] -= factor * a[pivot][col];
                }
                b[row] -= factor * b[pivot];
            }
        }
        return b;
    }
}
