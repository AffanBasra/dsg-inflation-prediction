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
     * @throws IllegalArgumentException if fewer than MIN_RESPONSES are provided
     */
    public static double[] recoverFullGradient(int[] respondedWorkerIds, double[][] codedGradients) {
        // TODO: Rimsha — implement Gaussian elimination / pseudo-inverse recovery
        //
        // Steps:
        // 1. Build C_sub: for each responded worker, get their row from Config.CODING_MATRIX
        //    → C_sub is a [numResponded × NUM_SHARDS] matrix
        //
        // 2. We want to find w (length = numResponded) such that:
        //    w[0]*C_sub[0] + w[1]*C_sub[1] + ... = [1, 1, 1, 1]
        //    i.e., wᵀ · C_sub = onesVector
        //
        // 3. This is a system of NUM_SHARDS equations in numResponded unknowns.
        //    Use Gaussian elimination or least-squares.
        //
        // 4. Once w is found:
        //    fullGradient[j] = w[0]*codedGradients[0][j] + w[1]*codedGradients[1][j] + ...
        //
        // Dependency Buster Test:
        //    Create a unit test with hardcoded gradients:
        //    g0=[1,2], g1=[3,4], g2=[5,6], g3=[7,8]
        //    coded0 = g0+g1 = [4,6], coded1 = g1+g2 = [8,10], coded2 = g2+g3 = [12,14]
        //    recoverFullGradient({0,1,2}, {coded0,coded1,coded2}) should return [16,20]

        throw new UnsupportedOperationException("GaussianElimination not yet implemented — Rimsha's task");
    }
}
