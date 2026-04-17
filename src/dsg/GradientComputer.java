package dsg;

/**
 * MSE Gradient Computation for linear regression.
 * 
 * Responsibilities (Fizza Kashif — 466184):
 * 
 * Model: y_pred = X · w[0..10] + w[11]   (11 features + 1 bias)
 * Loss:  MSE = (1/N) · Σ(y_pred - y_actual)²
 * Gradient: ∇MSE_wj = (2/N) · Σ (y_pred_i - y_actual_i) · X_ij    for j = 0..10
 *           ∇MSE_bias = (2/N) · Σ (y_pred_i - y_actual_i)          (bias gradient)
 * 
 * The weight vector has dimension NUM_FEATURES + 1 = 12:
 *   weights[0..10]  → feature weights
 *   weights[11]     → bias term
 */
public class GradientComputer {

    /**
     * Compute the MSE gradient on the given data shard.
     * 
     * @param X        feature matrix [N][NUM_FEATURES]
     * @param y        target vector [N]
     * @param weights  current weight vector [NUM_FEATURES + 1] (last element = bias)
     * @return gradient vector [NUM_FEATURES + 1]
     */
    public static double[] computeGradient(double[][] X, double[] y, double[] weights) {
        // TODO: Fizza — implement MSE derivative
        //
        // int N = X.length;
        // int D = Config.NUM_FEATURES;
        // double[] gradient = new double[D + 1];
        //
        // for (int i = 0; i < N; i++) {
        //     // Compute prediction: y_pred = Σ(X[i][j] * weights[j]) + weights[D]
        //     double pred = weights[D]; // start with bias
        //     for (int j = 0; j < D; j++) {
        //         pred += X[i][j] * weights[j];
        //     }
        //
        //     // Error
        //     double error = pred - y[i];
        //
        //     // Accumulate gradient
        //     for (int j = 0; j < D; j++) {
        //         gradient[j] += (2.0 / N) * error * X[i][j];
        //     }
        //     gradient[D] += (2.0 / N) * error; // bias gradient
        // }
        //
        // return gradient;

        throw new UnsupportedOperationException("GradientComputer not yet implemented — Fizza's task");
    }
}
