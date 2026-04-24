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
/**
 * Computes gradients for linear regression with MSE loss.
 * Weight vector format: [w0, w1, ..., w10, bias]
 */
public final class GradientComputer {

    private GradientComputer() {}

    /**
     * Computes the full average MSE gradient on a shard.
     * Returns gradient of size NUM_FEATURES + 1 (last term is bias gradient).
     */
    public static double[] computeGradient(double[][] x, double[] y, double[] weights) {
        return computeGradientRange(x, y, weights, 0, x.length);
    }

    /**
     * Computes the average MSE gradient over rows [start, end).
     * Returns gradient of size NUM_FEATURES + 1.
     */
    public static double[] computeGradientRange(double[][] x, double[] y, double[] weights, int start, int end) {
        int featureCount = Config.NUM_FEATURES;
        double[] gradient = new double[featureCount + 1];

        int count = end - start;
        if (count <= 0) {
            return gradient;
        }

        for (int i = start; i < end; i++) {
            double prediction = predict(x[i], weights);
            double error = prediction - y[i];

            for (int j = 0; j < featureCount; j++) {
                gradient[j] += error * x[i][j];
            }

            gradient[featureCount] += error; // bias term
        }

        double scale = 2.0 / count;
        for (int j = 0; j < gradient.length; j++) {
            gradient[j] *= scale;
        }

        return gradient;
    }

    /**
     * Predicts y_hat = w.x + b
     */
    public static double predict(double[] features, double[] weights) {
        double sum = weights[Config.NUM_FEATURES]; // bias at end
        for (int j = 0; j < Config.NUM_FEATURES; j++) {
            sum += features[j] * weights[j];
        }
        return sum;
    }

    /**
     * Adds src into dest in-place.
     */
    public static void addInPlace(double[] dest, double[] src) {
        for (int i = 0; i < dest.length; i++) {
            dest[i] += src[i];
        }
    }

    /**
     * Multiplies vector by scalar in-place.
     */
    public static void scaleInPlace(double[] vector, double scalar) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }
    }
}

