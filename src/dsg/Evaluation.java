package dsg;

/**
 * Model evaluation utilities — metrics and de-normalization.
 *
 * Responsibilities (Rimsha Mahmood — 455080):
 *
 * Computes standard regression metrics (MSE, MAE, R²) and handles
 * de-normalization of predictions back to real inflation values
 * using scaler_params.json.
 */
public class Evaluation {

    private static void validateInputs(double[] predicted, double[] actual) {
        if (predicted == null) {
            throw new IllegalArgumentException("predicted array must not be null");
        }
        if (actual == null) {
            throw new IllegalArgumentException("actual array must not be null");
        }
        if (predicted.length == 0) {
            throw new IllegalArgumentException("predicted array must not be empty");
        }
        if (actual.length == 0) {
            throw new IllegalArgumentException("actual array must not be empty");
        }
        if (predicted.length != actual.length) {
            throw new IllegalArgumentException(String.format(
                "predicted and actual arrays must have the same length: predicted=%d, actual=%d",
                predicted.length, actual.length));
        }
    }

    /**
     * Compute Mean Squared Error.
     * MSE = (1/N) · Σ(predicted_i - actual_i)²
     *
     * @param predicted model predictions [N]
     * @param actual    ground truth values [N]
     * @return MSE value
     */
    public static double mse(double[] predicted, double[] actual) {
        validateInputs(predicted, actual);
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double error = predicted[i] - actual[i];
            sum += error * error;
        }
        return sum / predicted.length;
    }

    /**
     * Compute Mean Absolute Error.
     * MAE = (1/N) · Σ|predicted_i - actual_i|
     *
     * @param predicted model predictions [N]
     * @param actual    ground truth values [N]
     * @return MAE value
     */
    public static double mae(double[] predicted, double[] actual) {
        validateInputs(predicted, actual);
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            sum += Math.abs(predicted[i] - actual[i]);
        }
        return sum / predicted.length;
    }

    /**
     * Compute R² (coefficient of determination).
     * R² = 1 - SS_res / SS_tot
     * where SS_res = Σ(actual_i - predicted_i)² and SS_tot = Σ(actual_i - mean)²
     *
     * @param predicted model predictions [N]
     * @param actual    ground truth values [N]
     * @return R² value (1.0 = perfect fit, 0.0 = baseline, negative = worse than mean)
     */
    public static double r2(double[] predicted, double[] actual) {
        validateInputs(predicted, actual);
        double mean = 0.0;
        for (double value : actual) {
            mean += value;
        }
        mean /= actual.length;

        double ssRes = 0.0;
        double ssTot = 0.0;
        for (int i = 0; i < actual.length; i++) {
            double error = actual[i] - predicted[i];
            ssRes += error * error;
            double deviation = actual[i] - mean;
            ssTot += deviation * deviation;
        }

        if (ssTot == 0.0) {
            return ssRes == 0.0 ? 1.0 : 0.0;
        }
        return 1.0 - (ssRes / ssTot);
    }
}
