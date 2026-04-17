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

    /**
     * Compute Mean Squared Error.
     * MSE = (1/N) · Σ(predicted_i - actual_i)²
     * 
     * @param predicted model predictions [N]
     * @param actual    ground truth values [N]
     * @return MSE value
     */
    public static double mse(double[] predicted, double[] actual) {
        // TODO: Rimsha
        throw new UnsupportedOperationException("Evaluation.mse not yet implemented");
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
        // TODO: Rimsha
        throw new UnsupportedOperationException("Evaluation.mae not yet implemented");
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
        // TODO: Rimsha
        throw new UnsupportedOperationException("Evaluation.r2 not yet implemented");
    }
}
