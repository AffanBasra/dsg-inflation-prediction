package dsg;

import java.util.Arrays;

/**
 * Offline test for gradient correctness.
 * Compares single-thread and threaded outputs on mock data.
 */
public class WorkerTests {

    public static void main(String[] args) {
        double[][] x = {
            {1.0, 2.0, 0.5, 1.0, 0.0, 3.0, 1.0, 2.0, 0.1, 0.2, 0.3},
            {2.0, 1.0, 1.5, 0.0, 1.0, 2.0, 2.0, 1.0, 0.3, 0.1, 0.2},
            {0.5, 1.5, 2.0, 1.0, 2.0, 0.5, 0.5, 1.5, 0.2, 0.4, 0.1},
            {1.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.5, 0.5, 0.5, 0.3, 0.2}
        };

        double[] y = {5.0, 4.5, 6.0, 5.5};

        double[] weights = new double[Config.NUM_FEATURES + 1];
        Arrays.fill(weights, 0.1);

        double[] single = GradientComputer.computeGradient(x, y, weights);
        double[] threaded = ThreadedGradient.computeGradient(x, y, weights);

        System.out.println("Single-thread gradient : " + Arrays.toString(single));
        System.out.println("Threaded gradient      : " + Arrays.toString(threaded));

        for (int i = 0; i < single.length; i++) {
            if (Math.abs(single[i] - threaded[i]) > 1e-9) {
                throw new RuntimeException("Mismatch at index " + i);
            }
        }

        System.out.println("WorkerTests passed.");
    }
}
