package dsg;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Hybrid-parallel gradient computation using ExecutorService.
 * 
 * Responsibilities (Fizza Kashif — 466184):
 * 
 * Each worker processes ~9,854 rows per shard. This class splits that work
 * across THREADS_PER_WORKER (8) threads for intra-node parallelism:
 *   - Each thread handles ~1,232 rows (9,854 / 8)
 *   - Threads compute partial gradients independently
 *   - A synchronized reduction sums the partial gradients into the final result
 * 
 * This creates HYBRID parallelism:
 *   - Inter-node: 4 worker JVMs (distributed via TCP sockets)
 *   - Intra-node: 8 threads per worker (shared-memory via ExecutorService)
 */
/**
 * Computes gradients using intra-worker threading.
 */
public final class ThreadedGradient {

    private ThreadedGradient() {}

    public static double[] computeGradient(double[][] x, double[] y, double[] weights) {
        int rows = x.length;
        int threads = Math.min(Config.THREADS_PER_WORKER, Math.max(1, rows));

        if (rows == 0) {
            return new double[Config.NUM_FEATURES + 1];
        }

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        List<Future<double[]>> futures = new ArrayList<>();

        int chunkSize = (rows + threads - 1) / threads;

        for (int t = 0; t < threads; t++) {
            final int start = t * chunkSize;
            final int end = Math.min(start + chunkSize, rows);

            if (start >= end) {
                break;
            }

            Callable<double[]> task = () ->
                computePartialUnscaledGradient(x, y, weights, start, end);

            futures.add(pool.submit(task));
        }

        double[] total = new double[Config.NUM_FEATURES + 1];

        try {
            for (Future<double[]> future : futures) {
                double[] partial = future.get();
                synchronized (total) {
                    GradientComputer.addInPlace(total, partial);
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Threaded gradient interrupted", e);
        } catch (ExecutionException e) {
            throw new RuntimeException("Threaded gradient failed", e);
        } finally {
            pool.shutdown();
        }

        double scale = 2.0 / rows;
        GradientComputer.scaleInPlace(total, scale);
        return total;
    }

    /**
     * Computes unscaled partial gradient for rows [start, end).
     * Scaling is done once globally after reduction.
     */
    private static double[] computePartialUnscaledGradient(double[][] x, double[] y, double[] weights, int start, int end) {
        double[] gradient = new double[Config.NUM_FEATURES + 1];

        for (int i = start; i < end; i++) {
            double prediction = GradientComputer.predict(x[i], weights);
            double error = prediction - y[i];

            for (int j = 0; j < Config.NUM_FEATURES; j++) {
                gradient[j] += error * x[i][j];
            }

            gradient[Config.NUM_FEATURES] += error;
        }

        return gradient;
    }
}

