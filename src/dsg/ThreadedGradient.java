package dsg;

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
public class ThreadedGradient {

    /**
     * Compute gradient using multi-threaded parallelism.
     * 
     * @param X        feature matrix [N][NUM_FEATURES]
     * @param y        target vector [N]
     * @param weights  current weight vector [NUM_FEATURES + 1]
     * @return gradient vector [NUM_FEATURES + 1]
     */
    public static double[] computeParallel(double[][] X, double[] y, double[] weights) {
        // TODO: Fizza — implement threaded gradient
        //
        // 1. Create ExecutorService with Config.THREADS_PER_WORKER threads
        //    ExecutorService pool = Executors.newFixedThreadPool(Config.THREADS_PER_WORKER);
        //
        // 2. Partition rows into THREADS_PER_WORKER chunks
        //    int chunkSize = X.length / Config.THREADS_PER_WORKER;
        //
        // 3. Submit GradientComputer.computeGradient() on each chunk
        //    List<Future<double[]>> futures = new ArrayList<>();
        //
        // 4. Collect results and sum partial gradients
        //    double[] gradient = new double[Config.NUM_FEATURES + 1];
        //    for each future: gradient[j] += partialGrad[j];
        //
        // 5. Shutdown pool
        //    pool.shutdown();
        //
        // CRITICAL: The reduction step must be synchronized to avoid race conditions.

        throw new UnsupportedOperationException("ThreadedGradient not yet implemented — Fizza's task");
    }
}
