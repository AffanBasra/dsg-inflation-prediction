package dsg;

import java.util.Random;

/**
 * Injects artificial delay in Worker 3 to simulate a straggler node.
 * 
 * In a real distributed system, stragglers occur due to network latency,
 * hardware issues, or uneven workload. This class simulates that behavior
 * so the GC-DC recovery path in the Master can be tested.
 * 
 * Worker 3 is the designated straggler. On each epoch, it sleeps for a
 * random duration between 2–5 seconds, which exceeds the Master's
 * STRAGGLER_TIMEOUT_MS (5s), triggering the fallback gradient recovery.
 * 
 * Usage (inside Worker.java, called by Fizza):
 *   StragglerInjector.maybeDelay(workerId);
 * 
 * @author Affan Ahmed Basra (476173)
 */
public class StragglerInjector {

    /** The worker ID that will be artificially delayed. */
    private static final int STRAGGLER_WORKER_ID = 3;

    /** Minimum straggler delay in milliseconds. */
    private static final int MIN_DELAY_MS = 2000;

    /** Maximum straggler delay in milliseconds. */
    private static final int MAX_DELAY_MS = 5000;

    private StragglerInjector() {} // prevent instantiation

    /**
     * If the given worker is the designated straggler (Worker 3),
     * inject a random delay between MIN_DELAY_MS and MAX_DELAY_MS.
     * 
     * Non-straggler workers pass through immediately with no delay.
     * 
     * @param workerId the calling worker's ID
     */
    public static void maybeDelay(int workerId) {
        if (workerId == STRAGGLER_WORKER_ID) {
            Random rng = new Random();
            int delayMs = MIN_DELAY_MS + rng.nextInt(MAX_DELAY_MS - MIN_DELAY_MS + 1);
            System.out.printf("[Worker %d] ⚠ Straggler delay injected: %d ms%n", workerId, delayMs);
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.printf("[Worker %d] Straggler delay interrupted%n", workerId);
            }
        }
    }
}
