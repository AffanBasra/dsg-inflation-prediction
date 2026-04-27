package dsg;

import java.util.ArrayList;
import java.util.List;

/**
 * One-click launcher: spins up 1 Master JVM + 4 Worker JVMs using ProcessBuilder.
 * 
 * Usage: java dsg.IntegrationLauncher
 * 
 * This simulates a distributed system on a single machine. Each node runs
 * in its own JVM process with its own heap, communicating only via TCP sockets.
 * 
 * @author Affan Ahmed Basra (476173)
 */
public class IntegrationLauncher {

    /** Delay between starting the Master and the first Worker (ms). */
    private static final int MASTER_STARTUP_DELAY_MS = 2500;

    /** Delay between starting successive Workers (ms). */
    private static final int WORKER_STAGGER_DELAY_MS = 500;

    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════╗");
        System.out.println("║   DSG Inflation Forecasting — Distributed System Launcher ║");
        System.out.println("╠════════════════════════════════════════════════════════════╣");
        System.out.println("║   Workers: " + Config.NUM_WORKERS
            + "    Port: " + Config.MASTER_PORT
            + "    Epochs: " + Config.MAX_EPOCHS + "               ║");
        System.out.println("║   Straggler Timeout: " + Config.STRAGGLER_TIMEOUT_MS
            + "ms    Min Responses: " + Config.MIN_RESPONSES + "        ║");
        System.out.println("╚════════════════════════════════════════════════════════════╝");
        System.out.println();

        String classpath = "bin";
        List<Process> processes = new ArrayList<>();

        try {
            // ── 1. Start Master JVM ──────────────────────────────────
            System.out.println("[Launcher] Starting Master JVM...");
            ProcessBuilder masterPb = new ProcessBuilder(
                "java", "-cp", classpath, "dsg.Master"
            );
            masterPb.redirectErrorStream(true);
            masterPb.inheritIO();
            Process masterProc = masterPb.start();
            processes.add(masterProc);

            // Give the Master time to bind its ServerSocket
            Thread.sleep(MASTER_STARTUP_DELAY_MS);

            // ── 2. Start Worker JVMs ─────────────────────────────────
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                System.out.println("[Launcher] Starting Worker " + i + " JVM...");
                ProcessBuilder workerPb = new ProcessBuilder(
                    "java", "-cp", classpath, "dsg.Worker", String.valueOf(i)
                );
                workerPb.redirectErrorStream(true);
                workerPb.inheritIO();
                Process workerProc = workerPb.start();
                processes.add(workerProc);

                // Stagger worker starts so they connect in order
                Thread.sleep(WORKER_STAGGER_DELAY_MS);
            }

            System.out.println();
            System.out.println("[Launcher] All " + (1 + Config.NUM_WORKERS)
                + " JVM processes started. Waiting for Master to finish training...");
            System.out.println();

            // ── 3. Wait for Master to complete ───────────────────────
            int exitCode = masterProc.waitFor();
            System.out.println();
            System.out.println("[Launcher] Master exited with code " + exitCode);

        } catch (Exception e) {
            System.err.println("[Launcher] Fatal error: " + e.getMessage());
            e.printStackTrace();

        } finally {
            // ── 4. Clean up all processes ────────────────────────────
            for (Process p : processes) {
                if (p.isAlive()) {
                    p.destroyForcibly();
                }
            }
            System.out.println("[Launcher] All processes terminated.");
        }
    }
}
