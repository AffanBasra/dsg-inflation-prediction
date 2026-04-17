package dsg;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Defines the exact serializable message types for Master↔Worker communication.
 * 
 * Protocol flow per epoch:
 *   1. Master → Worker:  InitMsg      (once, at startup)
 *   2. Master → Worker:  BroadcastMsg (each epoch — current weights)
 *   3. Worker → Master:  GradReturnMsg (each epoch — coded gradient)
 * 
 * All classes use Java's built-in serialization over TCP (ObjectOutputStream).
 * 
 * @author Affan Ahmed Basra (476173)
 */
public final class MessageProtocol {

    private MessageProtocol() {} // prevent instantiation

    // ═════════════════════════════════════════════════════════════════
    // InitMsg: Master → Worker (one-time initialization payload)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Sent once at startup to give each worker its shard data,
     * coding coefficients, and the initial weight vector.
     * 
     * Each worker receives data from TWO shards (for GC-DC redundancy).
     * The worker computes: coded_grad = coeff1 * grad(shard1) + coeff2 * grad(shard2)
     */
    public static class InitMsg implements Serializable {
        private static final long serialVersionUID = 1L;

        private final int workerId;
        private final double[][] shard1X;       // first shard features  [rows][NUM_FEATURES]
        private final double[]   shard1Y;       // first shard targets   [rows]
        private final double[][] shard2X;       // second shard features [rows][NUM_FEATURES]
        private final double[]   shard2Y;       // second shard targets  [rows]
        private final double     shard1Coeff;   // coding coefficient for shard 1
        private final double     shard2Coeff;   // coding coefficient for shard 2
        private final double[]   initialWeights; // starting weights [NUM_FEATURES + 1] (bias at end)

        public InitMsg(int workerId,
                       double[][] shard1X, double[] shard1Y,
                       double[][] shard2X, double[] shard2Y,
                       double shard1Coeff, double shard2Coeff,
                       double[] initialWeights) {
            this.workerId = workerId;
            this.shard1X = shard1X;
            this.shard1Y = shard1Y;
            this.shard2X = shard2X;
            this.shard2Y = shard2Y;
            this.shard1Coeff = shard1Coeff;
            this.shard2Coeff = shard2Coeff;
            this.initialWeights = initialWeights;
        }

        public int        getWorkerId()       { return workerId; }
        public double[][] getShard1X()        { return shard1X; }
        public double[]   getShard1Y()        { return shard1Y; }
        public double[][] getShard2X()        { return shard2X; }
        public double[]   getShard2Y()        { return shard2Y; }
        public double     getShard1Coeff()    { return shard1Coeff; }
        public double     getShard2Coeff()    { return shard2Coeff; }
        public double[]   getInitialWeights() { return initialWeights; }

        @Override
        public String toString() {
            return String.format(
                "InitMsg[worker=%d, shard1=%d rows, shard2=%d rows, coeffs=(%.1f, %.1f), weightDim=%d]",
                workerId,
                shard1X != null ? shard1X.length : 0,
                shard2X != null ? shard2X.length : 0,
                shard1Coeff, shard2Coeff,
                initialWeights != null ? initialWeights.length : 0
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // BroadcastMsg: Master → Worker (per-epoch weight broadcast)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Sent at the start of each training epoch.
     * Contains the current model weights after the previous update.
     */
    public static class BroadcastMsg implements Serializable {
        private static final long serialVersionUID = 2L;

        private final double[] weights;  // current weight vector [NUM_FEATURES + 1]
        private final int epoch;

        public BroadcastMsg(double[] weights, int epoch) {
            this.weights = weights;
            this.epoch = epoch;
        }

        public double[] getWeights() { return weights; }
        public int      getEpoch()   { return epoch; }

        @Override
        public String toString() {
            return String.format("BroadcastMsg[epoch=%d, weightDim=%d]",
                epoch, weights != null ? weights.length : 0);
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // GradReturnMsg: Worker → Master (coded gradient response)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Sent by the worker after computing its coded gradient for the current epoch.
     * The coded gradient = coeff1 * grad(shard1) + coeff2 * grad(shard2).
     */
    public static class GradReturnMsg implements Serializable {
        private static final long serialVersionUID = 3L;

        private final int      workerId;
        private final double[] codedGradient;  // coded gradient [NUM_FEATURES + 1]
        private final int      epoch;

        public GradReturnMsg(int workerId, double[] codedGradient, int epoch) {
            this.workerId = workerId;
            this.codedGradient = codedGradient;
            this.epoch = epoch;
        }

        public int      getWorkerId()      { return workerId; }
        public double[] getCodedGradient() { return codedGradient; }
        public int      getEpoch()         { return epoch; }

        @Override
        public String toString() {
            return String.format("GradReturnMsg[worker=%d, epoch=%d, gradDim=%d]",
                workerId, epoch, codedGradient != null ? codedGradient.length : 0);
        }
    }
}
