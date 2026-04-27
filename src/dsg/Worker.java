package dsg;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.Arrays;

/**
 * Worker Node — Computes local coded gradients on assigned shards.
 * 
 * Usage: java dsg.Worker <workerId>
 * 
 * Responsibilities (Fizza Kashif — 466184):
 * 
 * 1. Connect to Master via socket connection (workerId)
 * 2. Receive InitMsg with shard data + coding coefficients
 * 3. Training loop:
 *    a. Receive BroadcastMsg (current weights)
 *    b. Compute gradient on shard 1 via ThreadedGradient
 *    c. Compute gradient on shard 2 via ThreadedGradient
 *    d. Coded gradient = shard1Coeff * grad1 + shard2Coeff * grad2
 *    e. Call StragglerInjector.maybeDelay(workerId) — adds delay for Worker 3
 *    f. Send GradReturnMsg back to Master
 * 
 * Available from InitMsg:
 *   - msg.getShard1X(), msg.getShard1Y()   → first shard data
 *   - msg.getShard2X(), msg.getShard2Y()   → second shard data
 *   - msg.getShard1Coeff(), msg.getShard2Coeff()  → coding coefficients
 *   - msg.getInitialWeights()              → starting weight vector
 */
public class Worker {

    private final int workerId;
    private double[][] shard1X;
    private double[] shard1Y;
    private double[][] shard2X;
    private double[] shard2Y;
    private double shard1Coeff;
    private double shard2Coeff;
    private double[] currentWeights;

    /**
     * Constructor: initialize worker with its ID.
     * 
     * @param workerId this worker's unique identifier (0..NUM_WORKERS-1)
     */
    public Worker(int workerId) {
        this.workerId = workerId;
    }

    /**
     * Main execution loop:
     * 1. Connect to Master
     * 2. Receive InitMsg (one-time initialization)
     * 3. Loop: Receive BroadcastMsg → Compute gradient → Send GradReturnMsg
     * 4. Handle graceful shutdown when Master closes connection
     * 
     * @throws Exception if connection fails or I/O error occurs
     */
    public void start() throws Exception {
        System.out.println("[Worker " + workerId + "] Starting...");
        
        Socket socket = null;
        ObjectOutputStream out = null;
        ObjectInputStream in = null;
        int maxRetries = 10;
        int retryDelay = 500; // milliseconds
        
        // ── Connect to Master with retries (Windows localhost workaround) ──
        for (int attempt = 0; attempt < maxRetries; attempt++) {
            try {
                socket = new Socket(Config.MASTER_HOST, Config.MASTER_PORT);
                out = new ObjectOutputStream(socket.getOutputStream());
                out.flush();
                in = new ObjectInputStream(socket.getInputStream());
                
                // Send worker ID handshake
                out.writeInt(workerId);
                out.flush();
                
                System.out.println("[Worker " + workerId + "] Connected to Master at "
                    + Config.MASTER_HOST + ":" + Config.MASTER_PORT);
                break;
            } catch (Exception e) {
                if (attempt < maxRetries - 1) {
                    System.out.println("[Worker " + workerId + "] Connection attempt " 
                        + (attempt + 1) + " failed, retrying in " + retryDelay + "ms...");
                    Thread.sleep(retryDelay);
                } else {
                    throw new Exception("Failed to connect to Master after " + maxRetries 
                        + " attempts: " + e.getMessage(), e);
                }
            }
        }
        
        // ── Main message loop ──
        try {
            while (true) {
                try {
                    Object message = in.readObject();

                    if (message instanceof MessageProtocol.InitMsg) {
                        // One-time initialization with shard data
                        handleInit((MessageProtocol.InitMsg) message);
                        
                    } else if (message instanceof MessageProtocol.BroadcastMsg) {
                        // Per-epoch: receive weights, compute gradient, send back
                        MessageProtocol.BroadcastMsg broadcastMsg = (MessageProtocol.BroadcastMsg) message;
                        MessageProtocol.GradReturnMsg response = handleBroadcast(broadcastMsg);
                        
                        // Inject straggler delay for Worker 3
                        StragglerInjector.maybeDelay(workerId);
                        
                        out.writeObject(response);
                        out.flush();
                        
                    } else {
                        System.out.println("[Worker " + workerId + "] Received unknown message type: " 
                            + message.getClass().getName());
                    }
                } catch (java.io.EOFException e) {
                    // Normal graceful shutdown when Master closes connection
                    System.out.println("[Worker " + workerId + "] Connection closed by Master (normal shutdown)");
                    break;
                }
            }
        } finally {
            // ── Clean up resources ──
            if (socket != null && !socket.isClosed()) {
                socket.close();
                System.out.println("[Worker " + workerId + "] Disconnected.");
            }
        }
    }

    /**
     * Handle InitMsg: store shard data and initialization weights.
     * Called once at startup.
     * 
     * @param initMsg the initialization message from Master
     */
    private void handleInit(MessageProtocol.InitMsg initMsg) {
        this.shard1X = initMsg.getShard1X();
        this.shard1Y = initMsg.getShard1Y();
        this.shard2X = initMsg.getShard2X();
        this.shard2Y = initMsg.getShard2Y();
        this.shard1Coeff = initMsg.getShard1Coeff();
        this.shard2Coeff = initMsg.getShard2Coeff();
        this.currentWeights = Arrays.copyOf(initMsg.getInitialWeights(), 
            initMsg.getInitialWeights().length);

        System.out.println("[Worker " + workerId + "] Initialized.");
        System.out.println("[Worker " + workerId + "] Shard 1: " + shard1X.length 
            + " rows, coeff=" + shard1Coeff);
        System.out.println("[Worker " + workerId + "] Shard 2: " + shard2X.length 
            + " rows, coeff=" + shard2Coeff);
    }

    /**
     * Handle BroadcastMsg: compute coded gradient and return it.
     * Called once per epoch.
     * 
     * Steps:
     * 1. Update currentWeights from broadcast
     * 2. Compute gradient on shard1 using ThreadedGradient
     * 3. Compute gradient on shard2 using ThreadedGradient
     * 4. Combine: codedGradient = shard1Coeff*grad1 + shard2Coeff*grad2
     * 5. Return GradReturnMsg to Master
     * 
     * @param broadcastMsg the broadcast message containing current weights
     * @return coded gradient response to send back to Master
     */
    private MessageProtocol.GradReturnMsg handleBroadcast(MessageProtocol.BroadcastMsg broadcastMsg) {
        // Update weights
        this.currentWeights = Arrays.copyOf(broadcastMsg.getWeights(), 
            broadcastMsg.getWeights().length);

        // Compute gradients using hybrid (multithreaded) computation
        double[] grad1 = ThreadedGradient.computeGradient(shard1X, shard1Y, currentWeights);
        double[] grad2 = ThreadedGradient.computeGradient(shard2X, shard2Y, currentWeights);

        // Combine gradients using coding coefficients
        double[] codedGradient = new double[Config.NUM_FEATURES + 1];
        for (int i = 0; i < codedGradient.length; i++) {
            codedGradient[i] = (shard1Coeff * grad1[i]) + (shard2Coeff * grad2[i]);
        }

        System.out.println("[Worker " + workerId + "] Computed coded gradient for epoch " 
            + broadcastMsg.getEpoch());

        return new MessageProtocol.GradReturnMsg(
            workerId,
            codedGradient,
            broadcastMsg.getEpoch()
        );
    }

    /**
     * Main entry point: parse workerId from command-line args and start worker.
     * 
     * Usage: java dsg.Worker <workerId>
     *   where workerId is an integer from 0 to (NUM_WORKERS - 1)
     * 
     * Example: java dsg.Worker 0    (start Worker 0)
     * 
     * @param args command-line arguments (must contain workerId)
     * @throws Exception if workerId is invalid or connection fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            throw new IllegalArgumentException("Usage: java dsg.Worker <workerId>");
        }
        
        try {
            int workerId = Integer.parseInt(args[0]);
            
            if (workerId < 0 || workerId >= Config.NUM_WORKERS) {
                throw new IllegalArgumentException(
                    "Worker ID must be between 0 and " + (Config.NUM_WORKERS - 1) 
                    + ", but got: " + workerId);
            }
            
            Worker worker = new Worker(workerId);
            worker.start();
            
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                "Invalid worker ID format: '" + args[0] + "' is not an integer", e);
        }
    }
}