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
 * 1. Connect to Master via SocketManager.WorkerClient(workerId)
 * 2. Receive InitMsg with shard data + coding coefficients
 * 3. Training loop:
 *    a. Receive BroadcastMsg (current weights)
 *    b. Compute gradient on shard 1 via GradientComputer/ThreadedGradient
 *    c. Compute gradient on shard 2 via GradientComputer/ThreadedGradient
 *    d. Coded gradient = shard1Coeff * grad1 + shard2Coeff * grad2
 *    e. Call StragglerInjector.maybeDelay(workerId) — adds delay for Worker 3
 *    f. Send GradReturnMsg back to Master
 * 
 * Available networking API (from Affan's SocketManager):
 *   - client = new SocketManager.WorkerClient(workerId)
 *   - client.receiveInit()        → InitMsg
 *   - client.receiveBroadcast()   → BroadcastMsg
 *   - client.sendGradient(msg)
 *   - client.close()
 * 
 * Available from InitMsg:
 *   - msg.getShard1X(), msg.getShard1Y()   → first shard data
 *   - msg.getShard2X(), msg.getShard2Y()   → second shard data
 *   - msg.getShard1Coeff(), msg.getShard2Coeff()  → coding coefficients
 *   - msg.getInitialWeights()              → starting weight vector
 */
/**
 * Worker node:
 * 1. receives its two shard payloads and coding coefficients
 * 2. waits for weight broadcasts from master
 * 3. computes coded gradient = c1*g1 + c2*g2
 * 4. sends gradient back to master
 */
public class Worker {

    private final String host;
    private final int port;

    private int workerId;
    private double[][] shard1X;
    private double[] shard1Y;
    private double[][] shard2X;
    private double[] shard2Y;
    private double shard1Coeff;
    private double shard2Coeff;
    private double[] currentWeights;

    public Worker(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void start() throws Exception {
        try (Socket socket = new Socket(host, port);
             ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
             ObjectInputStream in = new ObjectInputStream(socket.getInputStream())) {

            while (true) {
                Object message = in.readObject();

                if (message instanceof MessageProtocol.InitMsg initMsg) {
                    handleInit(initMsg);
                } else if (message instanceof MessageProtocol.BroadcastMsg broadcastMsg) {
                    MessageProtocol.GradReturnMsg response = handleBroadcast(broadcastMsg);
                    out.writeObject(response);
                    out.flush();
                } else {
                    System.out.println("Worker received unknown message: " + message);
                }
            }
        }
    }

    private void handleInit(MessageProtocol.InitMsg initMsg) {
        this.workerId = initMsg.getWorkerId();
        this.shard1X = initMsg.getShard1X();
        this.shard1Y = initMsg.getShard1Y();
        this.shard2X = initMsg.getShard2X();
        this.shard2Y = initMsg.getShard2Y();
        this.shard1Coeff = initMsg.getShard1Coeff();
        this.shard2Coeff = initMsg.getShard2Coeff();
        this.currentWeights = Arrays.copyOf(initMsg.getInitialWeights(), initMsg.getInitialWeights().length);

        System.out.println("Worker " + workerId + " initialized.");
        System.out.println("Shard 1 rows: " + shard1X.length + ", coeff: " + shard1Coeff);
        System.out.println("Shard 2 rows: " + shard2X.length + ", coeff: " + shard2Coeff);
    }

    private MessageProtocol.GradReturnMsg handleBroadcast(MessageProtocol.BroadcastMsg broadcastMsg) {
        this.currentWeights = Arrays.copyOf(broadcastMsg.getWeights(), broadcastMsg.getWeights().length);

        double[] grad1 = ThreadedGradient.computeGradient(shard1X, shard1Y, currentWeights);
        double[] grad2 = ThreadedGradient.computeGradient(shard2X, shard2Y, currentWeights);

        double[] codedGradient = new double[Config.NUM_FEATURES + 1];
        for (int i = 0; i < codedGradient.length; i++) {
            codedGradient[i] = (shard1Coeff * grad1[i]) + (shard2Coeff * grad2[i]);
        }

        System.out.println("Worker " + workerId + " computed coded gradient for epoch " + broadcastMsg.getEpoch());

        return new MessageProtocol.GradReturnMsg(
            workerId,
            codedGradient,
            broadcastMsg.getEpoch()
        );
    }

    public static void main(String[] args) throws Exception {
        String host = args.length > 0 ? args[0] : Config.MASTER_HOST;
        int port = args.length > 1 ? Integer.parseInt(args[1]) : Config.MASTER_PORT;

        Worker worker = new Worker(host, port);
        worker.start();
    }
}