package dsg;

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
public class Worker {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java dsg.Worker <workerId>");
            System.exit(1);
        }

        int workerId = Integer.parseInt(args[0]);
        System.out.println("[Worker " + workerId + "] Starting...");

        // TODO: Fizza — implement worker logic using the API above.
        //
        // Example skeleton:
        //
        // try (SocketManager.WorkerClient client = new SocketManager.WorkerClient(workerId)) {
        //     // Receive initialization
        //     MessageProtocol.InitMsg init = client.receiveInit();
        //     double[][] s1X = init.getShard1X();
        //     double[]   s1Y = init.getShard1Y();
        //     double[][] s2X = init.getShard2X();
        //     double[]   s2Y = init.getShard2Y();
        //     double c1 = init.getShard1Coeff();
        //     double c2 = init.getShard2Coeff();
        //
        //     // Training loop
        //     for (int epoch = 0; epoch < Config.MAX_EPOCHS; epoch++) {
        //         MessageProtocol.BroadcastMsg broadcast = client.receiveBroadcast();
        //         double[] weights = broadcast.getWeights();
        //
        //         // Compute gradients (use ThreadedGradient for parallelism)
        //         double[] grad1 = GradientComputer.computeGradient(s1X, s1Y, weights);
        //         double[] grad2 = GradientComputer.computeGradient(s2X, s2Y, weights);
        //
        //         // Compute coded gradient
        //         double[] codedGrad = new double[weights.length];
        //         for (int j = 0; j < weights.length; j++) {
        //             codedGrad[j] = c1 * grad1[j] + c2 * grad2[j];
        //         }
        //
        //         // Straggler injection (Worker 3 gets delayed)
        //         StragglerInjector.maybeDelay(workerId);
        //
        //         // Send coded gradient back
        //         client.sendGradient(
        //             new MessageProtocol.GradReturnMsg(workerId, codedGrad, epoch));
        //     }
        // }

        System.out.println("[Worker " + workerId + "] Not yet implemented — awaiting Fizza's code.");
    }
}
