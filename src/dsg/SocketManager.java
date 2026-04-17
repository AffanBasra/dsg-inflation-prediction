package dsg;

import java.io.Closeable;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.util.ArrayList;
import java.util.List;

/**
 * TCP socket management for Master↔Worker communication.
 * 
 * Two static inner classes:
 *   - MasterServer: accepts worker connections, broadcasts messages, collects gradients
 *   - WorkerClient: connects to master, sends/receives messages
 * 
 * All I/O uses Java's ObjectOutputStream/ObjectInputStream for serialization.
 * Straggler detection uses socket.setSoTimeout(5000ms) → SocketTimeoutException.
 * 
 * @author Affan Ahmed Basra (476173)
 */
public final class SocketManager {

    private SocketManager() {} // prevent instantiation

    // ═════════════════════════════════════════════════════════════════
    // MasterServer — Server-side socket management
    // ═════════════════════════════════════════════════════════════════

    /**
     * Manages the master's server socket and connected worker streams.
     * 
     * Lifecycle:
     *   1. new MasterServer()          — binds ServerSocket
     *   2. acceptWorkers()             — waits for NUM_WORKERS connections
     *   3. sendInit(workerId, msg)     — sends InitMsg to specific worker
     *   4. broadcast(msg)             — sends BroadcastMsg to all workers
     *   5. collectGradients(timeout)   — reads GradReturnMsgs with timeout
     *   6. close()                     — cleans up all sockets
     */
    public static class MasterServer implements Closeable {

        private final ServerSocket serverSocket;
        private final Socket[] workerSockets;
        private final ObjectOutputStream[] outputs;
        private final ObjectInputStream[] inputs;

        /**
         * Creates the master server and binds to Config.MASTER_PORT.
         * @throws IOException if the port is already in use
         */
        public MasterServer() throws IOException {
            this.serverSocket = new ServerSocket(Config.MASTER_PORT);
            this.workerSockets = new Socket[Config.NUM_WORKERS];
            this.outputs = new ObjectOutputStream[Config.NUM_WORKERS];
            this.inputs = new ObjectInputStream[Config.NUM_WORKERS];
            System.out.println("[Master] Server started on port " + Config.MASTER_PORT);
        }

        /**
         * Accepts connections from all workers.
         * Workers send their ID as the first integer after connecting.
         * This allows workers to connect in any order.
         * 
         * @throws IOException if a connection fails
         */
        public void acceptWorkers() throws IOException {
            System.out.println("[Master] Waiting for " + Config.NUM_WORKERS + " workers...");

            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                Socket socket = serverSocket.accept();

                // Create streams (ObjectOutputStream first to avoid deadlock)
                ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
                out.flush();
                ObjectInputStream in = new ObjectInputStream(socket.getInputStream());

                // Read the worker's self-reported ID
                int workerId = in.readInt();

                if (workerId < 0 || workerId >= Config.NUM_WORKERS) {
                    throw new IOException("Invalid worker ID: " + workerId);
                }
                if (workerSockets[workerId] != null) {
                    throw new IOException("Duplicate connection from worker " + workerId);
                }

                workerSockets[workerId] = socket;
                outputs[workerId] = out;
                inputs[workerId] = in;

                System.out.println("[Master] Worker " + workerId + " connected from "
                    + socket.getRemoteSocketAddress()
                    + " (" + (i + 1) + "/" + Config.NUM_WORKERS + ")");
            }

            System.out.println("[Master] All " + Config.NUM_WORKERS + " workers connected.");
        }

        /**
         * Send an InitMsg to a specific worker.
         * 
         * @param workerId  target worker index (0..NUM_WORKERS-1)
         * @param msg       the initialization message
         * @throws IOException if the send fails
         */
        public void sendInit(int workerId, MessageProtocol.InitMsg msg) throws IOException {
            outputs[workerId].writeObject(msg);
            outputs[workerId].flush();
            outputs[workerId].reset(); // prevent cached object memory leak
        }

        /**
         * Broadcast a BroadcastMsg to ALL connected workers.
         * 
         * @param msg the message to broadcast
         * @throws IOException if any send fails
         */
        public void broadcast(MessageProtocol.BroadcastMsg msg) throws IOException {
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                outputs[i].writeObject(msg);
                outputs[i].flush();
                outputs[i].reset();
            }
        }

        /**
         * Collect coded gradient responses from workers with a timeout.
         * 
         * Reads from each worker socket sequentially. If a worker does not
         * respond within timeoutMs, it is logged as a straggler and skipped.
         * 
         * @param timeoutMs  read timeout per worker in milliseconds
         * @return list of received GradReturnMsgs (may have fewer than NUM_WORKERS)
         */
        public List<MessageProtocol.GradReturnMsg> collectGradients(int timeoutMs) {
            List<MessageProtocol.GradReturnMsg> results = new ArrayList<>();

            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                try {
                    workerSockets[i].setSoTimeout(timeoutMs);
                    Object obj = inputs[i].readObject();

                    if (obj instanceof MessageProtocol.GradReturnMsg) {
                        MessageProtocol.GradReturnMsg msg = (MessageProtocol.GradReturnMsg) obj;
                        results.add(msg);
                    } else {
                        System.err.println("[Master] Unexpected message type from Worker "
                            + i + ": " + obj.getClass().getName());
                    }

                } catch (SocketTimeoutException e) {
                    System.out.println("[Master] ⚠ Worker " + i
                        + " timed out after " + timeoutMs + "ms (straggler detected)");

                } catch (IOException | ClassNotFoundException e) {
                    System.err.println("[Master] Error reading from Worker " + i
                        + ": " + e.getMessage());
                }
            }

            return results;
        }

        /**
         * Close all worker sockets and the server socket.
         */
        @Override
        public void close() throws IOException {
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                if (workerSockets[i] != null && !workerSockets[i].isClosed()) {
                    try {
                        workerSockets[i].close();
                    } catch (IOException e) {
                        System.err.println("[Master] Error closing Worker " + i + " socket: " + e.getMessage());
                    }
                }
            }
            if (!serverSocket.isClosed()) {
                serverSocket.close();
            }
            System.out.println("[Master] Server shut down.");
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // WorkerClient — Client-side socket management
    // ═════════════════════════════════════════════════════════════════

    /**
     * Manages a single worker's connection to the master.
     * 
     * Lifecycle:
     *   1. new WorkerClient(workerId)  — connects to master, sends ID
     *   2. receiveInit()               — reads the InitMsg
     *   3. receiveBroadcast()          — reads a BroadcastMsg (each epoch)
     *   4. sendGradient(msg)           — sends a GradReturnMsg (each epoch)
     *   5. close()                     — closes the socket
     */
    public static class WorkerClient implements Closeable {

        private final int workerId;
        private final Socket socket;
        private final ObjectOutputStream output;
        private final ObjectInputStream input;

        /**
         * Connects to the master and sends this worker's ID.
         * 
         * @param workerId this worker's ID (0..NUM_WORKERS-1)
         * @throws IOException if the connection fails
         */
        public WorkerClient(int workerId) throws IOException {
            this.workerId = workerId;
            this.socket = new Socket(Config.MASTER_HOST, Config.MASTER_PORT);

            // Create streams (ObjectOutputStream first to avoid deadlock)
            this.output = new ObjectOutputStream(socket.getOutputStream());
            this.output.flush();
            this.input = new ObjectInputStream(socket.getInputStream());

            // Send worker ID as handshake
            this.output.writeInt(workerId);
            this.output.flush();

            System.out.println("[Worker " + workerId + "] Connected to Master at "
                + Config.MASTER_HOST + ":" + Config.MASTER_PORT);
        }

        /**
         * Receive the one-time InitMsg from the master.
         * 
         * @return the initialization message
         * @throws IOException if read fails
         * @throws ClassNotFoundException if deserialization fails
         */
        public MessageProtocol.InitMsg receiveInit() throws IOException, ClassNotFoundException {
            Object obj = input.readObject();
            if (obj instanceof MessageProtocol.InitMsg) {
                MessageProtocol.InitMsg msg = (MessageProtocol.InitMsg) obj;
                System.out.println("[Worker " + workerId + "] Received: " + msg);
                return msg;
            }
            throw new IOException("Expected InitMsg, received: " + obj.getClass().getName());
        }

        /**
         * Receive a per-epoch BroadcastMsg from the master.
         * 
         * @return the broadcast message with current weights
         * @throws IOException if read fails
         * @throws ClassNotFoundException if deserialization fails
         */
        public MessageProtocol.BroadcastMsg receiveBroadcast() throws IOException, ClassNotFoundException {
            Object obj = input.readObject();
            if (obj instanceof MessageProtocol.BroadcastMsg) {
                return (MessageProtocol.BroadcastMsg) obj;
            }
            throw new IOException("Expected BroadcastMsg, received: " + obj.getClass().getName());
        }

        /**
         * Send a coded gradient response to the master.
         * 
         * @param msg the gradient return message
         * @throws IOException if write fails
         */
        public void sendGradient(MessageProtocol.GradReturnMsg msg) throws IOException {
            output.writeObject(msg);
            output.flush();
            output.reset();
        }

        /**
         * Close the connection to the master.
         */
        @Override
        public void close() throws IOException {
            if (socket != null && !socket.isClosed()) {
                socket.close();
                System.out.println("[Worker " + workerId + "] Disconnected.");
            }
        }
    }
}
