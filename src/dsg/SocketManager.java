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
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public final class SocketManager {

    private SocketManager() {} // prevent instantiation

    public static class MasterServer implements Closeable {

        private final ServerSocket serverSocket;
        private final Socket[] workerSockets;
        private final ObjectOutputStream[] outputs;
        private final ObjectInputStream[] inputs;

        public MasterServer() throws IOException {
            this.serverSocket = new ServerSocket(Config.MASTER_PORT);
            this.workerSockets = new Socket[Config.NUM_WORKERS];
            this.outputs = new ObjectOutputStream[Config.NUM_WORKERS];
            this.inputs = new ObjectInputStream[Config.NUM_WORKERS];
            System.out.println("[Master] Server started on port " + Config.MASTER_PORT);
        }

        public void acceptWorkers() throws IOException {
            System.out.println("[Master] Waiting for " + Config.NUM_WORKERS + " workers...");
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                Socket socket = serverSocket.accept();
                ObjectOutputStream out = new ObjectOutputStream(socket.getOutputStream());
                out.flush();
                ObjectInputStream in = new ObjectInputStream(socket.getInputStream());
                
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
                System.out.println("[Master] Worker " + workerId + " connected.");
            }
            System.out.println("[Master] All " + Config.NUM_WORKERS + " workers connected.");
        }

        public void sendInit(int workerId, MessageProtocol.InitMsg msg) throws IOException {
            outputs[workerId].writeObject(msg);
            outputs[workerId].flush();
            outputs[workerId].reset(); 
        }

        public void broadcast(MessageProtocol.BroadcastMsg msg) throws IOException {
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                outputs[i].writeObject(msg);
                outputs[i].flush();
                outputs[i].reset();
            }
        }

        /**
         * Collect coded gradient responses concurrently.
         * Master stops waiting the exact moment MIN_RESPONSES (3) are received.
         */
        public List<MessageProtocol.GradReturnMsg> collectGradients(int timeoutMs) {
            List<MessageProtocol.GradReturnMsg> results = new CopyOnWriteArrayList<>();
            CountDownLatch latch = new CountDownLatch(Config.MIN_RESPONSES);
            Thread[] readers = new Thread[Config.NUM_WORKERS];

            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                final int workerIdx = i;
                readers[i] = new Thread(() -> {
                    try {
                        workerSockets[workerIdx].setSoTimeout(timeoutMs);
                        Object obj = inputs[workerIdx].readObject();
                        if (obj instanceof MessageProtocol.GradReturnMsg) {
                            results.add((MessageProtocol.GradReturnMsg) obj);
                            latch.countDown(); // Signal that a response arrived
                        }
                    } catch (SocketTimeoutException e) {
                        System.out.println("[Master] ⚠ Worker " + workerIdx + " timed out.");
                    } catch (Exception e) {
                        // Ignore standard connection resets
                    }
                });
                readers[i].start();
            }

            try {
                // Wait for EXACTLY MIN_RESPONSES, or timeout.
                latch.await(timeoutMs, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            // Interrupt any reader threads still waiting for the straggler
            for (Thread t : readers) {
                if (t != null && t.isAlive()) {
                    t.interrupt();
                }
            }

            return new ArrayList<>(results);
        }

        @Override
        public void close() throws IOException {
            for (int i = 0; i < Config.NUM_WORKERS; i++) {
                if (workerSockets[i] != null && !workerSockets[i].isClosed()) {
                    try {
                        workerSockets[i].close();
                    } catch (IOException e) {
                        System.err.println("[Master] Error closing socket: " + e.getMessage());
                    }
                }
            }
            if (!serverSocket.isClosed()) {
                serverSocket.close();
            }
            System.out.println("[Master] Server shut down.");
        }
    }

    public static class WorkerClient implements Closeable {
        private final int workerId;
        private final Socket socket;
        private final ObjectOutputStream output;
        private final ObjectInputStream input;

        public WorkerClient(int workerId) throws IOException {
            this.workerId = workerId;
            this.socket = new Socket(Config.MASTER_HOST, Config.MASTER_PORT);
            this.output = new ObjectOutputStream(socket.getOutputStream());
            this.output.flush();
            this.input = new ObjectInputStream(socket.getInputStream());
            this.output.writeInt(workerId);
            this.output.flush();
            System.out.println("[Worker " + workerId + "] Connected to Master.");
        }

        public MessageProtocol.InitMsg receiveInit() throws IOException, ClassNotFoundException {
            Object obj = input.readObject();
            if (obj instanceof MessageProtocol.InitMsg) {
                return (MessageProtocol.InitMsg) obj;
            }
            throw new IOException("Expected InitMsg");
        }

        public MessageProtocol.BroadcastMsg receiveBroadcast() throws IOException, ClassNotFoundException {
            Object obj = input.readObject();
            if (obj instanceof MessageProtocol.BroadcastMsg) {
                return (MessageProtocol.BroadcastMsg) obj;
            }
            throw new IOException("Expected BroadcastMsg");
        }

        public void sendGradient(MessageProtocol.GradReturnMsg msg) throws IOException {
            output.writeObject(msg);
            output.flush();
            output.reset();
        }

        @Override
        public void close() throws IOException {
            if (socket != null && !socket.isClosed()) {
                socket.close();
                System.out.println("[Worker " + workerId + "] Disconnected.");
            }
        }
    }
}