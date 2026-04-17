package dsg;

import java.io.*;
import java.net.*;
import java.util.Arrays;

/**
 * Smoke tests for Affan's deliverables.
 * Tests: Config, CsvLoader, MessageProtocol serialization, SocketManager, StragglerInjector.
 * 
 * Run after compile:  java -cp bin dsg.SmokeTest
 */
public class SmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println();
        System.out.println("======================================================");
        System.out.println("        DSG Smoke Tests - Affan's Code");
        System.out.println("======================================================");
        System.out.println();

        testConfig();
        testCsvLoaderShards();
        testCsvLoaderTestData();
        testMessageProtocolSerialization();
        testSocketConnection();
        testStragglerInjector();

        System.out.println();
        System.out.println("======================================================");
        System.out.printf("  RESULTS: %d passed, %d failed, %d total%n", passed, failed, passed + failed);
        System.out.println("======================================================");

        if (failed > 0) {
            System.exit(1);
        }
    }

    // ── Config Tests ─────────────────────────────────────────────────

    private static void testConfig() {
        section("Config.java");

        check("NUM_FEATURES == 11", Config.NUM_FEATURES == 11);
        check("NUM_WORKERS == 4", Config.NUM_WORKERS == 4);
        check("NUM_SHARDS == 4", Config.NUM_SHARDS == 4);
        check("MIN_RESPONSES == 3", Config.MIN_RESPONSES == 3);
        check("STRAGGLER_TIMEOUT_MS == 5000", Config.STRAGGLER_TIMEOUT_MS == 5000);
        check("CODING_MATRIX is 4x4",
            Config.CODING_MATRIX.length == 4 && Config.CODING_MATRIX[0].length == 4);
        check("SHARD_ASSIGNMENTS is 4x2",
            Config.SHARD_ASSIGNMENTS.length == 4 && Config.SHARD_ASSIGNMENTS[0].length == 2);
        check("FEATURE_NAMES has 11 entries", Config.FEATURE_NAMES.length == 11);
        check("shardPath(0) correct", Config.shardPath(0).equals("output/shards/shard_0.csv"));
        check("shardPath(3) correct", Config.shardPath(3).equals("output/shards/shard_3.csv"));

        // Verify coding matrix: each row sums to 2 (each worker has 2 shards)
        for (int w = 0; w < 4; w++) {
            double rowSum = 0;
            for (double v : Config.CODING_MATRIX[w]) rowSum += v;
            check("CODING_MATRIX row " + w + " sums to 2.0", Math.abs(rowSum - 2.0) < 1e-9);
        }

        // Verify coding matrix: each column sums to 2 (each shard appears in 2 workers)
        for (int s = 0; s < 4; s++) {
            double colSum = 0;
            for (int w = 0; w < 4; w++) colSum += Config.CODING_MATRIX[w][s];
            check("CODING_MATRIX col " + s + " sums to 2.0", Math.abs(colSum - 2.0) < 1e-9);
        }

        // Verify GC-DC recovery: dropping any 1 worker, remaining can recover full gradient
        // For cyclic shift: sum of workers {i, i+2} always equals the full sum
        check("GC-DC recovery: drop worker 0 -> sum(coded1,coded3) = full",
            Config.CODING_MATRIX[1][0] + Config.CODING_MATRIX[3][0] == 1 &&
            Config.CODING_MATRIX[1][1] + Config.CODING_MATRIX[3][1] == 1 &&
            Config.CODING_MATRIX[1][2] + Config.CODING_MATRIX[3][2] == 1 &&
            Config.CODING_MATRIX[1][3] + Config.CODING_MATRIX[3][3] == 1);
        check("GC-DC recovery: drop worker 1 -> sum(coded0,coded2) = full",
            Config.CODING_MATRIX[0][0] + Config.CODING_MATRIX[2][0] == 1 &&
            Config.CODING_MATRIX[0][1] + Config.CODING_MATRIX[2][1] == 1 &&
            Config.CODING_MATRIX[0][2] + Config.CODING_MATRIX[2][2] == 1 &&
            Config.CODING_MATRIX[0][3] + Config.CODING_MATRIX[2][3] == 1);
    }

    // ── CsvLoader Tests ──────────────────────────────────────────────

    private static void testCsvLoaderShards() throws Exception {
        section("CsvLoader - Shard Loading");

        for (int i = 0; i < Config.NUM_SHARDS; i++) {
            String path = Config.shardPath(i);
            File f = new File(path);
            check("shard_" + i + ".csv exists", f.exists());

            if (f.exists()) {
                CsvLoader.CsvData data = CsvLoader.loadShard(path);
                check("shard_" + i + " rows > 0", data.getNumRows() > 0);
                check("shard_" + i + " has " + Config.NUM_FEATURES + " features",
                    data.getNumFeatures() == Config.NUM_FEATURES);
                check("shard_" + i + " X.length == y.length",
                    data.getX().length == data.getY().length);
                check("shard_" + i + " ~9854 rows (within 100)",
                    Math.abs(data.getNumRows() - 9854) < 100);
                System.out.println("       -> " + data);
            }
        }
    }

    private static void testCsvLoaderTestData() throws Exception {
        section("CsvLoader - Test Data Loading");

        File xFile = new File(Config.TEST_X_PATH);
        File yFile = new File(Config.TEST_Y_PATH);
        check("X_test.csv exists", xFile.exists());
        check("y_test.csv exists", yFile.exists());

        if (xFile.exists() && yFile.exists()) {
            double[][] xTest = CsvLoader.loadFeatures(Config.TEST_X_PATH);
            double[] yTest = CsvLoader.loadTargets(Config.TEST_Y_PATH);
            check("X_test rows > 0", xTest.length > 0);
            check("X_test has " + Config.NUM_FEATURES + " features",
                xTest[0].length == Config.NUM_FEATURES);
            check("y_test row count matches X_test", yTest.length == xTest.length);
            System.out.println("       -> X_test: " + xTest.length + " rows x " + xTest[0].length + " features");
            System.out.println("       -> y_test: " + yTest.length + " rows");
        }
    }

    // ── MessageProtocol Serialization Tests ──────────────────────────

    private static void testMessageProtocolSerialization() throws Exception {
        section("MessageProtocol - Serialization Roundtrip");

        // Test InitMsg
        {
            double[][] s1X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            double[]   s1Y = {0.5, 0.6};
            double[][] s2X = {{7.0, 8.0, 9.0}};
            double[]   s2Y = {0.7};
            double[] weights = {0.1, 0.2, 0.3, 0.0};

            MessageProtocol.InitMsg original = new MessageProtocol.InitMsg(
                2, s1X, s1Y, s2X, s2Y, 1.0, 1.0, weights);

            MessageProtocol.InitMsg restored = (MessageProtocol.InitMsg) roundtrip(original);

            check("InitMsg: workerId preserved", restored.getWorkerId() == 2);
            check("InitMsg: shard1X rows preserved", restored.getShard1X().length == 2);
            check("InitMsg: shard1X[0][0] value correct", restored.getShard1X()[0][0] == 1.0);
            check("InitMsg: shard2X rows preserved", restored.getShard2X().length == 1);
            check("InitMsg: shard1Y preserved", Arrays.equals(restored.getShard1Y(), s1Y));
            check("InitMsg: shard2Y preserved", Arrays.equals(restored.getShard2Y(), s2Y));
            check("InitMsg: shard1Coeff == 1.0", restored.getShard1Coeff() == 1.0);
            check("InitMsg: shard2Coeff == 1.0", restored.getShard2Coeff() == 1.0);
            check("InitMsg: weights preserved", Arrays.equals(restored.getInitialWeights(), weights));
            check("InitMsg: toString works", restored.toString().contains("worker=2"));
        }

        // Test BroadcastMsg
        {
            double[] weights = {0.5, -0.3, 1.2, 0.0};
            MessageProtocol.BroadcastMsg original = new MessageProtocol.BroadcastMsg(weights, 42);
            MessageProtocol.BroadcastMsg restored = (MessageProtocol.BroadcastMsg) roundtrip(original);

            check("BroadcastMsg: epoch preserved", restored.getEpoch() == 42);
            check("BroadcastMsg: weights preserved", Arrays.equals(restored.getWeights(), weights));
            check("BroadcastMsg: toString contains epoch", restored.toString().contains("epoch=42"));
        }

        // Test GradReturnMsg
        {
            double[] grad = {-0.1, 0.2, -0.3, 0.01};
            MessageProtocol.GradReturnMsg original = new MessageProtocol.GradReturnMsg(3, grad, 7);
            MessageProtocol.GradReturnMsg restored = (MessageProtocol.GradReturnMsg) roundtrip(original);

            check("GradReturnMsg: workerId preserved", restored.getWorkerId() == 3);
            check("GradReturnMsg: epoch preserved", restored.getEpoch() == 7);
            check("GradReturnMsg: gradient preserved", Arrays.equals(restored.getCodedGradient(), grad));
            check("GradReturnMsg: toString contains worker", restored.toString().contains("worker=3"));
        }
    }

    // ── SocketManager Connection Test ────────────────────────────────

    private static void testSocketConnection() throws Exception {
        section("SocketManager - TCP Connection + Message Exchange");

        // Use a different port to avoid conflicts
        final int TEST_PORT = Config.MASTER_PORT + 1;
        final MessageProtocol.GradReturnMsg[] received = {null};
        final Exception[] serverError = {null};

        // Start a raw server thread (simulates Master)
        Thread serverThread = new Thread(() -> {
            try {
                ServerSocket ss = new ServerSocket(TEST_PORT);
                Socket sock = ss.accept();

                ObjectOutputStream out = new ObjectOutputStream(sock.getOutputStream());
                out.flush();
                ObjectInputStream in = new ObjectInputStream(sock.getInputStream());

                // Read worker handshake ID
                int wId = in.readInt();

                // Send a BroadcastMsg
                out.writeObject(new MessageProtocol.BroadcastMsg(
                    new double[]{0.1, 0.2, 0.3}, 0));
                out.flush();

                // Receive response
                sock.setSoTimeout(5000);
                Object obj = in.readObject();
                if (obj instanceof MessageProtocol.GradReturnMsg) {
                    received[0] = (MessageProtocol.GradReturnMsg) obj;
                }

                sock.close();
                ss.close();
            } catch (Exception e) {
                serverError[0] = e;
            }
        });
        serverThread.start();
        Thread.sleep(500);

        // Connect as worker (raw socket, same protocol as WorkerClient)
        Socket clientSock = new Socket("localhost", TEST_PORT);
        ObjectOutputStream clientOut = new ObjectOutputStream(clientSock.getOutputStream());
        clientOut.flush();
        ObjectInputStream clientIn = new ObjectInputStream(clientSock.getInputStream());

        // Send handshake
        clientOut.writeInt(0);
        clientOut.flush();

        // Receive broadcast
        Object obj = clientIn.readObject();
        boolean isBroadcast = obj instanceof MessageProtocol.BroadcastMsg;
        check("Socket: received BroadcastMsg", isBroadcast);

        if (isBroadcast) {
            MessageProtocol.BroadcastMsg bm = (MessageProtocol.BroadcastMsg) obj;
            check("Socket: broadcast epoch == 0", bm.getEpoch() == 0);
            check("Socket: broadcast weights length == 3", bm.getWeights().length == 3);
        }

        // Send gradient response
        clientOut.writeObject(new MessageProtocol.GradReturnMsg(
            0, new double[]{-0.01, -0.02, -0.03}, 0));
        clientOut.flush();

        clientSock.close();
        serverThread.join(3000);

        check("Socket: server received GradReturnMsg", received[0] != null);
        if (received[0] != null) {
            check("Socket: received workerId == 0", received[0].getWorkerId() == 0);
            check("Socket: received gradient[0] == -0.01",
                Math.abs(received[0].getCodedGradient()[0] - (-0.01)) < 1e-9);
        }
        check("Socket: no server errors", serverError[0] == null);
        if (serverError[0] != null) {
            System.err.println("       Server error: " + serverError[0].getMessage());
        }
    }

    // ── StragglerInjector Test ───────────────────────────────────────

    private static void testStragglerInjector() {
        section("StragglerInjector");

        // Worker 0: should NOT be delayed
        long start = System.currentTimeMillis();
        StragglerInjector.maybeDelay(0);
        long elapsed = System.currentTimeMillis() - start;
        check("Worker 0: no delay (< 100ms)", elapsed < 100);

        // Worker 2: should NOT be delayed
        start = System.currentTimeMillis();
        StragglerInjector.maybeDelay(2);
        elapsed = System.currentTimeMillis() - start;
        check("Worker 2: no delay (< 100ms)", elapsed < 100);

        // Worker 3: SHOULD be delayed 2-5 seconds
        start = System.currentTimeMillis();
        StragglerInjector.maybeDelay(3);
        elapsed = System.currentTimeMillis() - start;
        check("Worker 3: delayed >= 1900ms", elapsed >= 1900);
        check("Worker 3: delayed <= 5500ms", elapsed <= 5500);
        System.out.println("       -> Actual straggler delay: " + elapsed + "ms");
    }

    // ── Utilities ────────────────────────────────────────────────────

    private static Object roundtrip(Object obj) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(obj);
        oos.close();

        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        ObjectInputStream ois = new ObjectInputStream(bais);
        return ois.readObject();
    }

    private static void section(String name) {
        System.out.println("-- " + name + " ------------------------------------");
    }

    private static void check(String name, boolean condition) {
        if (condition) {
            System.out.println("  [PASS] " + name);
            passed++;
        } else {
            System.out.println("  [FAIL] " + name);
            failed++;
        }
    }
}
