package dsg;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * CSV file loader for shard data and test data.
 * 
 * Responsibilities (Affan Ahmed Basra — 476173):
 * 
 * 1. loadShard(path) — reads a shard CSV, returns CsvData with X and y
 * 2. loadFeatures(path) — reads feature-only CSV (no target column)
 * 3. loadTargets(path) — reads target-only CSV (inflation column)
 * 
 * CSV format for shards:
 *   - First row: comma-separated column headers
 *   - Data rows: comma-separated doubles
 *   - Columns: 11 features + 1 target (inflation), in the order defined by Config.FEATURE_NAMES + Config.TARGET_NAME
 *   - No internal commas or quotes (simple format)
 */
public final class CsvLoader {

    private CsvLoader() {} // prevent instantiation

    /**
     * Container for loaded shard data.
     */
    public static class CsvData {
        private final double[][] X;  // features [numRows][NUM_FEATURES]
        private final double[] y;    // targets [numRows]

        public CsvData(double[][] X, double[] y) {
            this.X = X;
            this.y = y;
        }

        public double[][] getX() { return X; }
        public double[] getY() { return y; }
        public int getNumRows() { return X.length; }
        public int getNumFeatures() { return X.length > 0 ? X[0].length : 0; }

        @Override
        public String toString() {
            return String.format("CsvData[%d rows × %d features, target]",
                getNumRows(), getNumFeatures());
        }
    }

    /**
     * Load a shard CSV file (features + target in one file).
     * 
     * Expected format:
     *   - Row 0: headers (comma-separated)
     *   - Rows 1+: data (doubles, comma-separated)
     *   - Last column is always the target (inflation)
     *   - First 11 columns are features
     * 
     * @param path file path to the shard CSV
     * @return CsvData with X [numRows × 11] and y [numRows]
     * @throws IOException if file cannot be read
     * @throws IllegalArgumentException if format is invalid
     */
    public static CsvData loadShard(String path) throws IOException {
        List<double[]> xList = new ArrayList<>();
        List<Double> yList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("CSV file is empty: " + path);
            }

            String[] headers = headerLine.split(",");
            if (headers.length != Config.NUM_FEATURES + 1) {
                throw new IllegalArgumentException(String.format(
                    "Expected %d columns (11 features + 1 target) in %s, got %d columns",
                    Config.NUM_FEATURES + 1, path, headers.length));
            }

            String line;
            int rowNum = 1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue; // skip blank lines
                }

                String[] parts = line.split(",");
                if (parts.length != headers.length) {
                    throw new IllegalArgumentException(String.format(
                        "Row %d: expected %d columns, got %d",
                        rowNum, headers.length, parts.length));
                }

                try {
                    double[] features = new double[Config.NUM_FEATURES];
                    for (int i = 0; i < Config.NUM_FEATURES; i++) {
                        features[i] = Double.parseDouble(parts[i]);
                    }
                    xList.add(features);

                    double target = Double.parseDouble(parts[Config.NUM_FEATURES]);
                    yList.add(target);

                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(String.format(
                        "Row %d: non-numeric value: %s",
                        rowNum, e.getMessage()));
                }
                rowNum++;
            }
        }

        if (xList.isEmpty()) {
            throw new IllegalArgumentException("CSV file has no data rows: " + path);
        }

        double[][] X = xList.toArray(new double[0][]);
        double[] y = new double[yList.size()];
        for (int i = 0; i < yList.size(); i++) {
            y[i] = yList.get(i);
        }

        return new CsvData(X, y);
    }

    /**
     * Load a feature-only CSV file (no target column).
     * 
     * Expected format:
     *   - Row 0: headers (comma-separated)
     *   - Rows 1+: data (doubles, comma-separated)
     *   - Exactly NUM_FEATURES columns
     * 
     * @param path file path to the features CSV (e.g., X_test.csv)
     * @return features array [numRows × NUM_FEATURES]
     * @throws IOException if file cannot be read
     * @throws IllegalArgumentException if format is invalid
     */
    public static double[][] loadFeatures(String path) throws IOException {
        List<double[]> xList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("CSV file is empty: " + path);
            }

            String[] headers = headerLine.split(",");
            if (headers.length != Config.NUM_FEATURES) {
                throw new IllegalArgumentException(String.format(
                    "Expected %d columns in %s, got %d columns",
                    Config.NUM_FEATURES, path, headers.length));
            }

            String line;
            int rowNum = 1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length != headers.length) {
                    throw new IllegalArgumentException(String.format(
                        "Row %d: expected %d columns, got %d",
                        rowNum, headers.length, parts.length));
                }

                try {
                    double[] features = new double[Config.NUM_FEATURES];
                    for (int i = 0; i < Config.NUM_FEATURES; i++) {
                        features[i] = Double.parseDouble(parts[i]);
                    }
                    xList.add(features);

                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(String.format(
                        "Row %d: non-numeric value: %s",
                        rowNum, e.getMessage()));
                }
                rowNum++;
            }
        }

        if (xList.isEmpty()) {
            throw new IllegalArgumentException("CSV file has no data rows: " + path);
        }

        return xList.toArray(new double[0][]);
    }

    /**
     * Load a target-only CSV file (single column).
     * 
     * Expected format:
     *   - Row 0: header (single column name)
     *   - Rows 1+: data (doubles, one per row)
     * 
     * @param path file path to the targets CSV (e.g., y_test.csv)
     * @return target values [numRows]
     * @throws IOException if file cannot be read
     * @throws IllegalArgumentException if format is invalid
     */
    public static double[] loadTargets(String path) throws IOException {
        List<Double> yList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IllegalArgumentException("CSV file is empty: " + path);
            }

            String line;
            int rowNum = 1;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                try {
                    double value = Double.parseDouble(line);
                    yList.add(value);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(String.format(
                        "Row %d: non-numeric value: %s",
                        rowNum, e.getMessage()));
                }
                rowNum++;
            }
        }

        if (yList.isEmpty()) {
            throw new IllegalArgumentException("CSV file has no data rows: " + path);
        }

        double[] y = new double[yList.size()];
        for (int i = 0; i < yList.size(); i++) {
            y[i] = yList.get(i);
        }

        return y;
    }
}
