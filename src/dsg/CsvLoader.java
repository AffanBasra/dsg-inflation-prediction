package dsg;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for loading CSV data files produced by the Python data pipeline.
 * 
 * No external dependencies — uses BufferedReader + String.split for parsing.
 * Handles three CSV formats:
 *   1. Shard files: features + target in last column (loadShard)
 *   2. Feature-only files: X_test.csv (loadFeatures)
 *   3. Single-column target files: y_test.csv (loadTargets)
 * 
 * @author Affan Ahmed Basra (476173)
 */
public final class CsvLoader {

    private CsvLoader() {} // prevent instantiation

    // ═════════════════════════════════════════════════════════════════
    // CsvData — Container for shard data (features + target)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Holds the parsed contents of a shard CSV file.
     */
    public static class CsvData {
        private final double[][] X;           // features [rows][features]
        private final double[]   y;           // target values [rows]
        private final String[]   featureNames; // column headers (excluding target)

        public CsvData(double[][] X, double[] y, String[] featureNames) {
            this.X = X;
            this.y = y;
            this.featureNames = featureNames;
        }

        public double[][] getX()              { return X; }
        public double[]   getY()              { return y; }
        public String[]   getFeatureNames()   { return featureNames; }
        public int        getNumRows()        { return X.length; }
        public int        getNumFeatures()    { return X.length > 0 ? X[0].length : 0; }

        @Override
        public String toString() {
            return String.format("CsvData[%d rows × %d features]", getNumRows(), getNumFeatures());
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // loadShard — Load a shard CSV (features + target in last column)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Load a shard CSV file where the last column is the target (inflation)
     * and all preceding columns are features.
     * 
     * Expected format (from pipeline output):
     *   exchange_rate,gdp_growth,...,inflation_lag3,inflation
     *   -0.636,0.487,...,0.579,0.470
     *   ...
     * 
     * @param path relative or absolute path to the CSV file
     * @return CsvData with features and target separated
     * @throws IOException if file cannot be read or is empty
     */
    public static CsvData loadShard(String path) throws IOException {
        List<double[]> featureRows = new ArrayList<>();
        List<Double> targets = new ArrayList<>();
        String[] featureNames;

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            // Parse header
            String header = br.readLine();
            if (header == null) {
                throw new IOException("Empty CSV file: " + path);
            }

            String[] columns = header.split(",");
            int numCols = columns.length;

            // All columns except the last are features
            featureNames = new String[numCols - 1];
            for (int i = 0; i < numCols - 1; i++) {
                featureNames[i] = columns[i].trim();
            }

            // Parse data rows
            String line;
            int lineNum = 1;
            while ((line = br.readLine()) != null) {
                lineNum++;
                if (line.trim().isEmpty()) continue;

                String[] parts = line.split(",");
                if (parts.length != numCols) {
                    throw new IOException(String.format(
                        "Column count mismatch at %s line %d: expected %d, got %d",
                        path, lineNum, numCols, parts.length));
                }

                double[] features = new double[numCols - 1];
                for (int i = 0; i < numCols - 1; i++) {
                    features[i] = Double.parseDouble(parts[i].trim());
                }
                featureRows.add(features);
                targets.add(Double.parseDouble(parts[numCols - 1].trim()));
            }
        }

        double[][] X = featureRows.toArray(new double[0][]);
        double[] y = new double[targets.size()];
        for (int i = 0; i < targets.size(); i++) {
            y[i] = targets.get(i);
        }

        return new CsvData(X, y, featureNames);
    }

    // ═════════════════════════════════════════════════════════════════
    // loadFeatures — Load a features-only CSV (e.g., X_test.csv)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Load a CSV containing only feature columns (no target).
     * Used for X_test.csv.
     * 
     * @param path relative or absolute path to the CSV file
     * @return feature matrix [rows][features]
     * @throws IOException if file cannot be read
     */
    public static double[][] loadFeatures(String path) throws IOException {
        List<double[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // skip header

            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i].trim());
                }
                rows.add(row);
            }
        }

        return rows.toArray(new double[0][]);
    }

    // ═════════════════════════════════════════════════════════════════
    // loadTargets — Load a single-column target CSV (e.g., y_test.csv)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Load a CSV containing a single target column.
     * Used for y_test.csv.
     * 
     * @param path relative or absolute path to the CSV file
     * @return target values array [rows]
     * @throws IOException if file cannot be read
     */
    public static double[] loadTargets(String path) throws IOException {
        List<Double> values = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // skip header

            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                values.add(Double.parseDouble(line.trim()));
            }
        }

        double[] result = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            result[i] = values.get(i);
        }
        return result;
    }
}
