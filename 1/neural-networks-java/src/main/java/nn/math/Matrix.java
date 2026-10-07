package nn.math;

import java.util.Random;

public final class Matrix {
    private final double[][] values;
    public Matrix(int rows, int columns) { values = new double[rows][columns]; }
    public static Matrix random(int rows, int columns, Random random) {
        Matrix matrix = new Matrix(rows, columns);
        double scale = Math.sqrt(2.0 / (rows + columns));
        for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) matrix.values[r][c] = random.nextGaussian() * scale;
        return matrix;
    }
    public int rows() { return values.length; }
    public int columns() { return values[0].length; }
    public double get(int row, int column) { return values[row][column]; }
    public void addTo(int row, int column, double amount) { values[row][column] += amount; }
    public Vector multiply(Vector input) {
        if (input.size() != columns()) throw new IllegalArgumentException("matrix/input shape mismatch");
        double[] output = new double[rows()];
        for (int r = 0; r < rows(); r++) for (int c = 0; c < columns(); c++) output[r] += values[r][c] * input.get(c);
        return new Vector(output);
    }
}
