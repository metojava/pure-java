package com.neuralscratch.math;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

public final class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be positive");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public static Matrix of(double[][] values) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException("Matrix must have positive dimensions");
        }
        Matrix result = new Matrix(values.length, values[0].length);
        for (int row = 0; row < result.rows; row++) {
            if (values[row] == null || values[row].length != result.cols) {
                throw new IllegalArgumentException("All matrix rows must have equal length");
            }
            System.arraycopy(values[row], 0, result.data[row], 0, result.cols);
        }
        return result;
    }

    public static Matrix column(Vector vector) {
        Objects.requireNonNull(vector, "vector");
        Matrix result = new Matrix(vector.size(), 1);
        for (int row = 0; row < vector.size(); row++) {
            result.data[row][0] = vector.get(row);
        }
        return result;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public Vector toColumnVector() {
        if (cols != 1) {
            throw new IllegalStateException("Expected one column, got " + cols);
        }
        double[] values = new double[rows];
        for (int row = 0; row < rows; row++) {
            values[row] = data[row][0];
        }
        return new Vector(values);
    }

    public Matrix add(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] + other.data[row][col];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] - other.data[row][col];
            }
        }
        return result;
    }

    public Matrix scale(double factor) {
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] * factor;
            }
        }
        return result;
    }

    public Matrix hadamard(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Hadamard multiplication pairs entries at the same position.
                result.data[row][col] = data[row][col] * other.data[row][col];
            }
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (cols != other.rows) {
            throw new IllegalArgumentException(
                "Cannot multiply " + rows + "x" + cols + " by "
                    + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(rows, other.cols);
        for (int row = 0; row < rows; row++) {
            for (int shared = 0; shared < cols; shared++) {
                double left = data[row][shared];
                for (int col = 0; col < other.cols; col++) {
                    // Accumulate one row/column dot product in each output cell.
                    result.data[row][col] += left * other.data[shared][col];
                }
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[col][row] = data[row][col];
            }
        }
        return result;
    }

    public Matrix reshape(int newRows, int newCols) {
        if (newRows <= 0 || newCols <= 0
                || (long) newRows * newCols != (long) rows * cols) {
            throw new IllegalArgumentException("Reshape must preserve the entry count");
        }
        Matrix result = new Matrix(newRows, newCols);
        int index = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Keep row-major order while placing each entry in its new shape.
                result.data[index / newCols][index % newCols] = data[row][col];
                index++;
            }
        }
        return result;
    }

    public Matrix addRowVector(Vector bias) {
        Objects.requireNonNull(bias, "bias");
        if (bias.size() != cols) {
            throw new IllegalArgumentException(
                "Bias size " + bias.size() + " must equal column count " + cols
            );
        }
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] + bias.get(col);
            }
        }
        return result;
    }

    public Matrix apply(DoubleUnaryOperator function) {
        Objects.requireNonNull(function, "function");
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = function.applyAsDouble(data[row][col]);
            }
        }
        return result;
    }

    private void requireSameShape(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException(
                "Matrix shapes differ: " + rows + "x" + cols + " and "
                    + other.rows + "x" + other.cols
            );
        }
    }
}