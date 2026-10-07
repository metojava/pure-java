package com.neuralscratch.math;

import java.util.Random;

public final class MathChecks {
    private static final double TOLERANCE = 1e-8;

    private MathChecks() {
    }

    public static void main(String[] args) {
        double[] original = {3, 4};
        Vector vector = new Vector(original);
        original[0] = 99;
        close(3, vector.get(0));
        double[] exported = vector.toArray();
        exported[0] = 99;
        close(3, vector.get(0));
        close(5, vector.magnitude());
        vectorClose(new double[] {0.6, 0.8}, vector.normalize());
        close(11, new Vector(new double[] {1, 2}).dot(
            new Vector(new double[] {3, 4})));
        vectorClose(new double[] {4, 6},
            new Vector(new double[] {1, 2}).add(new Vector(new double[] {3, 4})));
        vectorClose(new double[] {-2, -2},
            new Vector(new double[] {1, 2}).subtract(new Vector(new double[] {3, 4})));
        vectorClose(new double[] {2, 4}, new Vector(new double[] {1, 2}).scale(2));

        double[][] source = {{1, 2, 3}, {4, 5, 6}};
        Matrix left = Matrix.of(source);
        source[0][0] = 99;
        close(1, left.get(0, 0));
        Matrix right = Matrix.of(new double[][] {{7, 8}, {9, 10}, {11, 12}});
        matrixClose(new double[][] {{58, 64}, {139, 154}}, left.multiply(right));
        matrixClose(new double[][] {{1, 4}, {2, 5}, {3, 6}}, left.transpose());
        matrixClose(new double[][] {{1, 2}, {3, 4}, {5, 6}}, left.reshape(3, 2));
        matrixClose(sourceWithoutMutation(), left.transpose().transpose());
        vectorClose(new double[] {3, 4}, Matrix.column(vector).toColumnVector());

        Matrix first = Matrix.of(new double[][] {{1, 2}, {3, 4}});
        Matrix second = Matrix.of(new double[][] {{2, 3}, {4, 5}});
        matrixClose(new double[][] {{3, 5}, {7, 9}}, first.add(second));
        matrixClose(new double[][] {{-1, -1}, {-1, -1}}, first.subtract(second));
        matrixClose(new double[][] {{2, 6}, {12, 20}}, first.hadamard(second));
        matrixClose(new double[][] {{2, 4}, {6, 8}}, first.scale(2));
        matrixClose(new double[][] {{1, 4}, {9, 16}}, first.apply(x -> x * x));
        matrixClose(new double[][] {{11, 22}, {13, 24}},
            first.addRowVector(new Vector(new double[] {10, 20})));

        close(6, Calculus.derivative(x -> x * x, 3), 1e-5);
        vectorClose(new double[] {6, 8}, Calculus.gradient(
            point -> point.get(0) * point.get(0)
                + point.get(1) * point.get(1),
            new Vector(new double[] {3, 4})), 1e-5);

        Random random = new Random(42);
        for (int trial = 0; trial < 50; trial++) {
            int rows = 1 + random.nextInt(4);
            int cols = 1 + random.nextInt(4);
            double[][] values = new double[rows][cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    values[row][col] = random.nextDouble();
                }
            }
            Matrix sample = Matrix.of(values);
            matrixClose(values, sample.transpose().transpose());
            matrixClose(values, sample.reshape(cols, rows).reshape(rows, cols));
        }

        expectIllegalArgument(() -> Matrix.of(new double[][] {{1}, {2, 3}}));
        expectIllegalArgument(() -> left.multiply(first));
        expectIllegalArgument(() -> left.reshape(4, 2));
        expectIllegalArgument(() -> first.addRowVector(new Vector(new double[] {1})));
        expectIllegalArgument(() -> vector.dot(new Vector(new double[] {1})));
        try {
            Vector.zeros(1).normalize();
            throw new AssertionError("Expected zero-vector normalization to fail");
        } catch (ArithmeticException expected) {
            // The zero vector has no direction.
        }
        System.out.println("All math checks passed");
    }

    private static double[][] sourceWithoutMutation() {
        return new double[][] {{1, 2, 3}, {4, 5, 6}};
    }

    private static void close(double expected, double actual) {
        close(expected, actual, TOLERANCE);
    }

    private static void close(double expected, double actual, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > tolerance) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void vectorClose(double[] expected, Vector actual) {
        vectorClose(expected, actual, TOLERANCE);
    }

    private static void vectorClose(double[] expected, Vector actual, double tolerance) {
        if (expected.length != actual.size()) {
            throw new AssertionError("Vector size mismatch");
        }
        for (int index = 0; index < expected.length; index++) {
            close(expected[index], actual.get(index), tolerance);
        }
    }

    private static void matrixClose(double[][] expected, Matrix actual) {
        if (expected.length != actual.rows() || expected[0].length != actual.cols()) {
            throw new AssertionError("Matrix shape mismatch");
        }
        for (int row = 0; row < actual.rows(); row++) {
            for (int col = 0; col < actual.cols(); col++) {
                close(expected[row][col], actual.get(row, col));
            }
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // Invalid shapes must fail before any arithmetic begins.
        }
    }
}