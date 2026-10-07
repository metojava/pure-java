package com.neuralscratch.nn;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import java.util.Random;

public final class NetworkChecks {
    private static final double TOLERANCE = 1e-9;

    private NetworkChecks() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{2, 3}, {1, 1}});
        DenseLayer linear = new DenseLayer(
            Matrix.of(new double[][] {{1, 0, 2}, {4, 5, 2}}),
            new Vector(new double[] {1, -1, 0}),
            new Identity());
        matrixClose(new double[][] {{15, 14, 10}, {6, 4, 4}}, linear.forward(inputs));
        check(linear.inputSize() == 2 && linear.outputSize() == 3, "layer sizes");

        DenseLayer squashed = new DenseLayer(linear.weights(), linear.biases(), new Sigmoid());
        Sigmoid sigmoid = new Sigmoid();
        matrixClose(new double[][] {
            {sigmoid.apply(15), sigmoid.apply(14), sigmoid.apply(10)},
            {sigmoid.apply(6), sigmoid.apply(4), sigmoid.apply(4)}
        }, squashed.forward(inputs));

        Network xor = new Network(
            new DenseLayer(Matrix.of(new double[][] {{1, -1}, {1, -1}}),
                new Vector(new double[] {-0.5, 1.5}), new Step()),
            new DenseLayer(Matrix.of(new double[][] {{1}, {1}}),
                new Vector(new double[] {-1.5}), new Step()));
        Matrix xorInputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        matrixClose(new double[][] {{0}, {1}, {1}, {0}}, xor.predict(xorInputs));

        DenseLayer first = DenseLayer.random(2, 3, new Sigmoid(), new Random(7));
        DenseLayer second = DenseLayer.random(2, 3, new Sigmoid(), new Random(7));
        matrixClose(toArray(first.weights()), second.weights());
        check(first.weights().rows() == 2 && first.weights().cols() == 3, "random shape");
        for (int col = 0; col < 3; col++) {
            close(0.0, first.biases().get(col));
        }

        DenseLayer wide = DenseLayer.random(200, 300, new Identity(), new Random(1));
        double sumSquares = 0.0;
        for (int row = 0; row < 200; row++) {
            for (int col = 0; col < 300; col++) {
                sumSquares += wide.weights().get(row, col) * wide.weights().get(row, col);
            }
        }
        double observedStddev = Math.sqrt(sumSquares / (200 * 300));
        double expectedStddev = Math.sqrt(2.0 / 500);
        check(Math.abs(observedStddev - expectedStddev) < 0.05 * expectedStddev,
            "Xavier stddev " + observedStddev);

        Random random = new Random(42);
        Network network = new Network(
            DenseLayer.random(2, 3, new Sigmoid(), random),
            DenseLayer.random(3, 1, new Sigmoid(), random));
        Matrix predictions = network.predict(xorInputs);
        check(predictions.rows() == 4 && predictions.cols() == 1, "2x3x1 output shape");
        check(network.inputSize() == 2 && network.outputSize() == 1, "network sizes");
        check(network.layers().size() == 2, "layer count");

        DenseLayer trainable = new DenseLayer(
            Matrix.of(new double[][] {{1, 2}, {3, 4}}),
            new Vector(new double[] {0.5, -0.5}),
            new Identity());
        trainable.applyGradients(
            Matrix.of(new double[][] {{10, 0}, {-10, 20}}),
            new Vector(new double[] {1, -1}),
            0.1);
        matrixClose(new double[][] {{0, 2}, {4, 2}}, trainable.weights());
        close(0.4, trainable.biases().get(0));
        close(-0.4, trainable.biases().get(1));

        Matrix goodGradient = Matrix.of(new double[][] {{1, 1}, {1, 1}});
        Vector goodBias = new Vector(new double[] {1, 1});
        expectIllegalArgument(() -> trainable.applyGradients(
            Matrix.of(new double[][] {{1, 1, 1}, {1, 1, 1}}), goodBias, 0.1));
        expectIllegalArgument(() -> trainable.applyGradients(
            goodGradient, new Vector(new double[] {1}), 0.1));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, 0.0));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, -1.0));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, Double.NaN));
        matrixClose(new double[][] {{0, 2}, {4, 2}}, trainable.weights());

        expectIllegalArgument(() -> new DenseLayer(
            Matrix.of(new double[][] {{1, 2}}), new Vector(new double[] {1}), new Identity()));
        expectIllegalArgument(() -> linear.forward(Matrix.of(new double[][] {{1, 2, 3}})));
        expectIllegalArgument(() -> DenseLayer.random(0, 3, new Sigmoid(), new Random(1)));
        expectIllegalArgument(() -> new Network());
        expectIllegalArgument(() -> new Network(
            DenseLayer.random(2, 3, new Sigmoid(), new Random(1)),
            DenseLayer.random(4, 1, new Sigmoid(), new Random(1))));
        expectIllegalArgument(() -> network.predict(Matrix.of(new double[][] {{1, 2, 3}})));
        try {
            network.layers().clear();
            throw new AssertionError("layers() must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            // The validated layer sequence cannot be changed.
        }
        System.out.println("All layer and network checks passed");
    }

    private static double[][] toArray(Matrix matrix) {
        double[][] values = new double[matrix.rows()][matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row][col] = matrix.get(row, col);
            }
        }
        return values;
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError("Check failed: " + message);
        }
    }

    private static void close(double expected, double actual) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > TOLERANCE) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
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
            // Invalid shapes and rates are rejected before any state changes.
        }
    }
}