package com.neuralscratch.training;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.Random;

public final class BackpropagationChecks {
    private static final double TOLERANCE = 1e-9;

    private BackpropagationChecks() {
    }

    public static void main(String[] args) {
        OutputObjective mse = new MeanSquaredError();
        Matrix output = Matrix.of(new double[][] {{1, 2}, {3, 4}});
        Matrix target = Matrix.of(new double[][] {{0, 2}, {3, 2}});
        close(1.25, mse.loss(output, output, target));
        matrixClose(new double[][] {{0.5, 0}, {0, 1}},
            mse.delta(output, output, target, new Identity()));
        Sigmoid sigmoid = new Sigmoid();
        Matrix logits = Matrix.of(new double[][] {{0.3, -1.2}});
        Matrix squashed = logits.apply(sigmoid::apply);
        Matrix goal = Matrix.of(new double[][] {{1, 0}});
        matrixClose(new double[][] {{
            (squashed.get(0, 0) - 1) * sigmoid.derivative(0.3),
            (squashed.get(0, 1) - 0) * sigmoid.derivative(-1.2)
        }}, mse.delta(logits, squashed, goal, sigmoid));
        expectIllegalArgument(() -> mse.loss(output, output, Matrix.of(new double[][] {{1, 2}})));

        Vector sums = Backpropagation.rowSum(output);
        close(4, sums.get(0));
        close(6, sums.get(1));

        DenseLayer single = new DenseLayer(
            Matrix.of(new double[][] {{0.5}}), new Vector(new double[] {-0.5}), sigmoid);
        double loss = Backpropagation.trainBatch(new Network(single),
            Matrix.of(new double[][] {{2}}), Matrix.of(new double[][] {{1}}), mse, 1.0);
        double a = sigmoid.apply(0.5);
        double delta = (a - 1) * a * (1 - a);
        close(0.5 * (a - 1) * (a - 1), loss);
        close(0.5 - 2 * delta, single.weights().get(0, 0));
        close(-0.5 - delta, single.biases().get(0));

        Random random = new Random(11);
        gradientCheck(new int[] {2, 3, 1}, randomMatrix(4, 2, random), randomMatrix(4, 1, random), random);
        gradientCheck(new int[] {3, 4, 3, 2}, randomMatrix(5, 3, random), randomMatrix(5, 2, random), random);

        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {1}, {1}, {1}});
        int[] sizes = {2, 3, 1};
        Network network = randomNetwork(sizes, new Random(5));
        double before = forwardLoss(network, inputs, targets, mse);
        double reported = Backpropagation.trainBatch(network, inputs, targets, mse, 0.5);
        close(before, reported);
        double after = forwardLoss(network, inputs, targets, mse);
        if (!(after < before)) {
            throw new AssertionError("A small step should lower the loss: " + before + " -> " + after);
        }

        Vector frozen = flatten(network);
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, Matrix.of(new double[][] {{1, 2, 3}}), Matrix.of(new double[][] {{1}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, inputs, Matrix.of(new double[][] {{1}, {0}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, inputs, Matrix.of(new double[][] {{1, 0}, {1, 0}, {1, 0}, {1, 0}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(network, inputs, targets, mse, 0.0));
        expectIllegalArgument(() -> Backpropagation.trainBatch(network, inputs, targets, mse, Double.NaN));
        vectorClose(frozen, flatten(network));
        System.out.println("All backpropagation checks passed");
    }

    private static void gradientCheck(int[] sizes, Matrix inputs, Matrix targets, Random random) {
        OutputObjective mse = new MeanSquaredError();
        Vector theta = flatten(randomNetwork(sizes, random));
        Network copy = build(sizes, theta);
        Backpropagation.trainBatch(copy, inputs, targets, mse, 1.0);
        Vector analytic = theta.subtract(flatten(copy));
        Vector numeric = Calculus.gradient(
            point -> forwardLoss(build(sizes, point), inputs, targets, mse), theta);
        for (int index = 0; index < theta.size(); index++) {
            if (Math.abs(analytic.get(index) - numeric.get(index)) > 1e-7) {
                throw new AssertionError("Gradient mismatch at parameter " + index + ": "
                    + analytic.get(index) + " vs " + numeric.get(index));
            }
        }
    }

    private static Matrix randomMatrix(int rows, int cols, Random random) {
        double[][] values = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                values[row][col] = random.nextDouble();
            }
        }
        return Matrix.of(values);
    }

    private static Network randomNetwork(int[] sizes, Random random) {
        DenseLayer[] layers = new DenseLayer[sizes.length - 1];
        for (int index = 0; index < layers.length; index++) {
            layers[index] = DenseLayer.random(sizes[index], sizes[index + 1], new Sigmoid(), random);
        }
        return new Network(layers);
    }

    private static Vector flatten(Network network) {
        int count = 0;
        for (DenseLayer layer : network.layers()) {
            count += layer.inputSize() * layer.outputSize() + layer.outputSize();
        }
        double[] theta = new double[count];
        int offset = 0;
        for (DenseLayer layer : network.layers()) {
            for (int row = 0; row < layer.inputSize(); row++) {
                for (int col = 0; col < layer.outputSize(); col++) {
                    theta[offset++] = layer.weights().get(row, col);
                }
            }
            for (int col = 0; col < layer.outputSize(); col++) {
                theta[offset++] = layer.biases().get(col);
            }
        }
        return new Vector(theta);
    }

    private static Network build(int[] sizes, Vector theta) {
        DenseLayer[] layers = new DenseLayer[sizes.length - 1];
        int offset = 0;
        for (int index = 0; index < layers.length; index++) {
            double[][] weights = new double[sizes[index]][sizes[index + 1]];
            for (int row = 0; row < sizes[index]; row++) {
                for (int col = 0; col < sizes[index + 1]; col++) {
                    weights[row][col] = theta.get(offset++);
                }
            }
            double[] biases = new double[sizes[index + 1]];
            for (int col = 0; col < biases.length; col++) {
                biases[col] = theta.get(offset++);
            }
            layers[index] = new DenseLayer(Matrix.of(weights), new Vector(biases), new Sigmoid());
        }
        return new Network(layers);
    }

    private static double forwardLoss(Network network, Matrix inputs, Matrix targets,
                                      OutputObjective objective) {
        Matrix activation = inputs;
        Matrix z = null;
        for (DenseLayer layer : network.layers()) {
            z = activation.multiply(layer.weights()).addRowVector(layer.biases());
            activation = z.apply(layer.activation()::apply);
        }
        return objective.loss(z, activation, targets);
    }

    private static void close(double expected, double actual) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > TOLERANCE) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void vectorClose(Vector expected, Vector actual) {
        if (expected.size() != actual.size()) {
            throw new AssertionError("Vector size mismatch");
        }
        for (int index = 0; index < expected.size(); index++) {
            close(expected.get(index), actual.get(index));
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
            // Validation happens before any parameter changes.
        }
    }
}