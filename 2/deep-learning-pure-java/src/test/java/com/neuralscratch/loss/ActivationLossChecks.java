package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.ReLU;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Softmax;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;

public final class ActivationLossChecks {
    private static final double TOLERANCE = 1e-9;

    private ActivationLossChecks() {
    }

    public static void main(String[] args) {
        checkRelu();
        checkSoftmax();
        checkCrossEntropyValues();
        checkCrossEntropyGradient();
        checkMeanSquaredErrorGradient();
        checkSaturation();
        checkValidation();
        System.out.println("All activation and loss checks passed");
    }

    private static void checkRelu() {
        ReLU relu = new ReLU();
        close(0.0, relu.apply(-3.5), TOLERANCE);
        close(0.0, relu.apply(0.0), TOLERANCE);
        close(2.5, relu.apply(2.5), TOLERANCE);
        close(0.0, relu.derivative(-3.5), TOLERANCE);
        close(0.0, relu.derivative(0.0), TOLERANCE);
        close(1.0, relu.derivative(2.5), TOLERANCE);
        close(1.0, Calculus.derivative(relu::apply, 2.5), 1e-6);
        close(0.0, Calculus.derivative(relu::apply, -2.5), 1e-6);
    }

    private static void checkSoftmax() {
        Matrix logits = Matrix.of(new double[][] {{1, 2, 3}, {0, 0, 0}});
        Matrix probabilities = Softmax.apply(logits);
        close(0.09003057317038046, probabilities.get(0, 0), 1e-12);
        close(0.24472847105479764, probabilities.get(0, 1), 1e-12);
        close(0.6652409557748219, probabilities.get(0, 2), 1e-12);
        close(1.0 / 3.0, probabilities.get(1, 1), 1e-12);
        for (int row = 0; row < probabilities.rows(); row++) {
            double sum = 0.0;
            for (int col = 0; col < probabilities.cols(); col++) {
                sum += probabilities.get(row, col);
            }
            close(1.0, sum, 1e-12);
        }

        Matrix shifted = Softmax.apply(Matrix.of(new double[][] {{1001, 1002, 1003}}));
        Matrix huge = Softmax.apply(Matrix.of(new double[][] {{1000, 1001, 1002}}));
        for (int col = 0; col < 3; col++) {
            close(probabilities.get(0, col), shifted.get(0, col), 1e-12);
            close(probabilities.get(0, col), huge.get(0, col), 1e-12);
        }
        if (Double.isFinite(Math.exp(1000.0))) {
            throw new AssertionError("The naive exponential should overflow here");
        }

        expectIllegalArgument(() -> Softmax.apply(Matrix.of(new double[][] {{1, Double.NaN}})));
        expectIllegalArgument(() -> Softmax.apply(
            Matrix.of(new double[][] {{Double.POSITIVE_INFINITY, 1}})));
    }

    private static void checkCrossEntropyValues() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix uniform = Matrix.of(new double[][] {{0, 0}});
        Matrix classZero = Matrix.of(new double[][] {{1, 0}});
        close(Math.log(2.0), objective.loss(uniform, uniform, classZero), 1e-12);

        Matrix logits = Matrix.of(new double[][] {{2.0, 1.0, 0.1}});
        double denominator = Math.exp(2.0) + Math.exp(1.0) + Math.exp(0.1);
        double expected = -Math.log(Math.exp(2.0) / denominator);
        Matrix oneHot = Matrix.of(new double[][] {{1, 0, 0}});
        close(expected, objective.loss(logits, logits, oneHot), 1e-12);

        Matrix soft = Matrix.of(new double[][] {{0.5, 0.5, 0.0}});
        double distributionLoss = -0.5 * Math.log(Math.exp(2.0) / denominator)
            - 0.5 * Math.log(Math.exp(1.0) / denominator);
        close(distributionLoss, objective.loss(logits, logits, soft), 1e-12);

        Matrix confidentlyWrong = Matrix.of(new double[][] {{1000, 0}});
        Matrix classOne = Matrix.of(new double[][] {{0, 1}});
        close(1000.0, objective.loss(confidentlyWrong, confidentlyWrong, classOne), 1e-9);
    }

    private static void checkCrossEntropyGradient() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{0.3, -1.2, 2.0}, {1.5, 0.2, -0.7}});
        Matrix targets = Matrix.of(new double[][] {{0, 0, 1}, {0.25, 0.75, 0}});
        Matrix delta = objective.delta(logits, logits, targets, new Identity());

        Vector numeric = Calculus.gradient(
            flat -> {
                Matrix z = unflatten(flat, 2, 3);
                return objective.loss(z, z, targets);
            },
            flatten(logits)
        );
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 3; col++) {
                close(numeric.get(row * 3 + col), delta.get(row, col), 1e-7);
            }
        }
        Matrix probabilities = Softmax.apply(logits);
        close((probabilities.get(0, 0) - 0.0) / 2.0, delta.get(0, 0), TOLERANCE);
    }

    private static void checkMeanSquaredErrorGradient() {
        MeanSquaredError objective = new MeanSquaredError();
        Activation sigmoid = new Sigmoid();
        Matrix logits = Matrix.of(new double[][] {{0.4, -0.9}, {1.1, 0.3}});
        Matrix targets = Matrix.of(new double[][] {{1, 0}, {0, 1}});
        Matrix output = logits.apply(sigmoid::apply);
        Matrix delta = objective.delta(logits, output, targets, sigmoid);

        Vector numeric = Calculus.gradient(
            flat -> {
                Matrix z = unflatten(flat, 2, 2);
                return objective.loss(z, z.apply(sigmoid::apply), targets);
            },
            flatten(logits)
        );
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 2; col++) {
                close(numeric.get(row * 2 + col), delta.get(row, col), 1e-7);
            }
        }
    }

    private static void checkSaturation() {
        Activation sigmoid = new Sigmoid();
        MeanSquaredError mse = new MeanSquaredError();
        Matrix z = Matrix.of(new double[][] {{-10.0}});
        Matrix target = Matrix.of(new double[][] {{1.0}});
        double mseSlope = mse.delta(z, z.apply(sigmoid::apply), target, sigmoid).get(0, 0);

        SoftmaxCrossEntropy crossEntropy = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{10.0, 0.0}});
        Matrix wrongTarget = Matrix.of(new double[][] {{0.0, 1.0}});
        double ceSlope = crossEntropy.delta(logits, logits, wrongTarget, new Identity()).get(0, 0);
        if (Math.abs(mseSlope) > 1e-4 || Math.abs(ceSlope) < 0.9) {
            throw new AssertionError("Expected tiny MSE slope and large cross-entropy slope");
        }
        System.out.println("Confidently wrong, MSE with sigmoid: |dL/dz| = " + Math.abs(mseSlope));
        System.out.println("Confidently wrong, softmax cross-entropy: |dL/dz| = " + Math.abs(ceSlope));
    }

    private static void checkValidation() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{1, 2}});
        Matrix good = Matrix.of(new double[][] {{1, 0}});
        expectIllegalArgument(() -> objective.delta(logits, logits, good, new Sigmoid()));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 0, 0}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 1}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{-0.5, 1.5}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 0}, {0, 1}})));
    }

    private static Vector flatten(Matrix matrix) {
        double[] values = new double[matrix.rows() * matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row * matrix.cols() + col] = matrix.get(row, col);
            }
        }
        return new Vector(values);
    }

    private static Matrix unflatten(Vector vector, int rows, int cols) {
        double[][] values = new double[rows][cols];
        for (int index = 0; index < vector.size(); index++) {
            values[index / cols][index % cols] = vector.get(index);
        }
        return Matrix.of(values);
    }

    private static void close(double expected, double actual, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > tolerance) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // The invalid input was rejected before any loss was computed.
        }
    }
}