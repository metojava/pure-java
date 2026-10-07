package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Vector;

public final class NeuronChecks {
    private static final double TOLERANCE = 1e-9;

    private NeuronChecks() {
    }

    public static void main(String[] args) {
        Activation sigmoid = new Sigmoid();
        close(0.5, sigmoid.apply(0));
        close(1.0 / (1.0 + Math.exp(-2.0)), sigmoid.apply(2));
        close(1.0 / (1.0 + Math.exp(3.0)), sigmoid.apply(-3));
        close(0.25, sigmoid.derivative(0));
        // Symmetry: sigma(-z) = 1 - sigma(z).
        close(1.0 - sigmoid.apply(1.7), sigmoid.apply(-1.7));

        for (double extreme : new double[] {-1000, -745, 709, 1000}) {
            double value = sigmoid.apply(extreme);
            double slope = sigmoid.derivative(extreme);
            if (!Double.isFinite(value) || value < 0 || value > 1 || !Double.isFinite(slope)) {
                throw new AssertionError("Sigmoid unstable at z = " + extreme);
            }
        }
        close(1.0, sigmoid.apply(1000));
        close(0.0, sigmoid.apply(-1000));

        for (double z = -6; z <= 6; z += 0.5) {
            close(Calculus.derivative(sigmoid::apply, z), sigmoid.derivative(z), 1e-8);
            close(Calculus.derivative(new Identity()::apply, z), new Identity().derivative(z), 1e-8);
        }
        close(-2.5, new Identity().apply(-2.5));

        Step step = new Step();
        close(1, step.apply(0));
        close(1, step.apply(3));
        close(0, step.apply(-0.001));
        try {
            step.derivative(0);
            throw new AssertionError("Step.derivative should throw");
        } catch (UnsupportedOperationException expected) {
            // Step cannot be trained with gradients.
        }

        Neuron or = new Neuron(new Vector(new double[] {1, 1}), -0.5, step);
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] targets = {0, 1, 1, 1};
        for (int row = 0; row < inputs.length; row++) {
            close(targets[row], or.predict(new Vector(inputs[row])));
        }

        Neuron weighted = new Neuron(new Vector(new double[] {0.8, 1.5}), -1.0, new Identity());
        close(1.3, weighted.weightedSum(new Vector(new double[] {1, 1})));
        close(1.3, weighted.predict(new Vector(new double[] {1, 1})));
        if (weighted.inputSize() != 2) {
            throw new AssertionError("inputSize should be 2");
        }

        expectIllegalArgument(() -> weighted.predict(new Vector(new double[] {1, 2, 3})));
        expectIllegalArgument(() -> new Neuron(new Vector(new double[] {1}), Double.NaN, step));
        try {
            new Neuron(null, 0, step);
            throw new AssertionError("Expected NullPointerException");
        } catch (NullPointerException expected) {
            // Weights are required.
        }
        System.out.println("All neuron checks passed");
    }

    private static void close(double expected, double actual) {
        close(expected, actual, TOLERANCE);
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
            // Bad shapes and values are rejected before computing.
        }
    }
}