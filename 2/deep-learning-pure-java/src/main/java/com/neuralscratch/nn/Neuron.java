package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Vector;
import java.util.Objects;

public final class Neuron {
    private final Vector weights;
    private final double bias;
    private final Activation activation;

    public Neuron(Vector weights, double bias, Activation activation) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.activation = Objects.requireNonNull(activation, "activation");
        if (!Double.isFinite(bias)) {
            throw new IllegalArgumentException("Bias must be a finite number, got " + bias);
        }
        this.bias = bias;
    }

    public Vector weights() {
        return weights;
    }

    public double bias() {
        return bias;
    }

    public Activation activation() {
        return activation;
    }

    public int inputSize() {
        return weights.size();
    }

    public double weightedSum(Vector input) {
        Objects.requireNonNull(input, "input");
        if (input.size() != weights.size()) {
            throw new IllegalArgumentException(
                "Neuron expects " + weights.size() + " inputs, got " + input.size()
            );
        }
        // z = w . x + b
        return weights.dot(input) + bias;
    }

    public double predict(Vector input) {
        // a = f(z) = f(w . x + b)
        return activation.apply(weightedSum(input));
    }
}