package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import java.util.Objects;
import java.util.Random;

public final class DenseLayer {
    private Matrix weights;
    private Vector biases;
    private final Activation activation;

    public DenseLayer(Matrix weights, Vector biases, Activation activation) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(biases, "biases");
        Objects.requireNonNull(activation, "activation");
        if (biases.size() != weights.cols()) {
            throw new IllegalArgumentException(
                "Bias size " + biases.size() + " must equal output size " + weights.cols()
            );
        }
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;
    }

    public static DenseLayer random(int inputSize, int outputSize,
                                    Activation activation, Random random) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException("Layer sizes must be positive");
        }
        Objects.requireNonNull(activation, "activation");
        Objects.requireNonNull(random, "random");
        // Xavier/Glorot: Var(w) = 2 / (fanIn + fanOut) keeps signal spread stable.
        double stddev = Math.sqrt(2.0 / (inputSize + outputSize));
        double[][] values = new double[inputSize][outputSize];
        for (int row = 0; row < inputSize; row++) {
            for (int col = 0; col < outputSize; col++) {
                // nextGaussian has mean 0 and stddev 1; scaling sets stddev.
                values[row][col] = random.nextGaussian() * stddev;
            }
        }
        return new DenseLayer(Matrix.of(values), Vector.zeros(outputSize), activation);
    }

    public Matrix weights() {
        return weights;
    }

    public Vector biases() {
        return biases;
    }

    public Activation activation() {
        return activation;
    }

    public int inputSize() {
        return weights.rows();
    }

    public int outputSize() {
        return weights.cols();
    }

    public Matrix forward(Matrix inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.cols() != inputSize()) {
            throw new IllegalArgumentException(
                "Layer expects " + inputSize() + " input columns, got " + inputs.cols()
            );
        }
        // A = f(X W + b): (batch x in)(in x out) -> (batch x out), bias added per row.
        return inputs.multiply(weights).addRowVector(biases).apply(activation::apply);
    }

    public void applyGradients(Matrix weightGradient, Vector biasGradient,
                               double learningRate) {
        Objects.requireNonNull(weightGradient, "weightGradient");
        Objects.requireNonNull(biasGradient, "biasGradient");
        if (weightGradient.rows() != weights.rows() || weightGradient.cols() != weights.cols()) {
            throw new IllegalArgumentException(
                "Weight gradient must be " + weights.rows() + "x" + weights.cols()
                    + ", got " + weightGradient.rows() + "x" + weightGradient.cols()
            );
        }
        if (biasGradient.size() != biases.size()) {
            throw new IllegalArgumentException(
                "Bias gradient must have size " + biases.size() + ", got " + biasGradient.size()
            );
        }
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException(
                "Learning rate must be positive and finite, got " + learningRate
            );
        }
        // Gradient descent: step against the slope, W := W - rate * dL/dW.
        weights = weights.subtract(weightGradient.scale(learningRate));
        biases = biases.subtract(biasGradient.scale(learningRate));
    }
}