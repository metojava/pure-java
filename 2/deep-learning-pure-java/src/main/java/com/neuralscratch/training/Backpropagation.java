package com.neuralscratch.training;

import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class Backpropagation {
    private Backpropagation() {
    }

    public static double trainBatch(Network network, Matrix inputs, Matrix targets,
                                    OutputObjective objective, double learningRate) {
        Objects.requireNonNull(network, "network");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(objective, "objective");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException(
                "Learning rate must be positive and finite, got " + learningRate
            );
        }
        List<DenseLayer> layers = network.layers();
        int layerCount = layers.size();
        DenseLayer last = layers.get(layerCount - 1);
        if (inputs.cols() != network.inputSize()) {
            throw new IllegalArgumentException(
                "Inputs need " + network.inputSize() + " columns, got " + inputs.cols()
            );
        }
        if (targets.rows() != inputs.rows() || targets.cols() != last.outputSize()) {
            throw new IllegalArgumentException(
                "Targets must be " + inputs.rows() + "x" + last.outputSize()
                    + ", got " + targets.rows() + "x" + targets.cols()
            );
        }

        // Forward pass: cache Z_l and A_l; activations.get(0) is the input batch A_0 = X.
        List<Matrix> preActivations = new ArrayList<>(layerCount);
        List<Matrix> activations = new ArrayList<>(layerCount + 1);
        activations.add(inputs);
        Matrix current = inputs;
        for (DenseLayer layer : layers) {
            // Z_l = A_(l-1) W_l + b_l ; A_l = f(Z_l)
            Matrix z = current.multiply(layer.weights()).addRowVector(layer.biases());
            current = z.apply(layer.activation()::apply);
            preActivations.add(z);
            activations.add(current);
        }
        Matrix lastPreActivation = preActivations.get(layerCount - 1);
        Matrix lastActivation = activations.get(layerCount);
        double loss = objective.loss(lastPreActivation, lastActivation, targets);

        // Backward pass: compute every gradient while all weights are still unchanged.
        Matrix[] weightGradients = new Matrix[layerCount];
        Vector[] biasGradients = new Vector[layerCount];
        // delta_L = dL/dZ_L, supplied by the objective.
        Matrix delta = objective.delta(lastPreActivation, lastActivation, targets,
            last.activation());
        for (int index = layerCount - 1; index >= 0; index--) {
            // dL/dW_l = A_(l-1)^T delta_l : (in x batch)(batch x out) = in x out
            weightGradients[index] = activations.get(index).transpose().multiply(delta);
            // dL/db_l = column sums of delta_l
            biasGradients[index] = rowSum(delta);
            if (index > 0) {
                DenseLayer hidden = layers.get(index - 1);
                Matrix nextWeights = layers.get(index).weights();
                // delta_(l-1) = (delta_l W_l^T) (*) f'(Z_(l-1))
                delta = delta.multiply(nextWeights.transpose())
                    .hadamard(preActivations.get(index - 1)
                        .apply(hidden.activation()::derivative));
            }
        }

        // Update phase: gradient descent on every layer.
        for (int index = 0; index < layerCount; index++) {
            layers.get(index).applyGradients(
                weightGradients[index], biasGradients[index], learningRate);
        }
        return loss;
    }

    public static Vector rowSum(Matrix matrix) {
        Objects.requireNonNull(matrix, "matrix");
        double[] sums = new double[matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                // Add every row together: one total per column.
                sums[col] += matrix.get(row, col);
            }
        }
        return new Vector(sums);
    }
}