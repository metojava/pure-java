package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class MeanSquaredError implements OutputObjective {
    @Override
    public double loss(Matrix logits, Matrix output, Matrix targets) {
        requireSameShapes(logits, output, targets);
        double sumSquares = 0.0;
        for (int row = 0; row < output.rows(); row++) {
            for (int col = 0; col < output.cols(); col++) {
                double error = output.get(row, col) - targets.get(row, col);
                sumSquares += error * error;
            }
        }
        // L = 0.5 * sum (a - y)^2 / N, with N = batch rows.
        return 0.5 * sumSquares / output.rows();
    }

    @Override
    public Matrix delta(Matrix logits, Matrix output, Matrix targets,
                        Activation outputActivation) {
        requireSameShapes(logits, output, targets);
        Objects.requireNonNull(outputActivation, "outputActivation");
        // dL/da = (a - y) / N; chain rule: dL/dz = dL/da (*) f'(z).
        return output.subtract(targets)
            .scale(1.0 / output.rows())
            .hadamard(logits.apply(outputActivation::derivative));
    }

    private static void requireSameShapes(Matrix logits, Matrix output, Matrix targets) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(targets, "targets");
        if (output.rows() != targets.rows() || output.cols() != targets.cols()
                || logits.rows() != output.rows() || logits.cols() != output.cols()) {
            throw new IllegalArgumentException(
                "logits, output and targets must share a shape; got "
                    + logits.rows() + "x" + logits.cols() + ", "
                    + output.rows() + "x" + output.cols() + ", "
                    + targets.rows() + "x" + targets.cols()
            );
        }
    }
}