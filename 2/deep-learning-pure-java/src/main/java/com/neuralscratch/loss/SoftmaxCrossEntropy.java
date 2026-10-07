package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Softmax;
import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class SoftmaxCrossEntropy implements OutputObjective {
    private static final double ROW_SUM_TOLERANCE = 1e-6;

    @Override
    public double loss(Matrix logits, Matrix output, Matrix targets) {
        // The output argument is intentionally unused: the loss is computed from the raw logits.
        requireCompatible(logits, targets);
        double[] logSumExp = Softmax.logSumExp(logits);
        double total = 0.0;
        for (int row = 0; row < logits.rows(); row++) {
            for (int col = 0; col < logits.cols(); col++) {
                // -log p_k = logSumExp(z) - z_k, so L = sum_k t_k * (logSumExp(z) - z_k).
                total += targets.get(row, col) * (logSumExp[row] - logits.get(row, col));
            }
        }
        return total / logits.rows();
    }

    @Override
    public Matrix delta(Matrix logits, Matrix output, Matrix targets, Activation outputActivation) {
        if (!(outputActivation instanceof Identity)) {
            throw new IllegalArgumentException(
                "SoftmaxCrossEntropy needs an Identity output layer so that z are logits"
            );
        }
        requireCompatible(logits, targets);
        // dL/dz = (softmax(z) - t) / batchRows. The Identity derivative is 1, so this is already delta.
        return Softmax.apply(logits).subtract(targets).scale(1.0 / logits.rows());
    }

    private static void requireCompatible(Matrix logits, Matrix targets) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(targets, "targets");
        if (logits.rows() != targets.rows() || logits.cols() != targets.cols()) {
            throw new IllegalArgumentException(
                "Logits are " + logits.rows() + "x" + logits.cols() + " but targets are "
                    + targets.rows() + "x" + targets.cols()
            );
        }
        for (int row = 0; row < targets.rows(); row++) {
            double sum = 0.0;
            for (int col = 0; col < targets.cols(); col++) {
                double target = targets.get(row, col);
                if (!Double.isFinite(target) || target < 0.0 || target > 1.0) {
                    throw new IllegalArgumentException(
                        "Target at row " + row + ", column " + col + " must be in [0, 1]"
                    );
                }
                sum += target;
            }
            if (Math.abs(sum - 1.0) > ROW_SUM_TOLERANCE) {
                throw new IllegalArgumentException(
                    "Target row " + row + " must sum to 1 but sums to " + sum
                );
            }
        }
    }
}