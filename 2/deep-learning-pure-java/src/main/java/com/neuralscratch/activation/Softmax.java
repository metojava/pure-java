package com.neuralscratch.activation;

import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class Softmax {
    private Softmax() {
    }

    public static Matrix apply(Matrix logits) {
        requireFinite(logits);
        double[][] probabilities = new double[logits.rows()][logits.cols()];
        for (int row = 0; row < logits.rows(); row++) {
            double max = rowMax(logits, row);
            double sum = 0.0;
            for (int col = 0; col < logits.cols(); col++) {
                // exp(z - max) is at most 1, so it cannot overflow. The largest entry gives exactly 1.
                double shifted = Math.exp(logits.get(row, col) - max);
                probabilities[row][col] = shifted;
                sum += shifted;
            }
            for (int col = 0; col < logits.cols(); col++) {
                // p_k = exp(z_k - max) / sum_j exp(z_j - max); the common factor exp(-max) cancels.
                probabilities[row][col] /= sum;
            }
        }
        return Matrix.of(probabilities);
    }

    public static double[] logSumExp(Matrix logits) {
        requireFinite(logits);
        double[] result = new double[logits.rows()];
        for (int row = 0; row < logits.rows(); row++) {
            double max = rowMax(logits, row);
            double sum = 0.0;
            for (int col = 0; col < logits.cols(); col++) {
                sum += Math.exp(logits.get(row, col) - max);
            }
            // log(sum_j exp(z_j)) = max + log(sum_j exp(z_j - max)); sum is at least 1.
            result[row] = max + Math.log(sum);
        }
        return result;
    }

    private static double rowMax(Matrix logits, int row) {
        double max = logits.get(row, 0);
        for (int col = 1; col < logits.cols(); col++) {
            max = Math.max(max, logits.get(row, col));
        }
        return max;
    }

    private static void requireFinite(Matrix logits) {
        Objects.requireNonNull(logits, "logits");
        for (int row = 0; row < logits.rows(); row++) {
            for (int col = 0; col < logits.cols(); col++) {
                if (!Double.isFinite(logits.get(row, col))) {
                    throw new IllegalArgumentException(
                        "Logit at row " + row + ", column " + col + " is not finite"
                    );
                }
            }
        }
    }
}