package com.neuralscratch.evaluation;

import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class Metrics {
    private Metrics() {
    }

    public static int[] argmax(Matrix scores) {
        Objects.requireNonNull(scores, "scores");
        int[] winners = new int[scores.rows()];
        for (int row = 0; row < scores.rows(); row++) {
            int best = 0;
            for (int col = 0; col < scores.cols(); col++) {
                double value = scores.get(row, col);
                if (Double.isNaN(value)) {
                    throw new IllegalArgumentException("NaN score at row " + row);
                }
                // Strict comparison keeps the first class when scores tie.
                if (value > scores.get(row, best)) {
                    best = col;
                }
            }
            winners[row] = best;
        }
        return winners;
    }

    public static double accuracy(int[] actual, int[] predicted) {
        requireSameLength(actual, predicted);
        int correct = 0;
        for (int index = 0; index < actual.length; index++) {
            if (actual[index] == predicted[index]) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }

    public static int[][] confusionMatrix(int[] actual, int[] predicted, int classes) {
        requireSameLength(actual, predicted);
        if (classes <= 0) {
            throw new IllegalArgumentException("classes must be positive");
        }
        int[][] counts = new int[classes][classes];
        for (int index = 0; index < actual.length; index++) {
            if (actual[index] < 0 || actual[index] >= classes
                    || predicted[index] < 0 || predicted[index] >= classes) {
                throw new IllegalArgumentException("Class index out of range at sample " + index);
            }
            // Row = true class, column = predicted class.
            counts[actual[index]][predicted[index]]++;
        }
        return counts;
    }

    public static String formatConfusion(int[][] counts) {
        StringBuilder text = new StringBuilder("actual\\predicted");
        for (int col = 0; col < counts.length; col++) {
            text.append(String.format("%6d", col));
        }
        text.append(System.lineSeparator());
        for (int row = 0; row < counts.length; row++) {
            text.append(String.format("%16d", row));
            for (int col = 0; col < counts[row].length; col++) {
                text.append(String.format("%6d", counts[row][col]));
            }
            text.append(System.lineSeparator());
        }
        return text.toString();
    }

    private static void requireSameLength(int[] actual, int[] predicted) {
        Objects.requireNonNull(actual, "actual");
        Objects.requireNonNull(predicted, "predicted");
        if (actual.length == 0 || actual.length != predicted.length) {
            throw new IllegalArgumentException(
                "Need equal, non-empty label arrays but got " + actual.length
                    + " and " + predicted.length);
        }
    }
}