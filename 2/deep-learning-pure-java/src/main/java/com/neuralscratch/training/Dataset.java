package com.neuralscratch.training;

import com.neuralscratch.math.Matrix;
import java.util.Objects;

public record Dataset(Matrix inputs, Matrix targets) {
    public Dataset {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(targets, "targets");
        if (inputs.rows() <= 0 || inputs.rows() != targets.rows()) {
            throw new IllegalArgumentException(
                "Inputs and targets need the same positive row count, got "
                    + inputs.rows() + " and " + targets.rows()
            );
        }
    }

    public int size() {
        return inputs.rows();
    }

    public Dataset select(int[] order, int from, int to) {
        Objects.requireNonNull(order, "order");
        if (from < 0 || from >= to || to > order.length) {
            throw new IllegalArgumentException(
                "Invalid slice [" + from + ", " + to + ") of " + order.length + " indices"
            );
        }
        int count = to - from;
        double[][] selectedInputs = new double[count][inputs.cols()];
        double[][] selectedTargets = new double[count][targets.cols()];
        for (int position = 0; position < count; position++) {
            int source = order[from + position];
            if (source < 0 || source >= size()) {
                throw new IllegalArgumentException("Sample index out of range: " + source);
            }
            for (int col = 0; col < inputs.cols(); col++) {
                selectedInputs[position][col] = inputs.get(source, col);
            }
            for (int col = 0; col < targets.cols(); col++) {
                selectedTargets[position][col] = targets.get(source, col);
            }
        }
        return new Dataset(Matrix.of(selectedInputs), Matrix.of(selectedTargets));
    }
}