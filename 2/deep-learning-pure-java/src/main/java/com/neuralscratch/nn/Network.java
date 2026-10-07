package com.neuralscratch.nn;

import com.neuralscratch.math.Matrix;
import java.util.List;
import java.util.Objects;

public final class Network {
    private final List<DenseLayer> layers;

    public Network(DenseLayer... layers) {
        Objects.requireNonNull(layers, "layers");
        if (layers.length == 0) {
            throw new IllegalArgumentException("A network needs at least one layer");
        }
        for (int index = 0; index < layers.length; index++) {
            Objects.requireNonNull(layers[index], "layer " + index);
            if (index > 0 && layers[index - 1].outputSize() != layers[index].inputSize()) {
                throw new IllegalArgumentException(
                    "Layer " + (index - 1) + " outputs " + layers[index - 1].outputSize()
                        + " values but layer " + index + " expects "
                        + layers[index].inputSize()
                );
            }
        }
        // List.of copies the array into an unmodifiable list.
        this.layers = List.of(layers);
    }

    public List<DenseLayer> layers() {
        return layers;
    }

    public int inputSize() {
        return layers.get(0).inputSize();
    }

    public int outputSize() {
        return layers.get(layers.size() - 1).outputSize();
    }

    public Matrix predict(Matrix inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.cols() != inputSize()) {
            throw new IllegalArgumentException(
                "Network expects " + inputSize() + " input columns, got " + inputs.cols()
            );
        }
        Matrix current = inputs;
        for (DenseLayer layer : layers) {
            // Each layer's activations become the next layer's inputs.
            current = layer.forward(current);
        }
        return current;
    }
}