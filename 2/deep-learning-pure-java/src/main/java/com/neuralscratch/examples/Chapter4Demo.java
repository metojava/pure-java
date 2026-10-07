package com.neuralscratch.examples;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.Locale;
import java.util.Random;

public final class Chapter4Demo {
    private Chapter4Demo() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});

        Random random = new Random(42);
        DenseLayer hidden = DenseLayer.random(2, 3, new Sigmoid(), random);
        DenseLayer output = DenseLayer.random(3, 1, new Sigmoid(), random);
        Network network = new Network(hidden, output);

        System.out.println("Random 2x3x1 network, seed 42");
        print("hidden weights W1", hidden.weights());
        print("output weights W2", output.weights());
        Matrix hiddenActivations = hidden.forward(inputs);
        print("hidden activations A1", hiddenActivations);
        print("predictions A2", network.predict(inputs));

        DenseLayer orNand = new DenseLayer(
            Matrix.of(new double[][] {{1, -1}, {1, -1}}),
            new Vector(new double[] {-0.5, 1.5}),
            new Step());
        DenseLayer and = new DenseLayer(
            Matrix.of(new double[][] {{1}, {1}}),
            new Vector(new double[] {-1.5}),
            new Step());
        Network xor = new Network(orNand, and);
        print("hand-built XOR predictions", xor.predict(inputs));
    }

    private static void print(String label, Matrix matrix) {
        System.out.println(label + " (" + matrix.rows() + " x " + matrix.cols() + "):");
        for (int row = 0; row < matrix.rows(); row++) {
            StringBuilder line = new StringBuilder("  [");
            for (int col = 0; col < matrix.cols(); col++) {
                if (col > 0) {
                    line.append(", ");
                }
                line.append(String.format(Locale.ROOT, "%8.4f", matrix.get(row, col)));
            }
            System.out.println(line.append(']'));
        }
    }
}