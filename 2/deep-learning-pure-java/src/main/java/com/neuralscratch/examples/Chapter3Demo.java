package com.neuralscratch.examples;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.Neuron;
import java.util.Arrays;

public final class Chapter3Demo {
    private static final double[][] INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] OR_TARGETS = {0, 1, 1, 1};
    private static final double[] XOR_TARGETS = {0, 1, 1, 0};

    private Chapter3Demo() {
    }

    public static void main(String[] args) {
        Neuron orNeuron = new Neuron(new Vector(new double[] {1, 1}), -0.5, new Step());
        System.out.println("OR with a step neuron: weights [1, 1], bias -0.5");
        for (int row = 0; row < INPUTS.length; row++) {
            Vector input = new Vector(INPUTS[row]);
            System.out.printf("  %s  z = %5.2f  prediction = %.0f  target = %.0f%n",
                Arrays.toString(INPUTS[row]), orNeuron.weightedSum(input),
                orNeuron.predict(input), OR_TARGETS[row]);
        }

        Neuron smoothOr = new Neuron(new Vector(new double[] {4, 4}), -2, new Sigmoid());
        System.out.println("OR with a sigmoid neuron: weights [4, 4], bias -2");
        for (double[] values : INPUTS) {
            System.out.printf("  %s  output = %.4f%n",
                Arrays.toString(values), smoothOr.predict(new Vector(values)));
        }

        Neuron linear = new Neuron(new Vector(new double[] {2, -1}), 0.5, new Identity());
        // z = 2*3 + (-1)*4 + 0.5 = 2.5
        System.out.printf("Identity neuron on [3, 4]: %.2f%n",
            linear.predict(new Vector(new double[] {3, 4})));

        int bestCorrect = 0;
        int tried = 0;
        for (int i = -12; i <= 12; i++) {
            for (int j = -12; j <= 12; j++) {
                for (int k = -12; k <= 12; k++) {
                    Neuron candidate = new Neuron(
                        new Vector(new double[] {i * 0.25, j * 0.25}), k * 0.25, new Step());
                    bestCorrect = Math.max(bestCorrect, countCorrect(candidate, XOR_TARGETS));
                    tried++;
                }
            }
        }
        System.out.println("Tried " + tried + " step neurons on XOR; best score: "
            + bestCorrect + " of 4");
    }

    private static int countCorrect(Neuron neuron, double[] targets) {
        int correct = 0;
        for (int row = 0; row < INPUTS.length; row++) {
            if (neuron.predict(new Vector(INPUTS[row])) == targets[row]) {
                correct++;
            }
        }
        return correct;
    }
}