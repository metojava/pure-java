package com.neuralscratch.examples;

import java.util.Locale;

public final class Chapter1Demo {
    private Chapter1Demo() {
    }

    public static void main(String[] args) {
        double[] inputs = {1.0, 1.0};
        double[] weights = {0.8, 1.5};
        double bias = -1.0;
        double z = bias;
        for (int index = 0; index < inputs.length; index++) {
            z += weights[index] * inputs[index];
        }
        double output = 1.0 / (1.0 + Math.exp(-z));
        System.out.printf(Locale.ROOT, "z = %.1f, sigmoid(z) = %.4f%n", z, output);
    }
}