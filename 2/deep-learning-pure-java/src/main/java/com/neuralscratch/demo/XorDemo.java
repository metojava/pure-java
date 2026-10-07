package com.neuralscratch.demo;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import com.neuralscratch.training.Dataset;
import com.neuralscratch.training.Trainer;
import java.util.Random;

public final class XorDemo {
    private static final int EPOCHS = 8000;
    private static final int BATCH_SIZE = 4;
    private static final double LEARNING_RATE = 1.0;
    private static final int MAX_ATTEMPTS = 10;

    private XorDemo() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {1}, {1}, {0}});
        Dataset dataset = new Dataset(inputs, targets);

        System.out.println("Baseline: one sigmoid neuron, no hidden layer");
        Random baselineRandom = new Random(1);
        Network baseline = new Network(DenseLayer.random(2, 1, new Sigmoid(), baselineRandom));
        Trainer.train(baseline, dataset, new MeanSquaredError(), EPOCHS, BATCH_SIZE,
            LEARNING_RATE, baselineRandom);
        int baselineCorrect = report(baseline, inputs, targets);
        System.out.println("A single neuron cannot get all four rows right; it got "
            + baselineCorrect + " / 4.");
        System.out.println();

        System.out.println("Network: 2 -> 4 (sigmoid) -> 1 (sigmoid)");
        for (int attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            long seed = attempt;
            Random random = new Random(seed);
            Network network = new Network(
                DenseLayer.random(2, 4, new Sigmoid(), random),
                DenseLayer.random(4, 1, new Sigmoid(), random));
            double[] losses = Trainer.train(network, dataset, new MeanSquaredError(), EPOCHS,
                BATCH_SIZE, LEARNING_RATE, random);
            System.out.printf("seed %d: first epoch loss %.6f, final epoch loss %.6f%n",
                seed, losses[0], losses[EPOCHS - 1]);
            if (report(network, inputs, targets) == 4) {
                System.out.println("All four XOR rows are correct with seed " + seed + ".");
                return;
            }
            System.out.println("Not all rows correct; trying the next seed.");
        }
        System.out.println("No seed solved XOR. Increase EPOCHS, hidden units, or change the rate.");
    }

    private static int report(Network network, Matrix inputs, Matrix targets) {
        Matrix predictions = network.predict(inputs);
        int correct = 0;
        System.out.println("x1 x2  prediction  decision  target");
        for (int row = 0; row < inputs.rows(); row++) {
            double probability = predictions.get(row, 0);
            int decision = probability >= 0.5 ? 1 : 0;
            int expected = (int) targets.get(row, 0);
            correct += decision == expected ? 1 : 0;
            System.out.printf(" %d  %d     %.4f        %d        %d%n",
                (int) inputs.get(row, 0), (int) inputs.get(row, 1),
                probability, decision, expected);
        }
        return correct;
    }
}