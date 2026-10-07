package com.neuralscratch.demo;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import com.neuralscratch.training.Dataset;
import com.neuralscratch.training.Trainer;
import java.util.Random;

public final class AndGateDemo {
    private AndGateDemo() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {0}, {0}, {1}});
        Dataset dataset = new Dataset(inputs, targets);

        Random random = new Random(42);
        Network network = new Network(DenseLayer.random(2, 1, new Sigmoid(), random));

        int epochs = 2000;
        int batchSize = 2;
        double[] losses = Trainer.train(
            network, dataset, new MeanSquaredError(), epochs, batchSize, 1.0, random);

        System.out.println("Batches per epoch: " + Trainer.batchCount(dataset.size(), batchSize));
        for (int epoch = 0; epoch < epochs; epoch++) {
            boolean report = epoch == 0 || (epoch + 1) % 400 == 0;
            if (report) {
                System.out.printf("epoch %4d  mean loss %.6f%n", epoch + 1, losses[epoch]);
            }
        }

        Matrix predictions = network.predict(inputs);
        int correct = 0;
        for (int row = 0; row < inputs.rows(); row++) {
            double probability = predictions.get(row, 0);
            int decision = probability >= 0.5 ? 1 : 0;
            int expected = (int) targets.get(row, 0);
            correct += decision == expected ? 1 : 0;
            System.out.printf("%d AND %d -> %.4f -> %d (expected %d)%n",
                (int) inputs.get(row, 0), (int) inputs.get(row, 1),
                probability, decision, expected);
        }
        System.out.println("Correct after thresholding at 0.5: " + correct + " / 4");
    }
}