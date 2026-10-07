package com.neuralscratch.demo;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.ReLU;
import com.neuralscratch.activation.Softmax;
import com.neuralscratch.data.Mnist;
import com.neuralscratch.evaluation.Metrics;
import com.neuralscratch.loss.SoftmaxCrossEntropy;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import com.neuralscratch.training.Dataset;
import com.neuralscratch.training.Trainer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

public final class MnistDemo {
    private static final int TRAIN_LIMIT = 6000;
    private static final int TEST_LIMIT = 1000;
    private static final int EPOCHS = 10;
    private static final int BATCH_SIZE = 32;
    private static final double LEARNING_RATE = 0.05;
    private static final long SEED = 7L;

    private MnistDemo() {
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            System.err.println("Usage: java com.neuralscratch.demo.MnistDemo "
                + "<train-images> <train-labels> <test-images> <test-labels>");
            System.exit(2);
        }
        try {
            run(Path.of(args[0]), Path.of(args[1]), Path.of(args[2]), Path.of(args[3]));
        } catch (IOException | IllegalArgumentException | IllegalStateException failure) {
            System.err.println("MNIST demo failed: " + failure.getMessage());
            System.exit(1);
        }
    }

    private static void run(Path trainImages, Path trainLabels, Path testImages, Path testLabels)
            throws IOException {
        Dataset train = Mnist.load(trainImages, trainLabels, TRAIN_LIMIT);
        Dataset test = Mnist.load(testImages, testLabels, TEST_LIMIT);
        System.out.println("Loaded " + train.size() + " training and " + test.size()
            + " test images");

        Random random = new Random(SEED);
        Network network = new Network(
            DenseLayer.random(Mnist.PIXELS, 64, new ReLU(), random),
            DenseLayer.random(64, Mnist.CLASSES, new Identity(), random));
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();

        double[] losses = Trainer.train(network, train, objective, EPOCHS, BATCH_SIZE,
            LEARNING_RATE, random);
        for (int epoch = 0; epoch < losses.length; epoch++) {
            System.out.printf("epoch %2d  training loss %.4f%n", epoch + 1, losses[epoch]);
        }

        Matrix logits = network.predict(test.inputs());
        Matrix probabilities = Softmax.apply(logits);
        int[] predicted = Metrics.argmax(probabilities);
        int[] actual = Metrics.argmax(test.targets());

        System.out.printf("Held-out test loss: %.4f%n",
            objective.loss(logits, logits, test.targets()));
        System.out.printf("Held-out test accuracy: %.2f%% (%d images)%n",
            100.0 * Metrics.accuracy(actual, predicted), actual.length);
        System.out.println(Metrics.formatConfusion(
            Metrics.confusionMatrix(actual, predicted, Mnist.CLASSES)));
    }
}