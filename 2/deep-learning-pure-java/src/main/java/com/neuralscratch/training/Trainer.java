package com.neuralscratch.training;

import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public final class Trainer {
    private Trainer() {
    }

    public static double[] train(
        Network network,
        Dataset dataset,
        OutputObjective objective,
        int epochs,
        int batchSize,
        double learningRate,
        Random random
    ) {
        Objects.requireNonNull(network, "network");
        Objects.requireNonNull(dataset, "dataset");
        Objects.requireNonNull(objective, "objective");
        Objects.requireNonNull(random, "random");
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive");
        }
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException("learningRate must be positive and finite");
        }
        List<DenseLayer> layers = network.layers();
        if (layers.get(0).inputSize() != dataset.inputs().cols()) {
            throw new IllegalArgumentException("Network expects " + layers.get(0).inputSize()
                + " inputs but dataset has " + dataset.inputs().cols());
        }
        int outputSize = layers.get(layers.size() - 1).outputSize();
        if (outputSize != dataset.targets().cols()) {
            throw new IllegalArgumentException("Network produces " + outputSize
                + " outputs but dataset has " + dataset.targets().cols() + " target columns");
        }

        int samples = dataset.size();
        double[] epochLosses = new double[epochs];
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] order = shuffledIndices(samples, random);
            double weightedLossSum = 0.0;
            for (int start = 0; start < samples; ) {
                int count = Math.min(batchSize, samples - start);
                Dataset batch = dataset.select(order, start, start + count);
                double loss = Backpropagation.trainBatch(
                    network, batch.inputs(), batch.targets(), objective, learningRate);
                if (!Double.isFinite(loss)) {
                    throw new IllegalStateException(
                        "Loss became non-finite in epoch " + (epoch + 1)
                            + "; try a smaller learning rate");
                }
                // Each batch loss is a mean, so multiplying by its size restores the batch total.
                weightedLossSum += loss * count;
                start += count;
            }
            epochLosses[epoch] = weightedLossSum / samples;
        }
        return epochLosses;
    }

    public static int[] shuffledIndices(int size, Random random) {
        Objects.requireNonNull(random, "random");
        if (size <= 0) {
            throw new IllegalArgumentException("size must be positive");
        }
        int[] order = new int[size];
        for (int index = 0; index < size; index++) {
            order[index] = index;
        }
        for (int last = size - 1; last > 0; last--) {
            // Choose uniformly from the not-yet-fixed prefix 0..last, then fix that slot.
            int chosen = random.nextInt(last + 1);
            int swap = order[chosen];
            order[chosen] = order[last];
            order[last] = swap;
        }
        return order;
    }

    public static int batchCount(int samples, int batchSize) {
        if (samples <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException("samples and batchSize must be positive");
        }
        // Ceiling division without the overflow risk of samples + batchSize - 1.
        return samples / batchSize + (samples % batchSize == 0 ? 0 : 1);
    }
}