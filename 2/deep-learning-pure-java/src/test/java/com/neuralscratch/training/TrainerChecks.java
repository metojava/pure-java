package com.neuralscratch.training;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.Arrays;
import java.util.Random;

public final class TrainerChecks {
    private TrainerChecks() {
    }

    public static void main(String[] args) {
        checkBatchCounts();
        checkShuffle();
        checkDatasetValidation();
        checkSelectUsesActualBatchSize();
        checkTrainerValidation();
        checkLossFallsOnAnd();
        System.out.println("All trainer checks passed");
    }

    private static void checkBatchCounts() {
        require(Trainer.batchCount(4, 2) == 2, "4 samples, batch 2");
        require(Trainer.batchCount(4, 3) == 2, "4 samples, batch 3");
        require(Trainer.batchCount(6000, 32) == 188, "6000 samples, batch 32");
        require(Trainer.batchCount(4, 100) == 1, "batch larger than data");
        require(Trainer.batchCount(5, Integer.MAX_VALUE) == 1, "huge batch size");
    }

    private static void checkShuffle() {
        int[] first = Trainer.shuffledIndices(100, new Random(7));
        int[] second = Trainer.shuffledIndices(100, new Random(7));
        require(Arrays.equals(first, second), "same seed must give same order");
        int[] sorted = first.clone();
        Arrays.sort(sorted);
        for (int index = 0; index < sorted.length; index++) {
            require(sorted[index] == index, "shuffle must be a permutation");
        }
        require(!Arrays.equals(first, sorted), "100 items should not stay sorted");
        require(Trainer.shuffledIndices(1, new Random(1))[0] == 0, "single item");
    }

    private static void checkDatasetValidation() {
        Matrix two = Matrix.of(new double[][] {{1}, {2}});
        Matrix three = Matrix.of(new double[][] {{1}, {2}, {3}});
        expectIllegalArgument(() -> new Dataset(two, three));
        require(new Dataset(two, two).size() == 2, "size");
    }

    private static void checkSelectUsesActualBatchSize() {
        Dataset dataset = new Dataset(
            Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}}),
            Matrix.of(new double[][] {{10}, {11}, {12}}));
        Dataset last = dataset.select(new int[] {2, 0, 1}, 2, 3);
        require(last.size() == 1, "last batch should hold the one remaining sample");
        require(last.inputs().get(0, 0) == 0.0 && last.inputs().get(0, 1) == 1.0, "row copy");
        require(last.targets().get(0, 0) == 11.0, "target follows its input");
        expectIllegalArgument(() -> dataset.select(new int[] {0, 1, 5}, 0, 3));
        expectIllegalArgument(() -> dataset.select(new int[] {0, 1, 2}, 2, 2));
    }

    private static void checkTrainerValidation() {
        Dataset dataset = andGate();
        Network network = new Network(DenseLayer.random(2, 1, new Sigmoid(), new Random(1)));
        MeanSquaredError objective = new MeanSquaredError();
        Random random = new Random(1);
        expectIllegalArgument(() -> Trainer.train(network, dataset, objective, 0, 2, 0.5, random));
        expectIllegalArgument(() -> Trainer.train(network, dataset, objective, 1, 0, 0.5, random));
        expectIllegalArgument(() -> Trainer.train(network, dataset, objective, 1, 2, 0.0, random));
        expectIllegalArgument(() -> Trainer.train(network, dataset, objective, 1, 2,
            Double.NaN, random));
        Network wrongInputs = new Network(DenseLayer.random(3, 1, new Sigmoid(), new Random(1)));
        expectIllegalArgument(() -> Trainer.train(wrongInputs, dataset, objective, 1, 2, 0.5,
            random));
    }

    private static void checkLossFallsOnAnd() {
        Dataset dataset = andGate();
        Random random = new Random(42);
        Network network = new Network(DenseLayer.random(2, 1, new Sigmoid(), random));
        double[] losses = Trainer.train(network, dataset, new MeanSquaredError(), 1000, 3, 1.0,
            random);
        require(losses.length == 1000, "one loss per epoch");
        require(losses[999] < losses[0], "loss should be lower after 1000 epochs");
        System.out.printf("AND loss: first epoch %.6f, last epoch %.6f%n", losses[0], losses[999]);
    }

    private static Dataset andGate() {
        return new Dataset(
            Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}}),
            Matrix.of(new double[][] {{0}, {0}, {0}, {1}}));
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError("Check failed: " + message);
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // Invalid training setup is rejected before any weight changes.
        }
    }
}