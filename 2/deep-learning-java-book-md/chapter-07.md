
                "
"
                    "
"Chapter 7"
"

# Training the Network

Turn one backpropagation step into a repeatable, shuffled, monitored training loop.

                "
"

                "
"

## What We Are Building

`Backpropagation.trainBatch` performs exactly one weight update from one batch. A real model needs thousands of them, over data visited in a fresh random order, with a running record of whether the loss is improving. This chapter builds that outer loop from three small parts:

                        FileRole

                            `training/Dataset.java`A record pairing an input matrix with a target matrix, validated to describe the same samples.
                            `training/Trainer.java`Shuffles, slices batches, calls `trainBatch`, and returns one average loss per epoch.
                            `demo/AndGateDemo.java`A small runnable demonstration on the logical AND gate.
                            `training/TrainerChecks.java`A dependency-free check of batching, shuffling, validation and a falling loss.

XOR is deliberately absent. It needs a hidden layer, and Chapter 8 is dedicated to it. Here we use AND, which a single neuron can learn.

                "
"

                "
"

## 1. Epochs, Batches, and Iterations

Three words describe a training run, and they are easy to confuse:

                    "
"

- A **sample** is one row of the dataset: one input vector and its target.

- A **batch** is a small group of samples processed in one call to `trainBatch`, which averages their gradients and makes *one* weight update.

- An **epoch** is one full pass in which every sample appears exactly once.
                    "
"

With `N` samples and batch size `S`, an epoch contains `⌈N / S⌉` batches, so that many weight updates (also called iterations). The final batch is smaller when `S` does not divide `N`:

                        Samples NBatch size SBatch sizes in one epochUpdates per epoch

                            422, 22
                            433, 12
                            4441 (full-batch)
                            600032187 batches of 32, then 16188

Batch size trades noise against cost. A batch of 1 makes a noisy update from a single example; the whole dataset makes the smoothest update but only one per epoch. Medium batches such as 16 to 128 are a common compromise. Because the objectives from Chapters 5 and 6 divide by the number of rows in the batch, the gradient has the same scale whatever batch size you choose.

                "
"

                "
"

## 2. The Dataset Record

A Java `record` is a compact immutable value type. Our `Dataset` holds `inputs` (one row per sample) and `targets` (one row per sample), and its compact constructor rejects mismatched row counts at the moment of creation. A bad dataset therefore cannot travel into the training loop. `Matrix` itself already forbids zero rows, and the explicit check keeps the rule visible in this class.

The method `select` copies a slice of an index array into fresh matrices through `Matrix.of`. It is how the trainer builds a batch, and it also lets you split data into training and holdout parts later. The slice `[from, to)` may be shorter than the batch size, which is exactly what the final partial batch needs.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/training/Dataset.java"
"Copy"
"

```java
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
```

                    "
"
                "
"

                "
"

## 3. Shuffling with Fisher-Yates

If every epoch visits samples in the same order, the model can pick up spurious patterns from that order (for example, all examples of one class arriving last). We reshuffle an array of *indices* at the start of every epoch and leave the data matrices untouched.

The **Fisher-Yates shuffle** walks from the last position down to position 1. At position `last` it picks a random slot among `0…last` (inclusive), swaps it into `last`, and never touches `last` again. Every permutation is equally likely, assuming a fair random source, because there are `n × (n−1) × … × 1 = n!` equally likely sequences of choices and each yields a different permutation. A common mistake, swapping every position with a random slot from the *whole* array, produces `nⁿ` outcomes that cannot be divided evenly among the `n!` permutations, so some orders become more likely than others.

We pass in a `java.util.Random`. Constructing it with a seed such as `new Random(42)` makes the sequence of shuffles reproducible for debugging; a different seed changes the order and the random initial weights. Seeds are for repeatability, not for security.

                "
"

                "
"

## 4. The Trainer

Read `train` from top to bottom. It validates every argument and checks that the network's first layer accepts the dataset's columns and its last layer produces as many outputs as there are target columns. Then, for each epoch, it shuffles the indices, steps through them in slices, builds a batch with `Dataset.select`, and calls `Backpropagation.trainBatch`. That method returns the batch loss *measured before its weight update*.

### Why the epoch loss is weighted by batch size

Each loss returned by `trainBatch` is already an average over the rows in that batch. If you averaged those averages directly, a tiny final batch would count as much as a full one. Suppose batches of 3 and 1 samples report losses 0.4 and 0.8. The true per-sample mean is `(3 × 0.4 + 1 × 0.8) / 4 = 0.5`, but a naive mean of the two numbers gives 0.6. We multiply each batch loss by its actual size, sum, and divide by the total number of samples.

                    "
"
                        "
"What the reported number is, and is not"
"

Each batch is scored *before* its own update, but earlier batches in the same epoch have already changed the weights. The epoch value is therefore a running training signal. It is not a final evaluation of the weights you end with, and it says nothing about unseen data.

                    "
"

Trainer deliberately prints nothing, so tests and applications can decide how to report. It throws `IllegalStateException` if a loss becomes `NaN` or infinite, because continuing would only fill the weights with garbage; the message suggests lowering the learning rate. The helper methods `shuffledIndices` and `batchCount` are public so that the checks can test them directly.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/training/Trainer.java"
"Copy"
"

```java
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
```

                    "
"
                "
"

                "
"

## 5. Demo: Learning the AND Gate

The AND gate outputs 1 only when both inputs are 1:

                        x₁x₂target

                            000
                            010
                            100
                            111

A single sigmoid neuron computes `σ(w₁x₁ + w₂x₂ + b)` and answers "1" when `w₁x₁ + w₂x₂ + b > 0`. That condition draws a straight line through the input square. For AND, one line can separate `(1, 1)` from the other three corners, for example when both weights are positive and the bias is negative enough. The data are **linearly separable**, so this one-neuron network *can* represent the answer. Whether gradient descent finds weights in a given run depends on the initial weights, the learning rate and the number of epochs, so the demo prints what it actually measures.

The configuration below uses 4 samples with batch size 2, so each epoch makes 2 updates. It trains for 2000 epochs with learning rate 1.0, uses `MeanSquaredError`, and seeds `Random` with 42. It prints the epoch loss every 400 epochs and then the final prediction for each row, thresholded at 0.5.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/demo/AndGateDemo.java"
"Copy"
"

```java
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
```

                    "
"
                    "
"
                        "
""
"Run it (after compiling as in section 7)"
"Copy"
"

```
java -cp out com.neuralscratch.demo.AndGateDemo
```

                    "
"

What to look for: the first line should say `Batches per epoch: 2`; the loss column should trend downward; and for AND, the predictions for the first three rows should end near 0 and the last row near 1. The exact decimals depend on how your Chapter 4 `DenseLayer.random` draws its initial weights, so this page does not print numbers that would only be true for one implementation. The program reports the count of correct rows, so you do not have to trust a claim: either the final line says `4 / 4` or it does not, and then you experiment with epochs or the learning rate.

                "
"

                "
"

## 6. Shuffling, Holdout Data and Monitoring

### Do not judge a model on its training rows

The AND gate has only four possible inputs, and we train on all of them, so "4 / 4" means the model memorized the whole function. That is fine for a demo. For a real problem, keep some samples *out* of training and measure on them afterwards; otherwise a model that memorizes looks perfect. Split first, then train only on the training part. Shuffle indices before splitting if the file is sorted (for instance by class), or the holdout part may contain only one class.

                    "
"
                        "
""
"A holdout split with the pieces you already have"
"Copy"
"

```
int[] order = Trainer.shuffledIndices(dataset.size(), new Random(123));
int trainRows = dataset.size() * 8 / 10;
Dataset train = dataset.select(order, 0, trainRows);
Dataset holdout = dataset.select(order, trainRows, order.length);
// Train on `train`. Evaluate on `holdout` once, after the hyperparameters are chosen.
```

                    "
"

Repeatedly tuning epochs or learning rate while watching the holdout score gradually leaks that data into your decisions. Chapter 8 returns to this point for MNIST.

### What to watch while training

                    "
"

- **Steady decrease** of the epoch loss means the updates are helping. Small bumps are normal because every epoch uses different batches.

- **A flat line** can mean the learning rate is too small, the model is too weak, or the units are saturated or dead.

- **A rising loss or non-finite loss** usually means the learning rate is too large. The trainer stops with an exception instead of continuing.

- **Loss near zero on training data but poor holdout results** is overfitting.
                    "
"

When you log, print only counts, epoch numbers and losses. Do not echo credentials, file contents, or personal data, and avoid dumping full weight matrices into shared logs; a model's weights can leak information about what it was trained on.

                "
"

                "
"

## 7. Checks You Can Run

The checks verify batch arithmetic (including a batch size larger than the dataset and `Integer.MAX_VALUE`), that a shuffle is a permutation and repeatable under a seed, `Dataset` validation, that the last slice contains only the remaining sample, argument validation in `Trainer`, and that the AND-gate loss after 1000 epochs is lower than after the first. They do *not* claim that every possible initialization converges; they claim only a lower loss for the seeded run.

                    "
"
                        "
""
"src/test/java/com/neuralscratch/training/TrainerChecks.java"
"Copy"
"

```java
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
```

                    "
"
                    "
"
                        "
""
"PowerShell (project root)"
"Copy"
"

```
New-Item -ItemType Directory -Force out | Out-Null
javac -d out (Get-ChildItem -Recurse -Path src -Filter *.java).FullName
java -cp out com.neuralscratch.training.TrainerChecks
```

                    "
"
                    "
"
                        "
""
"macOS / Linux shell (project root)"
"Copy"
"

```
mkdir -p out
javac -d out $(find src -name "*.java")
java -cp out com.neuralscratch.training.TrainerChecks
```

                    "
"

It prints the first and last AND-gate epoch losses and, if everything passes, `All trainer checks passed`.

                    "
"
                        "
"Assumptions about earlier chapters"
"

As in Chapter 6, the demos use `new Sigmoid()` and `new MeanSquaredError()` with no-argument constructors, `DenseLayer.random(inputSize, outputSize, activation, random)`, `new Network(layer…)`, `network.layers()`, and `network.predict(Matrix)`. `Trainer` relies on `DenseLayer.inputSize()` and `outputSize()` for its shape checks and on `Backpropagation.trainBatch(network, inputs, targets, objective, learningRate)` returning the pre-update mean loss.

                    "
"
                "
"

                "
"

## What You Now Have

                    "
"

- `Dataset`: a validated, immutable pair of input and target matrices with a slice-copy method.

- `Trainer.train`: seeded Fisher-Yates shuffling, correctly sized final batches, one sample-weighted pre-update loss per epoch, and early failure on divergence.

- An AND-gate demo and checks that report measured behavior without promising convergence for every setting.
                    "
"
                    "
"
                        "
"Before Chapter 8"
"

Change the AND demo's batch size to 3, then to 4, and compare the printed batch count and the loss curves. Then change `new Random(42)` to another seed and observe how the first-epoch loss changes. You are seeing the role of initialization first-hand.

                    "
"

Next, we try to teach XOR to a network, learn why it needs a hidden layer, and read the real MNIST digit files to measure accuracy on images the network has never seen.

                "
"

                "
"
                    ← Previous: Chapter 6
                    Next: Chapter 8 →
                "
"
            