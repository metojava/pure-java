**Chapter 8**

# Evaluation & Testing

Solve XOR, read raw MNIST files, and measure held-out accuracy without fooling yourself.

## What We Are Building

A network is only useful if you can measure what it does honestly. This final chapter has two experiments and one discipline. First, **XOR**, the smallest problem that needs a hidden layer. Second, **MNIST**, the classic 28×28 handwritten-digit dataset, read from raw IDX files by our own parser and scored on images the network never trained on. The discipline is simple: report measured numbers, separate training data from test data, and check inputs before trusting them.

| File | Role |
| --- | --- |
| `demo/XorDemo.java` | Shows that one neuron fails XOR, then trains a 2 → 4 → 1 network with `Trainer`. |
| `data/Mnist.java` | A validating IDX reader that returns a `Dataset`. |
| `evaluation/Metrics.java` | Argmax, accuracy and confusion matrix in plain Java. |
| `demo/MnistDemo.java` | A runner that trains 784 → 64 → 10 and evaluates held-out test images. |
| `evaluation/EvaluationChecks.java` | Checks for metrics and for IDX magic numbers, counts, truncation and gzip. |

## 1. XOR: Why One Neuron Is Not Enough

XOR ("exclusive or") answers 1 when exactly one of its two inputs is 1:

| x₁ | x₂ | expected target |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This table lists the *targets* we want, not anything a network has produced. A single sigmoid neuron answers 1 when `w₁x₁ + w₂x₂ + b > 0`. Suppose it got all four rows right. Row `(0,0)` needs `b ≤ 0`. Rows `(0,1)` and `(1,0)` need `w₂ + b > 0` and `w₁ + b > 0`. Adding the last two gives `w₁ + w₂ + 2b > 0`. Row `(1,1)` needs `w₁ + w₂ + b ≤ 0`. Subtracting, `b > 0`, contradicting `b ≤ 0`. No weights exist, however long you train. A mathematical impossibility has been proven, and `XorDemo` will confirm that the baseline scores at most 3 of 4.

### What a hidden layer adds

A layer of hidden sigmoid neurons computes several different lines and bends the input square into a new space. In that space the two classes can be separated by one line, which the output neuron draws. The nonlinear activation is essential: without it, stacked layers collapse into one matrix multiplication and remain a single line. This is why Chapter 3's activations matter.

### The XOR runner

The program first trains the one-neuron baseline and reports its accuracy. Then it builds `2 → 4 → 1` with sigmoid activations, trains with `Trainer` and `MeanSquaredError` for 8000 epochs at learning rate 1.0, and prints the predictions. Neural-network training from random weights occasionally lands in a poor region, so the runner tries seeds 1 through 10, one fresh network each, and stops at the first one that gets all four rows right. It reports which seed worked or admits that none did. We do not hide the failures; a restart is a legitimate, honest technique, and the output says exactly what happened.

**src/main/java/com/neuralscratch/demo/XorDemo.java**

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
```

**Run it**

```
java -cp out com.neuralscratch.demo.XorDemo
```

Read the output like this. The baseline table must show at least one row where `decision` differs from `target`. For the hidden-layer network, a successful attempt prints decimals near 0 for rows 1 and 4 and near 1 for rows 2 and 3, followed by `All four XOR rows are correct`. The exact decimals depend on your `DenseLayer.random` initialization, so none are printed here. If no seed succeeds on your machine, raise `EPOCHS`, use more hidden neurons, or vary the learning rate, and rerun.

**Pure Java check for the result**

The program itself is the verification: it compares each thresholded prediction with the table above and counts matches. It never asserts success; it states how many of four rows were correct.

## 2. Reading MNIST from Raw IDX Files

MNIST distributes its images and labels in the **IDX** binary format. All integers are **big-endian** 32-bit values, which is exactly what `DataInputStream.readInt()` reads. An image file looks like this:

| Offset | Content | Expected value |
| --- | --- | --- |
| 0 | magic number | 2051 (`0x00000803`) |
| 4 | image count | for example 60000 or 10000 |
| 8 | rows | 28 |
| 12 | columns | 28 |
| 16 | pixels | count × 784 unsigned bytes, row by row |

A label file has magic number 2049 (`0x00000801`), a count, and then one byte per label from 0 to 9. The magic number is your first defense: a wrong file, a text file, or a compressed file that was not decompressed fails immediately with a clear message instead of producing nonsense.

Java's `byte` is *signed*, so a pixel of 255 would read as −1. `readUnsignedByte()` returns 0 to 255 as an `int`. We divide by 255.0 to get inputs in `[0, 1]`. Small, similarly scaled inputs keep the first layer's pre-activations moderate, which makes gradient descent far better behaved than raw 0–255 values. Each label becomes a **one-hot** row of length 10: all zeros except a 1 at the digit's index.

### Defensive checks in the reader

- Wrong magic numbers, a non-28×28 shape, a non-positive image count, and an image count that differs from the label count all raise `IOException` with an explanatory message.

- A short file is a **truncated** file. `DataInputStream` signals it with `EOFException`; we convert that into a message such as `Truncated IDX data while reading image 41`.

- A label above 9 is rejected.

- `maxSamples` must be positive and the loader reads at most that many rows. This bounds memory *and* prevents an enormous count in a corrupt header from triggering a huge allocation. If it covers the whole file, we also reject extra trailing bytes. If it does not, bytes beyond the requested prefix are not examined, so truncation after the prefix goes unnoticed by design.

- Both gzip-compressed (`.gz`) and raw files work: `load` peeks at the first two bytes for the gzip signature `0x1f 0x8b` and wraps the stream in `GZIPInputStream` only if needed. A try-with-resources block closes both file streams on every path, and `open` closes its stream if detection fails, so nothing leaks.

`Mnist.read` takes already-open streams, which lets our checks feed it small in-memory files; `Mnist.load` does the file handling for real use. We never load a path from a network address, and nothing here downloads data.

**src/main/java/com/neuralscratch/data/Mnist.java**

```java
package com.neuralscratch.data;

import com.neuralscratch.math.Matrix;
import com.neuralscratch.training.Dataset;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.zip.GZIPInputStream;

public final class Mnist {
    public static final int IMAGE_SIDE = 28;
    public static final int PIXELS = IMAGE_SIDE * IMAGE_SIDE;
    public static final int CLASSES = 10;

    private static final int IMAGE_MAGIC = 2051;
    private static final int LABEL_MAGIC = 2049;

    private Mnist() {
    }

    public static Dataset load(Path imagesPath, Path labelsPath, int maxSamples)
            throws IOException {
        Objects.requireNonNull(imagesPath, "imagesPath");
        Objects.requireNonNull(labelsPath, "labelsPath");
        requirePositiveLimit(maxSamples);
        try (InputStream images = open(imagesPath); InputStream labels = open(labelsPath)) {
            return read(images, labels, maxSamples);
        }
    }

    public static Dataset read(InputStream imageStream, InputStream labelStream, int maxSamples)
            throws IOException {
        Objects.requireNonNull(imageStream, "imageStream");
        Objects.requireNonNull(labelStream, "labelStream");
        requirePositiveLimit(maxSamples);
        DataInputStream images = new DataInputStream(new BufferedInputStream(imageStream));
        DataInputStream labels = new DataInputStream(new BufferedInputStream(labelStream));

        int imageMagic = readInt(images, "image magic number");
        if (imageMagic != IMAGE_MAGIC) {
            throw new IOException("Not an IDX image file: magic number " + imageMagic
                + " but expected " + IMAGE_MAGIC);
        }
        int imageCount = readInt(images, "image count");
        int rows = readInt(images, "image rows");
        int cols = readInt(images, "image columns");
        if (rows != IMAGE_SIDE || cols != IMAGE_SIDE) {
            throw new IOException("Expected 28x28 images but file declares " + rows + "x" + cols);
        }

        int labelMagic = readInt(labels, "label magic number");
        if (labelMagic != LABEL_MAGIC) {
            throw new IOException("Not an IDX label file: magic number " + labelMagic
                + " but expected " + LABEL_MAGIC);
        }
        int labelCount = readInt(labels, "label count");

        if (imageCount <= 0) {
            throw new IOException("Image file declares " + imageCount + " images");
        }
        if (imageCount != labelCount) {
            throw new IOException("Image count " + imageCount + " does not match label count "
                + labelCount);
        }

        int samples = Math.min(maxSamples, imageCount);
        double[][] pixels = new double[samples][PIXELS];
        double[][] oneHot = new double[samples][CLASSES];
        for (int sample = 0; sample < samples; sample++) {
            for (int pixel = 0; pixel < PIXELS; pixel++) {
                int value = readUnsignedByte(images, "image " + sample);
                // Scale 0..255 to 0..1 so the first layer sees small inputs.
                pixels[sample][pixel] = value / 255.0;
            }
            int label = readUnsignedByte(labels, "label " + sample);
            if (label >= CLASSES) {
                throw new IOException("Label " + label + " at sample " + sample
                    + " is outside 0..9");
            }
            oneHot[sample][label] = 1.0;
        }
        if (samples == imageCount && (images.read() != -1 || labels.read() != -1)) {
            throw new IOException("IDX file has extra bytes after the declared " + imageCount
                + " samples");
        }
        return new Dataset(Matrix.of(pixels), Matrix.of(oneHot));
    }

    private static InputStream open(Path path) throws IOException {
        InputStream raw = new BufferedInputStream(Files.newInputStream(path));
        try {
            raw.mark(2);
            int first = raw.read();
            int second = raw.read();
            raw.reset();
            if (first == 0x1f && second == 0x8b) {
                return new GZIPInputStream(raw);
            }
            return raw;
        } catch (IOException | RuntimeException failure) {
            raw.close();
            throw failure;
        }
    }

    private static void requirePositiveLimit(int maxSamples) {
        if (maxSamples <= 0) {
            throw new IllegalArgumentException("maxSamples must be positive");
        }
    }

    private static int readInt(DataInputStream in, String what) throws IOException {
        try {
            return in.readInt();
        } catch (EOFException truncated) {
            throw new IOException("Truncated IDX header while reading " + what, truncated);
        }
    }

    private static int readUnsignedByte(DataInputStream in, String what) throws IOException {
        try {
            return in.readUnsignedByte();
        } catch (EOFException truncated) {
            throw new IOException("Truncated IDX data while reading " + what, truncated);
        }
    }
}
```

**Memory budget**

A `double` is 8 bytes. 6000 images × 784 pixels × 8 bytes ≈ 38 MB, and `Matrix.of` temporarily keeps a copy, so peak use is roughly double. The full 60000-image training set would be about ten times that, plus batch copies, so the demo limits itself to 6000 training and 1000 test images. You can raise the limits if your JVM heap allows; add `-Xmx2g` to the `java` command for a larger heap.

## 3. Accuracy and the Confusion Matrix

**Accuracy** is the fraction of samples whose predicted class equals the true class. It is a good first number and a poor only number: a model can score 90 percent while never getting one digit right. A **confusion matrix** counts, for every pair (true class, predicted class), how many samples fell there. Rows are true digits, columns are predictions; the diagonal holds correct answers, and off-diagonal cells show specific mistakes such as 4s read as 9s.

To predict, run `network.predict(testInputs)` to get logits, apply `Softmax.apply` to get probabilities, and take the **argmax** of each row. Softmax never changes which entry is largest, so the argmax of probabilities equals the argmax of logits, but the probabilities are what you would show as "confidence". Targets are one-hot, so the same `argmax` recovers the true label.

**src/main/java/com/neuralscratch/evaluation/Metrics.java**

```java
package com.neuralscratch.evaluation;

import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class Metrics {
    private Metrics() {
    }

    public static int[] argmax(Matrix scores) {
        Objects.requireNonNull(scores, "scores");
        int[] winners = new int[scores.rows()];
        for (int row = 0; row < scores.rows(); row++) {
            int best = 0;
            for (int col = 0; col < scores.cols(); col++) {
                double value = scores.get(row, col);
                if (Double.isNaN(value)) {
                    throw new IllegalArgumentException("NaN score at row " + row);
                }
                // Strict comparison keeps the first class when scores tie.
                if (value > scores.get(row, best)) {
                    best = col;
                }
            }
            winners[row] = best;
        }
        return winners;
    }

    public static double accuracy(int[] actual, int[] predicted) {
        requireSameLength(actual, predicted);
        int correct = 0;
        for (int index = 0; index < actual.length; index++) {
            if (actual[index] == predicted[index]) {
                correct++;
            }
        }
        return (double) correct / actual.length;
    }

    public static int[][] confusionMatrix(int[] actual, int[] predicted, int classes) {
        requireSameLength(actual, predicted);
        if (classes <= 0) {
            throw new IllegalArgumentException("classes must be positive");
        }
        int[][] counts = new int[classes][classes];
        for (int index = 0; index < actual.length; index++) {
            if (actual[index] < 0 || actual[index] >= classes
                    || predicted[index] < 0 || predicted[index] >= classes) {
                throw new IllegalArgumentException("Class index out of range at sample " + index);
            }
            // Row = true class, column = predicted class.
            counts[actual[index]][predicted[index]]++;
        }
        return counts;
    }

    public static String formatConfusion(int[][] counts) {
        StringBuilder text = new StringBuilder("actual\\predicted");
        for (int col = 0; col < counts.length; col++) {
            text.append(String.format("%6d", col));
        }
        text.append(System.lineSeparator());
        for (int row = 0; row < counts.length; row++) {
            text.append(String.format("%16d", row));
            for (int col = 0; col < counts[row].length; col++) {
                text.append(String.format("%6d", counts[row][col]));
            }
            text.append(System.lineSeparator());
        }
        return text.toString();
    }

    private static void requireSameLength(int[] actual, int[] predicted) {
        Objects.requireNonNull(actual, "actual");
        Objects.requireNonNull(predicted, "predicted");
        if (actual.length == 0 || actual.length != predicted.length) {
            throw new IllegalArgumentException(
                "Need equal, non-empty label arrays but got " + actual.length
                    + " and " + predicted.length);
        }
    }
}
```

## 4. Training 784 → 64 → 10 on MNIST

The network has 784 inputs (one per pixel), 64 hidden ReLU neurons, and 10 `Identity` outputs, which are logits for `SoftmaxCrossEntropy`. The parameter count is `784×64 + 64 + 64×10 + 10 = 50,890`. The runner takes four local file paths in this order: training images, training labels, test images, test labels.

It reads the first 6000 training rows and the first 1000 test rows, trains 10 epochs with batch size 32 and learning rate 0.05, and prints the loss for each epoch. It then predicts the held-out images *once* at the end, prints test loss, test accuracy, and the confusion matrix. The program prints only counts, losses and metrics, and it does not write any files or open network connections. If a file cannot be opened, it prints a one-line error and exits with status 1.

**src/main/java/com/neuralscratch/demo/MnistDemo.java**

```java
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
```

### Getting the files

This book does not include the dataset or a download address. Obtain the four MNIST IDX files yourself from a source you trust, such as a copy your course or organization already provides, and verify them against the source's published checksums if available. They are normally named like `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`, with or without a `.gz` suffix. Some mirrors rename files or use different sets, so if the reader rejects a file, read its message: it tells you which header field was wrong.

**PowerShell (replace the four paths with your own files)**

```
java -cp out com.neuralscratch.demo.MnistDemo `
  data\train-images-idx3-ubyte data\train-labels-idx1-ubyte `
  data\t10k-images-idx3-ubyte data\t10k-labels-idx1-ubyte
```

**macOS / Linux shell**

```
java -cp out com.neuralscratch.demo.MnistDemo \
  data/train-images-idx3-ubyte data/train-labels-idx1-ubyte \
  data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte
```

### Reading the result without fooling yourself

No accuracy figure is printed on this page because it would be a guess. The number depends on your files, your initialization code from Chapter 4, and the fixed seed. In our own development runs against a reference implementation of the Chapters 3–5 contract, the loss fell steadily over the 10 epochs and held-out accuracy ended well above the 10 percent expected from random guessing, and it was not near-perfect. Treat that as an observation about one setup, not a promise about yours. Plain gradient descent with a small network and 6000 images is a learning exercise, not a record attempt.

- If `Trainer` reports a non-finite loss, lower the learning rate constant. Large initial weights can overflow on 784 inputs.

- Compare training loss with test loss. A large gap suggests overfitting; both being high suggests underfitting, so try more epochs, more hidden units, or a tuned learning rate.

- Look at the confusion matrix, not just the headline accuracy. Digits whose shapes are similar are mixed up more often.

- The test images must *never* take part in training or tuning. If you want to tune epochs or learning rate, carve a validation set out of the training file (a later slice of its rows, for instance), tune against it, and open the test set once at the end. Our runner touches the test data exactly once, after training.

## 5. Checks for the Reader and the Metrics

Data bugs are quiet; they simply make accuracy bad. These checks build tiny IDX files in memory with `DataOutputStream` and verify the behavior described above. They cover a valid file, the pixel scale (0 maps to 0.0 and 255 to 1.0), one-hot labels, the sample limit, a wrong image magic, a wrong label magic, mismatched counts, a non-28×28 header, a truncated header, truncated pixel data, truncated labels, a label of 10, extra bytes, a zero limit, and a real gzip file written to a temporary directory and deleted afterwards. They also verify argmax tie-breaking, accuracy, and that the confusion matrix counts every sample once.

**src/test/java/com/neuralscratch/evaluation/EvaluationChecks.java**

```java
package com.neuralscratch.evaluation;

import com.neuralscratch.data.Mnist;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.training.Dataset;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;

public final class EvaluationChecks {
    private EvaluationChecks() {
    }

    private interface IoAction {
        void run() throws IOException;
    }

    public static void main(String[] args) throws IOException {
        checkMetrics();
        checkValidIdx();
        checkRejectedIdx();
        checkGzipFiles();
        System.out.println("All evaluation checks passed");
    }

    private static void checkMetrics() {
        Matrix scores = Matrix.of(new double[][] {{0.1, 0.7, 0.2}, {0.5, 0.5, 0.0}, {0, 0, 1}});
        int[] predicted = Metrics.argmax(scores);
        require(predicted[0] == 1 && predicted[1] == 0 && predicted[2] == 2, "argmax and ties");
        int[] actual = {1, 1, 2};
        require(Math.abs(Metrics.accuracy(actual, predicted) - 2.0 / 3.0) < 1e-12, "accuracy");
        int[][] confusion = Metrics.confusionMatrix(actual, predicted, 3);
        require(confusion[1][1] == 1 && confusion[1][0] == 1 && confusion[2][2] == 1,
            "confusion cells");
        int total = 0;
        for (int[] row : confusion) {
            for (int count : row) {
                total += count;
            }
        }
        require(total == actual.length, "confusion counts every sample once");
        expectIllegalArgument(() -> Metrics.confusionMatrix(new int[] {3}, new int[] {0}, 3));
        expectIllegalArgument(() -> Metrics.accuracy(new int[] {1}, new int[] {1, 2}));
        expectIllegalArgument(() -> Metrics.argmax(Matrix.of(new double[][] {{Double.NaN, 1}})));
    }

    private static void checkValidIdx() throws IOException {
        byte[] images = imageFile(3, 28, 28, 3 * 784);
        byte[] labels = labelFile(3, new int[] {0, 7, 9});
        Dataset dataset = Mnist.read(stream(images), stream(labels), 100);
        require(dataset.size() == 3, "all three samples");
        require(dataset.inputs().cols() == 784 && dataset.targets().cols() == 10, "shapes");
        require(dataset.inputs().get(0, 0) == 0.0, "pixel 0 scales to 0.0");
        require(dataset.inputs().get(0, 255) == 1.0, "pixel 255 scales to 1.0");
        require(dataset.targets().get(1, 7) == 1.0 && dataset.targets().get(1, 0) == 0.0,
            "one-hot label");
        Dataset limited = Mnist.read(stream(images), stream(labels), 2);
        require(limited.size() == 2, "limit keeps the first two samples");
    }

    private static void checkRejectedIdx() {
        byte[] labels = labelFile(2, new int[] {1, 2});
        byte[] images = imageFile(2, 28, 28, 2 * 784);
        expectIo(() -> Mnist.read(stream(header(2049, 2, 28, 28)), stream(labels), 10));
        expectIo(() -> Mnist.read(stream(images), stream(header(2051, 2)), 10));
        expectIo(() -> Mnist.read(stream(images), stream(labelFile(3, new int[] {1, 2, 3})), 10));
        expectIo(() -> Mnist.read(stream(imageFile(2, 14, 14, 2 * 196)), stream(labels), 10));
        expectIo(() -> Mnist.read(stream(new byte[] {0, 0, 8}), stream(labels), 10));
        expectIo(() -> Mnist.read(stream(imageFile(2, 28, 28, 784 + 100)), stream(labels), 10));
        expectIo(() -> Mnist.read(stream(images), stream(labelFile(2, new int[] {1, 2}, 1)), 10));
        expectIo(() -> Mnist.read(stream(images), stream(labelFile(2, new int[] {1, 10})), 10));
        expectIo(() -> Mnist.read(stream(imageFile(2, 28, 28, 2 * 784 + 5)), stream(labels), 10));
        expectIllegalArgument(() -> {
            try {
                Mnist.read(stream(images), stream(labels), 0);
            } catch (IOException unexpected) {
                throw new AssertionError(unexpected);
            }
        });
    }

    private static void checkGzipFiles() throws IOException {
        Path directory = Files.createTempDirectory("mnist-check");
        Path imagePath = directory.resolve("images.gz");
        Path labelPath = directory.resolve("labels");
        try {
            try (OutputStream out = new GZIPOutputStream(Files.newOutputStream(imagePath))) {
                out.write(imageFile(2, 28, 28, 2 * 784));
            }
            Files.write(labelPath, labelFile(2, new int[] {4, 5}));
            Dataset dataset = Mnist.load(imagePath, labelPath, 2);
            require(dataset.size() == 2 && dataset.targets().get(1, 5) == 1.0, "gzip image file");
        } finally {
            Files.deleteIfExists(imagePath);
            Files.deleteIfExists(labelPath);
            Files.deleteIfExists(directory);
        }
    }

    private static byte[] header(int... values) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (DataOutputStream out = new DataOutputStream(bytes)) {
            for (int value : values) {
                out.writeInt(value);
            }
        } catch (IOException impossible) {
            throw new AssertionError(impossible);
        }
        return bytes.toByteArray();
    }

    private static byte[] imageFile(int declared, int rows, int cols, int pixelBytes) {
        byte[] head = header(2051, declared, rows, cols);
        byte[] file = new byte[head.length + pixelBytes];
        System.arraycopy(head, 0, file, 0, head.length);
        for (int index = 0; index < pixelBytes; index++) {
            file[head.length + index] = (byte) (index % 256);
        }
        return file;
    }

    private static byte[] labelFile(int declared, int[] labels) {
        return labelFile(declared, labels, labels.length);
    }

    private static byte[] labelFile(int declared, int[] labels, int written) {
        byte[] head = header(2049, declared);
        byte[] file = new byte[head.length + written];
        System.arraycopy(head, 0, file, 0, head.length);
        for (int index = 0; index < written; index++) {
            file[head.length + index] = (byte) labels[index];
        }
        return file;
    }

    private static ByteArrayInputStream stream(byte[] bytes) {
        return new ByteArrayInputStream(bytes);
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError("Check failed: " + message);
        }
    }

    private static void expectIo(IoAction action) {
        try {
            action.run();
            throw new AssertionError("Expected IOException");
        } catch (IOException expected) {
            // Bad IDX data must fail loudly instead of producing a partial dataset.
        }
    }

    private static void expectIllegalArgument(Runnable action) {
        try {
            action.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // Invalid arguments are rejected immediately.
        }
    }
}
```

**PowerShell (project root)**

```
New-Item -ItemType Directory -Force out | Out-Null
javac -d out (Get-ChildItem -Recurse -Path src -Filter *.java).FullName
java -cp out com.neuralscratch.evaluation.EvaluationChecks
```

**macOS / Linux shell (project root)**

```
mkdir -p out
javac -d out $(find src -name "*.java")
java -cp out com.neuralscratch.evaluation.EvaluationChecks
```

It prints `All evaluation checks passed` when every check holds. Nothing in this chapter needs a third-party library: only `java.io`, `java.nio.file`, `java.util` and `java.util.zip`.

**Assumptions about earlier chapters**

The demos use the agreed API: `new Sigmoid()`, `new ReLU()`, `new Identity()`, `new MeanSquaredError()`, `DenseLayer.random(...)`, `new Network(...)` and `network.predict(Matrix)`. If your constructors differ, change only those expressions.

## What You Now Have

- A proof, and a runnable demonstration, that XOR needs a nonlinear hidden layer.

- `Mnist`: a validating, leak-free IDX reader that handles raw and gzip files, scales pixels to `[0, 1]`, builds one-hot labels, and refuses malformed input.

- `Metrics` and `MnistDemo`: argmax, accuracy, confusion matrix, and a held-out evaluation workflow.

- A full dependency-free stack: matrices, calculus checks, neurons, layers, backpropagation, ReLU, softmax cross-entropy, a shuffled trainer, and an evaluation toolkit.
**Where to go next**

Try one change at a time and measure it: more hidden units, more epochs, a different seed, a 20% validation split from the training rows, or a second hidden layer. Because everything is plain Java, you can step through each calculation in a debugger and verify it against the formulas in these chapters.

You started with a single weighted sum and now have a trainable, testable classifier written entirely with the standard library.

[← Previous: Chapter 7](chapter-07.md)

[Back to Table of Contents →](README.md)
