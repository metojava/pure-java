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