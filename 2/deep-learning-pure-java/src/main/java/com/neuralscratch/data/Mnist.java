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