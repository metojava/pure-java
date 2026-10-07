package nn.data;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

/** Minimal reader for the standard IDX image and label files used by MNIST. */
public final class MnistReader {
    private MnistReader() { }
    public static double[][] readImages(InputStream source) throws IOException {
        try (DataInputStream input = new DataInputStream(source)) {
            if (input.readInt() != 2051) throw new IOException("not an IDX image file");
            int count = input.readInt(), rows = input.readInt(), columns = input.readInt();
            double[][] images = new double[count][rows * columns];
            for (int image = 0; image < count; image++) for (int pixel = 0; pixel < images[image].length; pixel++) images[image][pixel] = input.readUnsignedByte() / 255.0;
            return images;
        }
    }
    public static int[] readLabels(InputStream source) throws IOException {
        try (DataInputStream input = new DataInputStream(source)) {
            if (input.readInt() != 2049) throw new IOException("not an IDX label file");
            int[] labels = new int[input.readInt()];
            for (int i = 0; i < labels.length; i++) labels[i] = input.readUnsignedByte();
            return labels;
        }
    }
}
