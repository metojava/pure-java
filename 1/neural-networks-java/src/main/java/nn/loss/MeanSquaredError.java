package nn.loss;

import nn.math.Vector;

public final class MeanSquaredError implements Loss {
    @Override public double value(Vector prediction, Vector target) {
        check(prediction, target); double sum = 0;
        for (int i = 0; i < prediction.size(); i++) { double d = prediction.get(i) - target.get(i); sum += d * d; }
        return sum / prediction.size();
    }
    @Override public Vector gradient(Vector prediction, Vector target) {
        check(prediction, target); double[] gradient = new double[prediction.size()];
        for (int i = 0; i < gradient.length; i++) gradient[i] = 2.0 * (prediction.get(i) - target.get(i)) / gradient.length;
        return new Vector(gradient);
    }
    private static void check(Vector a, Vector b) { if (a.size() != b.size()) throw new IllegalArgumentException("loss shape mismatch"); }
}
