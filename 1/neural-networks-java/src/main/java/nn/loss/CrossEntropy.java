package nn.loss;

import nn.math.Vector;

public final class CrossEntropy implements Loss {
    @Override public double value(Vector prediction, Vector target) {
        check(prediction, target); double sum = 0;
        for (int i = 0; i < prediction.size(); i++) sum -= target.get(i) * Math.log(Math.max(prediction.get(i), 1e-12));
        return sum;
    }
    @Override public Vector gradient(Vector prediction, Vector target) {
        check(prediction, target); double[] gradient = new double[prediction.size()];
        for (int i = 0; i < gradient.length; i++) gradient[i] = -target.get(i) / Math.max(prediction.get(i), 1e-12);
        return new Vector(gradient);
    }
    private static void check(Vector a, Vector b) { if (a.size() != b.size()) throw new IllegalArgumentException("loss shape mismatch"); }
}
