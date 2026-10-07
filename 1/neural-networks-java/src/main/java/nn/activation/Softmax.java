package nn.activation;

import nn.math.Vector;

public final class Softmax {
    public Vector apply(Vector input) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < input.size(); i++) max = Math.max(max, input.get(i));
        double[] values = new double[input.size()]; double sum = 0.0;
        for (int i = 0; i < values.length; i++) { values[i] = Math.exp(input.get(i) - max); sum += values[i]; }
        for (int i = 0; i < values.length; i++) values[i] /= sum;
        return new Vector(values);
    }
}
