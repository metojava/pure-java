package nn.layer;

import nn.activation.ActivationFunction;
import nn.math.Vector;

public final class ActivationLayer implements Layer {
    private final ActivationFunction function;
    private Vector lastOutput;
    public ActivationLayer(ActivationFunction function) { this.function = function; }
    @Override public Vector forward(Vector input) {
        double[] output = new double[input.size()];
        for (int i = 0; i < output.length; i++) output[i] = function.activate(input.get(i));
        return lastOutput = new Vector(output);
    }
    @Override public Vector backward(Vector gradient, double learningRate) {
        if (lastOutput == null) throw new IllegalStateException("forward must be called before backward");
        double[] output = new double[gradient.size()];
        for (int i = 0; i < output.length; i++) output[i] = gradient.get(i) * function.derivativeFromOutput(lastOutput.get(i));
        return new Vector(output);
    }
}
