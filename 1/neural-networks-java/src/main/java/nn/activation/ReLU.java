package nn.activation;

public final class ReLU implements ActivationFunction {
    @Override public double activate(double value) { return Math.max(0.0, value); }
    @Override public double derivativeFromOutput(double output) { return output > 0.0 ? 1.0 : 0.0; }
}
