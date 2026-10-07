package nn.activation;

public final class SigmoidActivation implements ActivationFunction {
    @Override public double activate(double value) { return value >= 0 ? 1.0 / (1.0 + Math.exp(-value)) : Math.exp(value) / (1.0 + Math.exp(value)); }
    @Override public double derivativeFromOutput(double output) { return output * (1.0 - output); }
}
