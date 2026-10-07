package nn.activation;

public interface ActivationFunction {
    double activate(double value);
    double derivativeFromOutput(double output);
}
