package nn.layer;

import java.util.Random;
import nn.math.Matrix;
import nn.math.Vector;

public final class DenseLayer implements Layer {
    private final Matrix weights;
    private final Vector bias;
    private Vector lastInput;
    public DenseLayer(int inputs, int outputs, Random random) { weights = Matrix.random(outputs, inputs, random); bias = Vector.zeros(outputs); }
    @Override public Vector forward(Vector input) {
        lastInput = input.copy(); Vector output = weights.multiply(input);
        for (int i = 0; i < output.size(); i++) output.set(i, output.get(i) + bias.get(i));
        return output;
    }
    @Override public Vector backward(Vector gradient, double learningRate) {
        if (lastInput == null) throw new IllegalStateException("forward must be called before backward");
        double[] inputGradient = new double[lastInput.size()];
        for (int output = 0; output < gradient.size(); output++) {
            double g = gradient.get(output);
            for (int input = 0; input < lastInput.size(); input++) {
                inputGradient[input] += weights.get(output, input) * g;
                weights.addTo(output, input, -learningRate * g * lastInput.get(input));
            }
            bias.set(output, bias.get(output) - learningRate * g);
        }
        return new Vector(inputGradient);
    }
}
