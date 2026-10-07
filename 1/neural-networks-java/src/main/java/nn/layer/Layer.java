package nn.layer;

import nn.math.Vector;

public interface Layer {
    Vector forward(Vector input);
    Vector backward(Vector gradient, double learningRate);
}
