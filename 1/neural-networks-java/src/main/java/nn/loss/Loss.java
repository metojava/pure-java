package nn.loss;

import nn.math.Vector;

public interface Loss {
    double value(Vector prediction, Vector target);
    Vector gradient(Vector prediction, Vector target);
}
