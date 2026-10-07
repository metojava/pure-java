package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;

public interface OutputObjective {
    // L: one number summarising error over the batch (batch x outputs matrices).
    double loss(Matrix logits, Matrix output, Matrix targets);

    // delta = dL/dz for the last layer, shape batch x outputs.
    Matrix delta(Matrix logits, Matrix output, Matrix targets, Activation outputActivation);
}