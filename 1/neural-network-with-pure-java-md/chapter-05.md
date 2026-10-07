## Chapter 5: The Core of Learning – Backpropagation & Gradient Descent

Forward propagation lets the network make predictions.

Learning requires one additional capability:
the network must determine how its parameters contributed
to the error.

This is where backpropagation enters.

### 5.1 Prediction Versus Learning

Imagine that our network predicts:

prediction = 0.72

while the correct answer is:

target = 1.00

We need a numerical measurement of how wrong the prediction was.
That measurement is called the loss.

### 5.2 Mean Squared Error

For one prediction:

L = 1/2 (y − ŷ)²

where:

- **y** is the target

- **ŷ** is the prediction

The derivative with respect to the prediction is:

∂L/∂ŷ = ŷ − y

### 5.3 The Chain Rule

Suppose:

z = wx + b

and:

y = σ(z)

The loss depends on y.
Therefore, to determine how the loss changes with respect to
w, we can follow the chain:

∂L/∂w
=
∂L/∂y
×
∂y/∂z
×
∂z/∂w

Since:

z = wx + b

we have:

∂z/∂w = x

Therefore:

∂L/∂w
=
∂L/∂y
×
σ′(z)
×
x

### 5.4 Gradient Descent

Once we have a gradient, the parameter can be updated:

w ← w − η ∂L/∂w

where η is the learning rate.

### 5.5 A Trainable Neuron

```java
package nn.network;

import nn.activation.Sigmoid;

import java.util.Random;

public final class TrainableNeuron {

    private final double[] weights;

    private double bias;

    public TrainableNeuron(int inputCount, Random random) {

        weights =
            new double[inputCount];

        double scale =
            1.0 / Math.sqrt(inputCount);

        for (int i = 0; i < inputCount; i++) {

            weights[i] =
                (random.nextDouble() * 2 - 1)
                * scale;
        }
    }

    public double predict( double[] input) {

        double z = bias;

        for (int i = 0; i < input.length; i++) {

            z +=
                input[i] * weights[i];
        }

        return Sigmoid.activate(z);
    }

    public void train( double[] input, double target, double learningRate) {

        double z = bias;

        for (int i = 0; i < input.length; i++) {

            z +=
                input[i] * weights[i];
        }

        double output =
            Sigmoid.activate(z);

        /*
         * Mean squared error derivative:
         *
         * dL/dOutput = output - target
         */
        double dLossDOutput =
            output - target;

        /*
         * Sigmoid derivative:
         *
         * dOutput/dZ =
         * sigmoid(Z) * (1 - sigmoid(Z))
         */
        double dOutputDZ =
            output * (1.0 - output);

        /*
         * Chain rule:
         *
         * dLoss/dZ =
         * dLoss/dOutput *
         * dOutput/dZ
         */
        double dLossDZ =
            dLossDOutput
            * dOutputDZ;

        /*
         * dZ/dWeight = input
         *
         * Therefore:
         *
         * dLoss/dWeight =
         * dLoss/dZ * input
         */
        for (int i = 0; i < weights.length; i++) {

            double gradient =
                dLossDZ * input[i];

            weights[i] -=
                learningRate * gradient;
        }

        /*
         * dZ/dBias = 1
         */
        bias -=
            learningRate * dLossDZ;
    }
}
```

### 5.6 Backpropagation Through a Layer

For a dense layer:

z = Wx + b

Let:

δ = ∂L/∂z

Then the weight gradient is:

∂L/∂W = δxᵀ

and the bias gradient is:

∂L/∂b = δ

The gradient that must be sent to the previous layer is:

∂L/∂x = Wᵀδ

### 5.7 Dense Layer Backward Pass

```
public Vector backward( Vector gradient, double learningRate) {

    /*
     * gradient represents dL/dZ.
     */

    double[] inputValues =
        lastInput.toArray();

    /*
     * Save W^T * gradient before
     * changing W.
     */
    Vector inputGradient =
        weights.transpose()
               .multiply(gradient);

    for (int row = 0; row < weights.rows(); row++) {

        double outputGradient =
            gradient.get(row);

        for (int column = 0; column < weights.columns(); column++) {

            double weightGradient =
                outputGradient
                * inputValues[column];

            /*
             * Gradient descent:
             *
             * W = W - learningRate * dL/dW
             */
            double updatedWeight =
                weights.get(row, column)
                - learningRate
                * weightGradient;

            weights.set(
                row,
                column,
                updatedWeight
            );
        }

        double updatedBias =
            biases.get(row)
            - learningRate
            * outputGradient;

        biases.set(
            row,
            updatedBias
        );
    }

    return inputGradient;
}
```

### 5.8 Why Backpropagation Is Efficient

A naive approach might slightly change every weight, run the
entire network again, and estimate the effect.

That would require enormous numbers of forward passes.

Backpropagation reuses intermediate calculations and applies
the chain rule systematically.

This makes it possible to calculate thousands or millions of
parameter gradients efficiently.

### 5.9 Gradient Checking

Backpropagation is mathematically compact but easy to implement
incorrectly.

Numerical gradient checking provides an excellent debugging
technique.

numericalGradient
=
[L(θ + ε) − L(θ − ε)] / (2ε)

We compare this approximation with the gradient produced by
backpropagation.

```
static double numericalDerivative(
        java.util.function.DoubleFunction<Double> function,
        double x) {

    double epsilon = 1e-5;

    return (
        function.apply(x + epsilon)
        -
        function.apply(x - epsilon)
    ) / (2.0 * epsilon);
}
```

If your network learns very slowly or produces completely
unexpected results, do not immediately assume that the learning
rate is wrong.

First verify the gradients.

