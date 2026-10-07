## Chapter 4: Building the Network Layers

A useful neural network is rarely a single neuron.
We normally connect many neurons into layers.

The most common basic layer is a dense, or fully connected,
layer.

### 4.1 What Is a Dense Layer?

Suppose we have four input values and three neurons.
Every neuron receives all four inputs.

The layer therefore contains:

4 × 3 = 12 weights

plus three biases.

If the input is represented as a column vector:

x

the layer can be expressed as:

z = Wx + b

### 4.2 Dense Layer Class

```java
package nn.layer;

import nn.math.Matrix;
import nn.math.Vector;

import java.util.Random;

public final class DenseLayer {

    private final Matrix weights;

    private final Vector biases;

    private Vector lastInput;

    private Vector lastPreActivation;

    public DenseLayer(int inputCount, int outputCount, Random random) {

        double scale =
            Math.sqrt(2.0 / inputCount);

        this.weights =
            Matrix.random(
                outputCount,
                inputCount,
                random,
                scale
            );

        this.biases =
            new Vector(outputCount);
    }

    public Vector forward( Vector input) {

        if (input.size() !=
            weights.columns()) {

            throw new IllegalArgumentException("Input dimension mismatch");
        }

        lastInput = input;

        Vector weighted =
            weights.multiply(input);

        double[] result =
            new double[weighted.size()];

        for (int i = 0; i < weighted.size(); i++) {

            result[i] =
                weighted.get(i)
                + biases.get(i);
        }

        lastPreActivation =
            new Vector(result);

        return lastPreActivation;
    }

    public int inputCount() {

        return weights.columns();
    }

    public int outputCount() {

        return weights.rows();
    }
}
```

### 4.3 Forward Propagation

Forward propagation means moving input data from the beginning
of the network toward the output.

For a two-layer network:

input → Layer 1 → activation → Layer 2 → output

Every layer consumes the previous layer's output.

### 4.4 Adding an Activation Function

A clean design separates the dense transformation from the
activation function.

We can define an interface:

```java
package nn.activation;

public interface ActivationFunction {

    double activate(double value);

    double derivative(double value);
}
```

### 4.5 Sigmoid Activation Object

```java
package nn.activation;

public final class SigmoidActivation
        implements ActivationFunction {

    @Override
    public double activate( double value) {

        return 1.0 /
            (1.0 + Math.exp(-value));
    }

    @Override
    public double derivative( double value) {

        double output =
            activate(value);

        return output * (1.0 - output);
    }
}
```

### 4.6 Activation as a Layer

```java
package nn.layer;

import nn.activation.ActivationFunction;
import nn.math.Vector;

public final class ActivationLayer {

    private final ActivationFunction function;

    private Vector lastInput;

    public ActivationLayer(ActivationFunction function) {

        this.function = function;
    }

    public Vector forward( Vector input) {

        lastInput = input;

        double[] result =
            new double[input.size()];

        for (int i = 0; i < input.size(); i++) {

            result[i] =
                function.activate(
                    input.get(i)
                );
        }

        return new Vector(result);
    }
}
```

### 4.7 The Network

We can now build a simple sequential neural network.

```java
package nn.network;

import nn.math.Vector;

import java.util.ArrayList;
import java.util.List;

public final class NeuralNetwork {

    private final List<Object> layers =
        new ArrayList<>();

    public void add(Object layer) {

        layers.add(layer);
    }

    public Vector predict(Vector input) {

        Vector current = input;

        for (Object layer : layers) {

            if (layer instanceof
                    nn.layer.DenseLayer dense) {

                current =
                    dense.forward(current);

            } else if (layer instanceof
                    nn.layer.ActivationLayer activation) {

                current =
                    activation.forward(current);
            }
        }

        return current;
    }
}
```

In a production library we would probably introduce a common
layer interface. We are intentionally keeping the implementation
straightforward while learning the underlying mechanics.

### 4.8 Network Architecture

A small binary classifier might use:

```
Input:      2 values

Hidden:     4 neurons
Activation: Sigmoid

Output:     1 neuron
Activation: Sigmoid
```

Conceptually:

x
→ W₁x + b₁
→ sigmoid
→ W₂h + b₂
→ sigmoid
→ prediction

### 4.9 Why Multiple Layers?

A single linear transformation can only represent a limited
class of functions.

By introducing nonlinear activation functions between layers,
we can compose increasingly complex transformations.

This is the basic reason multilayer neural networks can solve
problems that a single perceptron cannot.

**Exercise:**

Draw a network containing:

- 2 input neurons

- 3 hidden neurons

- 1 output neuron

Then calculate the total number of weights and biases.

Answer:

weights = (2 × 3) + (3 × 1) = 9

biases = 3 + 1 = 4

