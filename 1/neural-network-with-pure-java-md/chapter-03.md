## Chapter 3: The Architecture of a Neuron

We now have the mathematical foundation required to build our
first actual neural-network component.

A neuron is a small mathematical function that transforms
several input values into one output value.

### 3.1 The Mathematical Model

Suppose a neuron receives three inputs:

x₁, x₂, x₃

Each input has an associated weight:

w₁, w₂, w₃

The neuron calculates:

z =
x₁w₁ +
x₂w₂ +
x₃w₃ +
b

This is simply a dot product followed by the addition of a bias.

The result is then passed through an activation function:

y = activation(z)

Therefore, the complete neuron can be written as:

y = activation(x · w + b)

### 3.2 Why Do We Need Weights?

Imagine a classifier that receives two measurements:

- Hours spent studying

- Hours spent sleeping

Perhaps studying has a stronger relationship with the prediction
than sleeping.

The neural network can represent that difference through its
weights.

A large positive weight means that increasing the input tends
to increase the neuron's output.

A negative weight means that increasing the input tends to
decrease the neuron's output.

### 3.3 The Bias

The bias is an additional parameter that allows the neuron to
shift its activation threshold.

z = x · w + b

Without the bias, the neuron would always be constrained by
a particular origin.

The bias gives the model additional flexibility.

### 3.4 The Perceptron

One of the earliest useful neuron models is the perceptron.

The classic perceptron uses a step function:

step(x) =
1, if x ≥ 0
0, otherwise

In Java:

```java
package nn.activation;

public final class StepActivation {

    private StepActivation() {
    }

    public static double activate(double value) {

        return value >= 0.0
                ? 1.0
                : 0.0;
    }
}
```

### 3.5 A Simple Perceptron

```java
package nn.network;

import nn.activation.StepActivation;

public final class Perceptron {

    private final double[] weights;

    private double bias;

    public Perceptron(int inputCount) {

        if (inputCount <= 0) {
            throw new IllegalArgumentException("Input count must be positive");
        }

        this.weights =
            new double[inputCount];

        this.bias = 0.0;
    }

    public double predict( double[] input) {

        if (input.length != weights.length) {
            throw new IllegalArgumentException("Input size does not match weights");
        }

        double weightedSum = bias;

        for (int i = 0; i < input.length; i++) {

            weightedSum +=
                input[i] * weights[i];
        }

        return StepActivation.activate(
            weightedSum
        );
    }
}
```

### 3.6 The AND Gate

We can represent logical AND using a perceptron.

A

B

AND

0

0

0

0

1

0

1

0

0

1

1

1

One possible parameterization is:

w₁ = 1
&nbsp;&nbsp;
w₂ = 1
&nbsp;&nbsp;
b = -1.5

When both inputs are one:

1(1) + 1(1) - 1.5 = 0.5

The step function therefore returns one.

### 3.7 The OR Gate

OR can similarly be represented with:

w₁ = 1
&nbsp;&nbsp;
w₂ = 1
&nbsp;&nbsp;
b = -0.5

At least one input being one produces a non-negative value.

### 3.8 The Problem With the Step Function

The step function is useful for understanding the perceptron,
but it creates a problem when we want to train a neural network
with gradient descent.

The step function is discontinuous.

We therefore introduce smoother activation functions.

### 3.9 Sigmoid

The sigmoid function is:

σ(x) = 1 / (1 + e⁻ˣ)

It maps any real number into the interval (0, 1).

Java:

```java
package nn.activation;

public final class Sigmoid {

    private Sigmoid() {
    }

    public static double activate( double value) {

        return 1.0 /
            (1.0 + Math.exp(-value));
    }

    public static double derivative( double value) {

        double output =
            activate(value);

        return output * (1.0 - output);
    }
}
```

Notice something important about the derivative.

σ′(x) = σ(x)(1 − σ(x))

This means that once we have calculated the sigmoid output,
we can calculate its derivative conveniently.

### 3.10 A Trainable Sigmoid Neuron

```java
package nn.network;

import nn.activation.Sigmoid;

import java.util.Random;

public final class SigmoidNeuron {

    private final double[] weights;

    private double bias;

    public SigmoidNeuron(int inputCount, Random random) {

        this.weights =
            new double[inputCount];

        double scale =
            1.0 / Math.sqrt(inputCount);

        for (int i = 0; i < inputCount; i++) {

            weights[i] =
                (random.nextDouble() * 2.0 - 1.0)
                * scale;
        }

        bias = 0.0;
    }

    public double forward( double[] input) {

        double weightedSum = bias;

        for (int i = 0; i < weights.length; i++) {

            weightedSum +=
                input[i] * weights[i];
        }

        return Sigmoid.activate(
            weightedSum
        );
    }
}
```

### 3.11 Why Random Initialization?

Neural networks contain many parameters.
If every neuron started with exactly the same weights, neurons
would initially behave identically.

Random initialization breaks that symmetry.

We also do not want extremely large initial values because
activation functions such as sigmoid can saturate.

A simple initialization strategy is:

weight ∼ Uniform(-1/√n, 1/√n)

where *n* is the number of inputs.

**Checkpoint:**

A neuron is not mysterious.

It is:

**weighted sum → bias → activation**

