## Chapter 6: Activation & Loss Functions

Different problems benefit from different activation and loss
functions.

### 6.1 ReLU

ReLU means Rectified Linear Unit.

ReLU(x) = max(0, x)

Java:

```java
package nn.activation;

public final class ReLU
        implements ActivationFunction {

    @Override
    public double activate( double value) {

        return Math.max(0.0, value);
    }

    @Override
    public double derivative( double value) {

        return value > 0.0
                ? 1.0
                : 0.0;
    }
}
```

### 6.2 Why ReLU?

ReLU is computationally simple and works well in many hidden
layers.

Unlike sigmoid, its positive region does not saturate as quickly.

However, neurons can become permanently inactive when they
remain on the negative side.

### 6.3 Softmax

Softmax converts a collection of arbitrary scores into
probabilities whose sum equals one.

softmax(zᵢ)
=
eᶻⁱ / Σⱼ eᶻʲ

A naive implementation can overflow when values become large.
We therefore subtract the largest input first.

```java
package nn.activation;

public final class Softmax {

    private Softmax() {
    }

    public static double[] activate( double[] values) {

        double maximum =
            Double.NEGATIVE_INFINITY;

        for (double value : values) {

            maximum =
                Math.max(maximum, value);
        }

        double[] exponentials =
            new double[values.length];

        double sum = 0.0;

        for (int i = 0; i < values.length; i++) {

            /*
             * Subtracting the maximum does not
             * change the final probability ratio,
             * but greatly improves numerical stability.
             */
            exponentials[i] =
                Math.exp(values[i] - maximum);

            sum += exponentials[i];
        }

        double[] probabilities =
            new double[values.length];

        for (int i = 0; i < values.length; i++) {

            probabilities[i] =
                exponentials[i] / sum;
        }

        return probabilities;
    }
}
```

### 6.4 Mean Squared Error

Mean squared error measures the average squared difference
between prediction and target.

MSE =
1/n Σ(ŷᵢ − yᵢ)²

```java
package nn.loss;

public final class MeanSquaredError {

    private MeanSquaredError() {
    }

    public static double calculate( double[] prediction, double[] target) {

        if (prediction.length !=
            target.length) {

            throw new IllegalArgumentException("Prediction and target sizes differ");
        }

        double sum = 0.0;

        for (int i = 0; i < prediction.length; i++) {

            double difference =
                prediction[i] - target[i];

            sum +=
                difference * difference;
        }

        return sum / prediction.length;
    }
}
```

### 6.5 Cross-Entropy

For classification problems, cross-entropy is often more
appropriate than MSE.

For one correct class:

L = −log(p)

where p is the predicted probability assigned to the correct
class.

```java
package nn.loss;

public final class CrossEntropy {

    private CrossEntropy() {
    }

    public static double calculate( double[] probabilities, int correctClass) {

        double probability =
            probabilities[correctClass];

        /*
         * Avoid log(0).
         */
        probability =
            Math.max(
                probability,
                1e-15
            );

        return -Math.log(probability);
    }
}
```

### 6.6 Why Softmax and Cross-Entropy Work Well Together

When softmax is combined with cross-entropy, the derivative
becomes especially elegant.

For each output class:

∂L/∂zᵢ = pᵢ − yᵢ

where p is the softmax probability vector and y is the
one-hot target vector.

This simple expression is one of the most useful formulas
in practical neural-network implementation.

### 6.7 One-Hot Encoding

Suppose the classes are:

```
0 = cat
1 = dog
2 = bird
```

If the correct class is dog:

```
[0, 1, 0]
```

If the network predicts:

```
[0.10, 0.80, 0.10]
```

the model has assigned 80% probability to the correct class.

