## Chapter 8: Evaluation, XOR & MNIST

We have reached the point where all the individual pieces can
be combined into complete machine-learning experiments.

We will begin with XOR because it is small enough to inspect
manually.

We will then move to MNIST, where each example is a handwritten
digit.

### 8.1 Why XOR?

XOR is a classic neural-network problem.

A

B

XOR

0

0

0

0

1

1

1

0

1

1

1

0

XOR cannot be solved by a single linear perceptron.
It therefore provides a compact demonstration of why hidden
nonlinear layers are useful.

### 8.2 XOR Network

We can use:

```
2 inputs
    ↓
4 sigmoid neurons
    ↓
1 sigmoid neuron
    ↓
prediction
```

### 8.3 XOR Dataset

```
List<TrainingExample> xor =
    List.of(

        new TrainingExample(new double[] {0, 0}, new double[] {0}),

        new TrainingExample(new double[] {0, 1}, new double[] {1}),

        new TrainingExample(new double[] {1, 0}, new double[] {1}),

        new TrainingExample(new double[] {1, 1}, new double[] {0})
    );
```

### 8.4 Training XOR

```
Random random =
    new Random(42);

NeuralNetwork network =
    new NeuralNetwork();

network.add(
    new DenseLayer(2, 4, random)
);

network.add(
    new ActivationLayer(new SigmoidActivation()
    )
);

network.add(
    new DenseLayer(4, 1, random)
);

network.add(
    new ActivationLayer(new SigmoidActivation()
    )
);
```

### 8.5 Expected Output

After successful training, predictions should approach:

```
Input       Expected    Prediction

0, 0          0         ~0.0
0, 1          1         ~1.0
1, 0          1         ~1.0
1, 1          0         ~0.0
```

The exact values will vary according to initialization,
architecture, learning rate, and training procedure.

### 8.6 Classification Threshold

If our binary classifier produces a sigmoid output, we can use
0.5 as a basic decision threshold.

```
int predictedClass =
    prediction >= 0.5
        ? 1
        : 0;
```

### 8.7 Accuracy

Accuracy is:

accuracy =
correctPredictions / totalPredictions

```
public static double accuracy( int[] predicted, int[] actual) {

    if (predicted.length != actual.length) {
        throw new IllegalArgumentException("Array lengths differ");
    }

    int correct = 0;

    for (int i = 0; i < predicted.length; i++) {

        if (predicted[i] == actual[i]) {
            correct++;
        }
    }

    return
        (double) correct
        / predicted.length;
}
```

### 8.8 Moving to MNIST

MNIST contains grayscale images of handwritten digits from
0 through 9.

Each image contains:

28 × 28 = 784

pixels.

Therefore, the input to our network can be represented as a
vector containing 784 values.

### 8.9 Pixel Normalization

MNIST pixel values are normally represented as unsigned
8-bit values from 0 to 255.

Neural networks generally benefit from smaller numerical
ranges.

We can normalize:

normalizedPixel = pixel / 255.0

### 8.10 Reading Binary Data in Java

We do not need an external dataset library.
Java's DataInputStream can read the MNIST binary format.

```java
package nn.data;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

public final class MnistReader {

    private MnistReader() {
    }

    public static Dataset readImages(
            InputStream inputStream)
            throws IOException {

        DataInputStream input =
            new DataInputStream(inputStream);

        int magic =
            input.readInt();

        int imageCount =
            input.readInt();

        int rows =
            input.readInt();

        int columns =
            input.readInt();

        if (magic != 2051) {

            throw new IOException("Invalid MNIST image file");
        }

        double[][] images =
            new double[imageCount]
                       [rows * columns];

        for (int image = 0; image < imageCount; image++) {

            for (int pixel = 0; pixel < rows * columns; pixel++) {

                int value =
                    input.readUnsignedByte();

                images[image][pixel] =
                    value / 255.0;
            }
        }

        return new Dataset(images, rows, columns);
    }

    public record Dataset( double[][] images, int rows, int columns ) {
    }
}
```

### 8.11 Reading MNIST Labels

```
public static int[] readLabels(
        InputStream inputStream)
        throws IOException {

    DataInputStream input =
        new DataInputStream(inputStream);

    int magic =
        input.readInt();

    int count =
        input.readInt();

    if (magic != 2049) {

        throw new IOException("Invalid MNIST label file");
    }

    int[] labels =
        new int[count];

    for (int i = 0; i < count; i++) {

        labels[i] =
            input.readUnsignedByte();
    }

    return labels;
}
```

### 8.12 Converting a Label to One-Hot Encoding

```
public static double[] oneHot( int label) {

    double[] result =
        new double[10];

    if (label < 0 ||
        label > 9) {

        throw new IllegalArgumentException("MNIST label must be between 0 and 9");
    }

    result[label] = 1.0;

    return result;
}
```

### 8.13 MNIST Network Architecture

A simple network might contain:

```
784 inputs
    ↓
128 neurons
    ↓
ReLU
    ↓
64 neurons
    ↓
ReLU
    ↓
10 neurons
    ↓
Softmax

```

The final ten outputs correspond to:

```
0 1 2 3 4 5 6 7 8 9
```

### 8.14 Why Ten Outputs?

The network needs to produce one score for every possible digit.

For example:

```
Digit       Probability

0           0.001
1           0.002
2           0.004
3           0.003
4           0.010
5           0.015
6           0.002
7           0.900
8           0.020
9           0.043
```

The predicted digit is the index with the largest probability.

### 8.15 Argmax

```
public static int argmax( double[] values) {

    int bestIndex = 0;

    double bestValue =
        values[0];

    for (int i = 1; i < values.length; i++) {

        if (values[i] > bestValue) {

            bestValue =
                values[i];

            bestIndex =
                i;
        }
    }

    return bestIndex;
}
```

### 8.16 Displaying an MNIST Image in the Terminal

We can even render an image using ordinary characters.

```
public static void printImage( double[] pixels, int rows, int columns) {

    for (int row = 0; row < rows; row++) {

        for (int column = 0; column < columns; column++) {

            double value =
                pixels[
                    row * columns + column
                ];

            char character;

            if (value > 0.75) {
                character = '#';
            } else if (value > 0.50) {
                character = 'O';
            } else if (value > 0.25) {
                character = '+';
            } else if (value > 0.05) {
                character = '.';
            } else {
                character = ' ';
            }

            System.out.print(character);
        }

        System.out.println();
    }
}
```

### 8.17 Evaluating MNIST

A basic evaluation procedure is:

- Load the test images.

- Load the test labels.

- Run every image through the network.

- Find the output with the highest probability.

- Compare it with the known label.

- Count correct predictions.

### 8.18 Confusion Matrix

Accuracy tells us how many predictions are correct, but it does
not tell us which digits are confused with each other.

A confusion matrix solves this problem.

```
int[][] confusionMatrix =
    new int[10][10];

for (int i = 0; i < predictions.length; i++) {

    int actual =
        labels[i];

    int predicted =
        predictions[i];

    confusionMatrix[actual][predicted]++;
}
```

The row represents the actual digit.
The column represents the predicted digit.

### 8.19 Improving the Pure-Java Implementation

Once the implementation works, performance becomes an
engineering problem.

Useful improvements include:

-
Reduce unnecessary object allocation.

-
Reuse arrays where practical.

-
Prefer contiguous primitive arrays for hot loops.

-
Avoid repeated conversion between arrays and objects.

-
Profile before optimizing.

-
Use batch matrix operations where appropriate.

### 8.20 Why We Still Do Not Need an ML Library

At this point, we have implemented the essential concepts behind
a small neural-network framework ourselves:

- Numerical vectors

- Matrices

- Matrix multiplication

- Transpose

- Neurons

- Dense layers

- Activation functions

- Loss functions

- Forward propagation

- Backpropagation

- Gradient descent

- Training loops

- Dataset loading

- Classification

- Evaluation

### 8.21 Final Project Architecture

A complete implementation can eventually look like:

```
src/main/java/nn/

├── Main.java
│
├── math/
│   ├── Vector.java
│   └── Matrix.java
│
├── activation/
│   ├── ActivationFunction.java
│   ├── SigmoidActivation.java
│   ├── ReLU.java
│   └── Softmax.java
│
├── loss/
│   ├── MeanSquaredError.java
│   └── CrossEntropy.java
│
├── layer/
│   ├── Layer.java
│   ├── DenseLayer.java
│   └── ActivationLayer.java
│
├── network/
│   ├── NeuralNetwork.java
│   └── TrainableNeuron.java
│
├── training/
│   ├── Trainer.java
│   └── TrainingExample.java
│
└── data/
    └── MnistReader.java
```

### 8.22 Final Checklist

By the end of this book, you should be able to answer all of the
following:

-
What does a neuron calculate?

-
Why do neural networks need weights?

-
What is the purpose of a bias?

-
What is a dot product?

-
How does matrix multiplication work?

-
Why is transposition useful?

-
What does an activation function do?

-
What is a loss function?

-
What is a derivative?

-
What does a gradient represent?

-
How does gradient descent update a parameter?

-
How does the chain rule enable backpropagation?

-
Why does a hidden nonlinear layer allow a network to
solve XOR?

-
Why does a classification network commonly use softmax
at its output?

-
How can MNIST be read without an external machine-learning
library?

### 8.23 The Bigger Picture

A neural network can look like an enormous and complicated
system when viewed from the outside.

But at its foundation, the system is a composition of relatively
simple operations.

numbers
→ multiplication
→ addition
→ nonlinear transformation
→ error measurement
→ derivatives
→ parameter updates

Repeat those operations many times, and the system learns.

That is the central idea.

The power of modern neural networks comes not from one mysterious
equation, but from composing simple mathematical operations
into large systems and efficiently optimizing their parameters.

**Final principle:**

You do not need a machine-learning framework to understand a neural
network.

Frameworks are useful engineering tools, but understanding begins
when you can look at the underlying operations and explain what
every multiplication, addition, derivative, and parameter update
is doing.

## Appendix A: Minimal Pure-Java Example

The following compact example demonstrates the central idea
behind a trainable binary neuron.

```java
import java.util.Random;

public class SimpleNeuron {

    private final double[] weights;

    private double bias;

    public SimpleNeuron(int inputCount, Random random) {

        weights =
            new double[inputCount];

        for (int i = 0; i < inputCount; i++) {

            weights[i] =
                random.nextDouble() - 0.5;
        }
    }

    private double sigmoid( double value) {

        return 1.0 /
            (1.0 + Math.exp(-value));
    }

    public double predict( double[] input) {

        double z = bias;

        for (int i = 0; i < weights.length; i++) {

            z +=
                weights[i]
                * input[i];
        }

        return sigmoid(z);
    }

    public void train( double[] input, double target, double learningRate) {

        double z = bias;

        for (int i = 0; i < weights.length; i++) {

            z +=
                weights[i]
                * input[i];
        }

        double output =
            sigmoid(z);

        /*
         * MSE derivative with respect
         * to the output.
         */
        double dLossDOutput =
            output - target;

        /*
         * Sigmoid derivative.
         */
        double dOutputDZ =
            output * (1.0 - output);

        /*
         * Chain rule.
         */
        double dLossDZ =
            dLossDOutput
            * dOutputDZ;

        /*
         * Update each weight.
         */
        for (int i = 0; i < weights.length; i++) {

            double gradient =
                dLossDZ * input[i];

            weights[i] -=
                learningRate * gradient;
        }

        /*
         * Since dZ/dBias = 1,
         * the bias gradient is dLossDZ.
         */
        bias -=
            learningRate * dLossDZ;
    }

    public static void main( String[] args) {

        Random random =
            new Random(42);

        SimpleNeuron neuron =
            new SimpleNeuron(2, random);

        double[][] inputs = {

            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] targets = {

            0,
            1,
            1,
            1
        };

        for (int epoch = 0; epoch < 5000; epoch++) {

            for (int i = 0; i < inputs.length; i++) {

                neuron.train(
                    inputs[i],
                    targets[i],
                    0.1
                );
            }
        }

        for (double[] input : inputs) {

            System.out.printf(
                "[%.0f, %.0f] -> %.4f%n",
                input[0],
                input[1],
                neuron.predict(input)
            );
        }
    }
}
```

**Neural Networks from Scratch in Pure Java**

Java 17+ · Standard Library Only ·
Built from First Principles

This educational implementation prioritizes clarity
and understanding over production-scale performance.

