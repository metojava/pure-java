**Chapter 4**

# Building the Network Layers

Stack neurons into a dense layer, chain layers into a network, and push a whole batch through with matrix multiplication.

## What We Are Building

Chapter 3 ended with a puzzle: one neuron cannot compute XOR, but three neurons arranged in two stages can. In this chapter we turn that idea into reusable code:

- `com.neuralscratch.nn.DenseLayer`: many neurons that all read the same inputs, stored as one weight matrix and one bias vector.

- `com.neuralscratch.nn.Network`: an ordered list of layers, where each layer's outputs become the next layer's inputs.

We will also initialize weights randomly but reproducibly, trace every shape through a 2 × 3 × 1 network, and verify the hand-built XOR network from Chapter 3. The `Neuron` class stays as a teaching tool; layers do not depend on it.

## 1. From One Neuron to a Layer

Suppose three neurons all receive the same two inputs. Each has its own two weights and its own bias. Writing them out:

z₀ = x₀w₀₀ + x₁w₁₀ + b₀    z₁ = x₀w₀₁ + x₁w₁₁ + b₁    z₂ = x₀w₀₂ + x₁w₁₂ + b₂

Read the subscript `wij` as "from input `i` to neuron `j`". If we place neuron `j`'s weights in *column* `j` of a matrix, the three sums become one row-times-matrix product from Chapter 2:

**One input row, three neurons**

```
x (1 × 2)      W (2 × 3)                 b (size 3)         z (1 × 3)
[ x0  x1 ]  ×  [ w00  w01  w02 ]   +   [ b0  b1  b2 ]   =   [ z0  z1  z2 ]
               [ w10  w11  w12 ]

row i of W  = everything input i sends out
column j    = everything neuron j listens to
```

So the weight matrix of a layer has **rows = input size** and **columns = output size**. Each column is exactly the weight vector of one Chapter 3 neuron.

### A whole batch at once

Now stack several input examples as rows of an input matrix `X`. Matrix multiplication handles every row independently, so the same formula processes the batch in one call. `addRowVector` adds the bias to every row, and `apply` runs the activation on every entry.

A = f(X · W + b)     (batch × in) · (in × out) → (batch × out)

With the Chapter 2 numbers, `X = [[2, 3], [1, 1]]`, `W = [[1, 0, 2], [4, 5, 2]]`, `b = [1, −1, 0]`, and Identity activation, the output is `[[15, 14, 10], [6, 4, 4]]`: two examples in, two rows of three neuron outputs out. Our check program verifies those exact numbers.

## 2. Choosing Starting Weights

Before training, weights need values. Two obvious choices both fail:

- **All zeros (or all equal).** Every neuron in a layer computes the same output and, in Chapter 5, receives the same correction. They stay identical forever, so three neurons behave like one. Random values break this *symmetry*.

- **Large random values.** A pre-activation sums one term per input. Many large terms make `|z|` large, pushing Sigmoid into its flat regions where the slope is nearly zero and learning stalls.

### Xavier (Glorot) initialization in plain terms

The **variance** of a random number measures how spread out it is; the **standard deviation** is its square root. If inputs have variance about 1 and each weight has variance `v`, a sum of `n` independent terms `x·w` has variance about `n·v`. To keep the forward signal's spread steady, we want `v ≈ 1/inputSize`. The backward signal in Chapter 5 flows through the same weights the other way and prefers `v ≈ 1/outputSize`. Xavier initialization compromises with the average:

variance = 2 / (inputSize + outputSize)     stddev = √(2 / (inputSize + outputSize))

For a 2 → 3 layer that is `√(2/5) ≈ 0.632`; for 3 → 1 it is `√(2/4) ≈ 0.707`. We draw each weight from `Random.nextGaussian()` (a bell curve with mean 0, standard deviation 1) and multiply by this stddev. Biases start at zero; the random weights already break symmetry. Chapter 6 introduces a slightly different scale suited to ReLU.

**Reproducibility**

The caller passes a `java.util.Random`. With the same seed and the same order of calls, you get the same weights on every run and every JVM, which makes bugs repeatable and examples comparable.

## 3. The `DenseLayer` Class

"Dense" (or "fully connected") means every input connects to every neuron. The layer holds its parameters and computes a forward pass. It is *mutable* in exactly one controlled way: `applyGradients` replaces the weight matrix and bias vector during training. Because `Matrix` and `Vector` are immutable, the getters can safely return the current values.

**src/main/java/com/neuralscratch/nn/DenseLayer.java**

```java
package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import java.util.Objects;
import java.util.Random;

public final class DenseLayer {
    private Matrix weights;
    private Vector biases;
    private final Activation activation;

    public DenseLayer(Matrix weights, Vector biases, Activation activation) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(biases, "biases");
        Objects.requireNonNull(activation, "activation");
        if (biases.size() != weights.cols()) {
            throw new IllegalArgumentException(
                "Bias size " + biases.size() + " must equal output size " + weights.cols()
            );
        }
        this.weights = weights;
        this.biases = biases;
        this.activation = activation;
    }

    public static DenseLayer random(int inputSize, int outputSize,
                                    Activation activation, Random random) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException("Layer sizes must be positive");
        }
        Objects.requireNonNull(activation, "activation");
        Objects.requireNonNull(random, "random");
        // Xavier/Glorot: Var(w) = 2 / (fanIn + fanOut) keeps signal spread stable.
        double stddev = Math.sqrt(2.0 / (inputSize + outputSize));
        double[][] values = new double[inputSize][outputSize];
        for (int row = 0; row < inputSize; row++) {
            for (int col = 0; col < outputSize; col++) {
                // nextGaussian has mean 0 and stddev 1; scaling sets stddev.
                values[row][col] = random.nextGaussian() * stddev;
            }
        }
        return new DenseLayer(Matrix.of(values), Vector.zeros(outputSize), activation);
    }

    public Matrix weights() {
        return weights;
    }

    public Vector biases() {
        return biases;
    }

    public Activation activation() {
        return activation;
    }

    public int inputSize() {
        return weights.rows();
    }

    public int outputSize() {
        return weights.cols();
    }

    public Matrix forward(Matrix inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.cols() != inputSize()) {
            throw new IllegalArgumentException(
                "Layer expects " + inputSize() + " input columns, got " + inputs.cols()
            );
        }
        // A = f(X W + b): (batch x in)(in x out) -> (batch x out), bias added per row.
        return inputs.multiply(weights).addRowVector(biases).apply(activation::apply);
    }

    public void applyGradients(Matrix weightGradient, Vector biasGradient,
                               double learningRate) {
        Objects.requireNonNull(weightGradient, "weightGradient");
        Objects.requireNonNull(biasGradient, "biasGradient");
        if (weightGradient.rows() != weights.rows() || weightGradient.cols() != weights.cols()) {
            throw new IllegalArgumentException(
                "Weight gradient must be " + weights.rows() + "x" + weights.cols()
                    + ", got " + weightGradient.rows() + "x" + weightGradient.cols()
            );
        }
        if (biasGradient.size() != biases.size()) {
            throw new IllegalArgumentException(
                "Bias gradient must have size " + biases.size() + ", got " + biasGradient.size()
            );
        }
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException(
                "Learning rate must be positive and finite, got " + learningRate
            );
        }
        // Gradient descent: step against the slope, W := W - rate * dL/dW.
        weights = weights.subtract(weightGradient.scale(learningRate));
        biases = biases.subtract(biasGradient.scale(learningRate));
    }
}
```

All validation happens *before* either field changes, so a rejected update never leaves a layer half-modified. We will not call `applyGradients` until Chapter 5; for now just notice its rule: move each parameter a small step in the direction that reduces the error.

`activation::apply` converts our two-method `Activation` into the single-method `DoubleUnaryOperator` that `Matrix.apply` expects. The derivative is unused in a forward pass, which is why a `Step` layer works fine here.

## 4. The `Network` Class

A network is a sequence of layers. Layer `k` produces `outputSize()` columns, so layer `k + 1` must expect exactly that many inputs. We check this *once*, in the constructor, so a mis-wired network fails at construction time instead of in the middle of training.

**src/main/java/com/neuralscratch/nn/Network.java**

```java
package com.neuralscratch.nn;

import com.neuralscratch.math.Matrix;
import java.util.List;
import java.util.Objects;

public final class Network {
    private final List<DenseLayer> layers;

    public Network(DenseLayer... layers) {
        Objects.requireNonNull(layers, "layers");
        if (layers.length == 0) {
            throw new IllegalArgumentException("A network needs at least one layer");
        }
        for (int index = 0; index < layers.length; index++) {
            Objects.requireNonNull(layers[index], "layer " + index);
            if (index > 0 && layers[index - 1].outputSize() != layers[index].inputSize()) {
                throw new IllegalArgumentException(
                    "Layer " + (index - 1) + " outputs " + layers[index - 1].outputSize()
                        + " values but layer " + index + " expects "
                        + layers[index].inputSize()
                );
            }
        }
        // List.of copies the array into an unmodifiable list.
        this.layers = List.of(layers);
    }

    public List<DenseLayer> layers() {
        return layers;
    }

    public int inputSize() {
        return layers.get(0).inputSize();
    }

    public int outputSize() {
        return layers.get(layers.size() - 1).outputSize();
    }

    public Matrix predict(Matrix inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.cols() != inputSize()) {
            throw new IllegalArgumentException(
                "Network expects " + inputSize() + " input columns, got " + inputs.cols()
            );
        }
        Matrix current = inputs;
        for (DenseLayer layer : layers) {
            // Each layer's activations become the next layer's inputs.
            current = layer.forward(current);
        }
        return current;
    }
}
```

The list returned by `layers()` cannot be modified, so callers cannot add, remove, or reorder layers after validation. The layers inside it remain the same objects, which is exactly what Chapter 5 needs: training updates them through `applyGradients`.

## 5. Tracing Shapes Through a 2 × 3 × 1 Network

"2 × 3 × 1" describes layer widths: 2 inputs, a hidden layer of 3 neurons, and 1 output neuron. Feed the four XOR inputs as one batch:

| Value | Computation | Shape |
| --- | --- | --- |
| `X` | input batch | 4 × 2 |
| `W₁`, `b₁` | hidden parameters | 2 × 3, size 3 |
| `Z₁` | `X · W₁ + b₁` | 4 × 3 |
| `A₁` | `σ(Z₁)` | 4 × 3 |
| `W₂`, `b₂` | output parameters | 3 × 1, size 1 |
| `Z₂` | `A₁ · W₂ + b₂` | 4 × 1 |
| `A₂` | `σ(Z₂)` = predictions | 4 × 1 |

The batch size (4) passes straight through; only the column count changes from layer to layer. This network has `2×3 + 3 + 3×1 + 1 = 13` parameters. With random weights its predictions are meaningless; finding good values is the job of Chapter 5.

### XOR with hand-picked weights

Chapter 3's construction fits this structure with a hidden width of 2. Column 0 of `W₁` is the OR neuron (`[1, 1]`, bias `−0.5`); column 1 is NAND (`[−1, −1]`, bias `1.5`). The output layer is AND (`[1, 1]`, bias `−1.5`).

| Input | OR | NAND | AND(OR, NAND) | XOR |
| --- | --- | --- | --- | --- |
| (0, 0) | 0 | 1 | 0 | 0 |
| (0, 1) | 1 | 1 | 1 | 1 |
| (1, 0) | 1 | 1 | 1 | 1 |
| (1, 1) | 1 | 0 | 0 | 0 |

The hidden layer *re-describes* each input in new coordinates (OR, NAND). In those coordinates the two classes are linearly separable, so a single output neuron can finish the job. This is the core reason depth helps.

## 6. A Runnable Demo

The demo creates a random 2 × 3 × 1 Sigmoid network from seed 42, prints its weights and intermediate shapes, and then runs the hand-built XOR network. It formats numbers with `Locale.ROOT` so the output looks the same everywhere.

**src/main/java/com/neuralscratch/examples/Chapter4Demo.java**

```java
package com.neuralscratch.examples;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.Locale;
import java.util.Random;

public final class Chapter4Demo {
    private Chapter4Demo() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});

        Random random = new Random(42);
        DenseLayer hidden = DenseLayer.random(2, 3, new Sigmoid(), random);
        DenseLayer output = DenseLayer.random(3, 1, new Sigmoid(), random);
        Network network = new Network(hidden, output);

        System.out.println("Random 2x3x1 network, seed 42");
        print("hidden weights W1", hidden.weights());
        print("output weights W2", output.weights());
        Matrix hiddenActivations = hidden.forward(inputs);
        print("hidden activations A1", hiddenActivations);
        print("predictions A2", network.predict(inputs));

        DenseLayer orNand = new DenseLayer(
            Matrix.of(new double[][] {{1, -1}, {1, -1}}),
            new Vector(new double[] {-0.5, 1.5}),
            new Step());
        DenseLayer and = new DenseLayer(
            Matrix.of(new double[][] {{1}, {1}}),
            new Vector(new double[] {-1.5}),
            new Step());
        Network xor = new Network(orNand, and);
        print("hand-built XOR predictions", xor.predict(inputs));
    }

    private static void print(String label, Matrix matrix) {
        System.out.println(label + " (" + matrix.rows() + " x " + matrix.cols() + "):");
        for (int row = 0; row < matrix.rows(); row++) {
            StringBuilder line = new StringBuilder("  [");
            for (int col = 0; col < matrix.cols(); col++) {
                if (col > 0) {
                    line.append(", ");
                }
                line.append(String.format(Locale.ROOT, "%8.4f", matrix.get(row, col)));
            }
            System.out.println(line.append(']'));
        }
    }
}
```

Expected output (the random values come from `java.util.Random`, whose algorithm is fixed by the Java specification):

**Output**

```
Random 2x3x1 network, seed 42
hidden weights W1 (2 x 3):
  [  0.7222,   0.5815,  -0.6007]
  [ -0.7001,   0.1777,   0.4330]
output weights W2 (3 x 1):
  [ -0.5779]
  [ -0.9876]
  [ -0.1350]
hidden activations A1 (4 x 3):
  [  0.5000,   0.5000,   0.5000]
  [  0.3318,   0.5443,   0.6066]
  [  0.6731,   0.6414,   0.3542]
  [  0.5055,   0.6812,   0.4582]
predictions A2 (4 x 1):
  [  0.2994]
  [  0.3076]
  [  0.2554]
  [  0.2637]
hand-built XOR predictions (4 x 1):
  [  0.0000]
  [  1.0000]
  [  1.0000]
  [  0.0000]
```

Read the first row of `A1`: input `(0, 0)` makes every pre-activation equal to its zero bias, and `σ(0) = 0.5`. The four random predictions all sit between 0.25 and 0.31: an untrained network has not learned anything about XOR. The hand-built network produces exactly `0, 1, 1, 0`.

## 7. Checks Without a Test Framework

These checks confirm the batched forward pass against hand-computed numbers, the XOR construction, reproducible initialization with a plausible spread, gradient application, and every validation rule.

**src/test/java/com/neuralscratch/nn/NetworkChecks.java**

```java
package com.neuralscratch.nn;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import java.util.Random;

public final class NetworkChecks {
    private static final double TOLERANCE = 1e-9;

    private NetworkChecks() {
    }

    public static void main(String[] args) {
        Matrix inputs = Matrix.of(new double[][] {{2, 3}, {1, 1}});
        DenseLayer linear = new DenseLayer(
            Matrix.of(new double[][] {{1, 0, 2}, {4, 5, 2}}),
            new Vector(new double[] {1, -1, 0}),
            new Identity());
        matrixClose(new double[][] {{15, 14, 10}, {6, 4, 4}}, linear.forward(inputs));
        check(linear.inputSize() == 2 && linear.outputSize() == 3, "layer sizes");

        DenseLayer squashed = new DenseLayer(linear.weights(), linear.biases(), new Sigmoid());
        Sigmoid sigmoid = new Sigmoid();
        matrixClose(new double[][] {
            {sigmoid.apply(15), sigmoid.apply(14), sigmoid.apply(10)},
            {sigmoid.apply(6), sigmoid.apply(4), sigmoid.apply(4)}
        }, squashed.forward(inputs));

        Network xor = new Network(
            new DenseLayer(Matrix.of(new double[][] {{1, -1}, {1, -1}}),
                new Vector(new double[] {-0.5, 1.5}), new Step()),
            new DenseLayer(Matrix.of(new double[][] {{1}, {1}}),
                new Vector(new double[] {-1.5}), new Step()));
        Matrix xorInputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        matrixClose(new double[][] {{0}, {1}, {1}, {0}}, xor.predict(xorInputs));

        DenseLayer first = DenseLayer.random(2, 3, new Sigmoid(), new Random(7));
        DenseLayer second = DenseLayer.random(2, 3, new Sigmoid(), new Random(7));
        matrixClose(toArray(first.weights()), second.weights());
        check(first.weights().rows() == 2 && first.weights().cols() == 3, "random shape");
        for (int col = 0; col < 3; col++) {
            close(0.0, first.biases().get(col));
        }

        DenseLayer wide = DenseLayer.random(200, 300, new Identity(), new Random(1));
        double sumSquares = 0.0;
        for (int row = 0; row < 200; row++) {
            for (int col = 0; col < 300; col++) {
                sumSquares += wide.weights().get(row, col) * wide.weights().get(row, col);
            }
        }
        double observedStddev = Math.sqrt(sumSquares / (200 * 300));
        double expectedStddev = Math.sqrt(2.0 / 500);
        check(Math.abs(observedStddev - expectedStddev) < 0.05 * expectedStddev,
            "Xavier stddev " + observedStddev);

        Random random = new Random(42);
        Network network = new Network(
            DenseLayer.random(2, 3, new Sigmoid(), random),
            DenseLayer.random(3, 1, new Sigmoid(), random));
        Matrix predictions = network.predict(xorInputs);
        check(predictions.rows() == 4 && predictions.cols() == 1, "2x3x1 output shape");
        check(network.inputSize() == 2 && network.outputSize() == 1, "network sizes");
        check(network.layers().size() == 2, "layer count");

        DenseLayer trainable = new DenseLayer(
            Matrix.of(new double[][] {{1, 2}, {3, 4}}),
            new Vector(new double[] {0.5, -0.5}),
            new Identity());
        trainable.applyGradients(
            Matrix.of(new double[][] {{10, 0}, {-10, 20}}),
            new Vector(new double[] {1, -1}),
            0.1);
        matrixClose(new double[][] {{0, 2}, {4, 2}}, trainable.weights());
        close(0.4, trainable.biases().get(0));
        close(-0.4, trainable.biases().get(1));

        Matrix goodGradient = Matrix.of(new double[][] {{1, 1}, {1, 1}});
        Vector goodBias = new Vector(new double[] {1, 1});
        expectIllegalArgument(() -> trainable.applyGradients(
            Matrix.of(new double[][] {{1, 1, 1}, {1, 1, 1}}), goodBias, 0.1));
        expectIllegalArgument(() -> trainable.applyGradients(
            goodGradient, new Vector(new double[] {1}), 0.1));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, 0.0));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, -1.0));
        expectIllegalArgument(() -> trainable.applyGradients(goodGradient, goodBias, Double.NaN));
        matrixClose(new double[][] {{0, 2}, {4, 2}}, trainable.weights());

        expectIllegalArgument(() -> new DenseLayer(
            Matrix.of(new double[][] {{1, 2}}), new Vector(new double[] {1}), new Identity()));
        expectIllegalArgument(() -> linear.forward(Matrix.of(new double[][] {{1, 2, 3}})));
        expectIllegalArgument(() -> DenseLayer.random(0, 3, new Sigmoid(), new Random(1)));
        expectIllegalArgument(() -> new Network());
        expectIllegalArgument(() -> new Network(
            DenseLayer.random(2, 3, new Sigmoid(), new Random(1)),
            DenseLayer.random(4, 1, new Sigmoid(), new Random(1))));
        expectIllegalArgument(() -> network.predict(Matrix.of(new double[][] {{1, 2, 3}})));
        try {
            network.layers().clear();
            throw new AssertionError("layers() must be unmodifiable");
        } catch (UnsupportedOperationException expected) {
            // The validated layer sequence cannot be changed.
        }
        System.out.println("All layer and network checks passed");
    }

    private static double[][] toArray(Matrix matrix) {
        double[][] values = new double[matrix.rows()][matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row][col] = matrix.get(row, col);
            }
        }
        return values;
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError("Check failed: " + message);
        }
    }

    private static void close(double expected, double actual) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > TOLERANCE) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void matrixClose(double[][] expected, Matrix actual) {
        if (expected.length != actual.rows() || expected[0].length != actual.cols()) {
            throw new AssertionError("Matrix shape mismatch");
        }
        for (int row = 0; row < actual.rows(); row++) {
            for (int col = 0; col < actual.cols(); col++) {
                close(expected[row][col], actual.get(row, col));
            }
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // Invalid shapes and rates are rejected before any state changes.
        }
    }
}
```

Compile all sources as in Chapter 3, then run:

**Terminal (project root, after regenerating sources.txt)**

```
javac -d out @sources.txt
java -cp out com.neuralscratch.examples.Chapter4Demo
java -cp out com.neuralscratch.nn.NetworkChecks
```

In PowerShell, quote the argument file as `"@sources.txt"`. Expected final line: `All layer and network checks passed`.

**Common shape mistake**

If you come from a textbook that writes `W · x` with `x` as a column, your weight matrix is the transpose of ours (`out × in`). Both conventions are correct; mixing them is not. In this book, inputs are rows and `W` is always `in × out`.

## What You Now Have

- `DenseLayer`: `in × out` weights, a bias per neuron, a batched forward pass `f(XW + b)`, seeded Xavier initialization, and a validated gradient-descent update.

- `Network`: dimension-checked layer chaining, an unmodifiable layer list, and batched prediction.

- A hand-built network that solves XOR, demonstrating why a hidden layer adds power.
**Before Chapter 5**

Pick one weight in `W₁` of the random network. Change it by `+0.01` and rerun `predict`. How much did each prediction move? Measuring that sensitivity for every weight, efficiently, is exactly what backpropagation does.

[← Previous: Chapter 3](chapter-03.md)

[Next: Chapter 5 →](chapter-05.md)
