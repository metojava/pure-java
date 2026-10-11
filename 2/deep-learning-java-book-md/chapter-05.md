**Chapter 5**

# Backpropagation & Gradient Descent

Measure how wrong the network is, trace the blame backward through every layer with the chain rule, and nudge each weight downhill.

## What We Are Building

Chapter 4's network can make predictions, but its weights are random. Learning means adjusting those weights so predictions move toward known answers. That needs three ingredients:

- A **loss**: one number saying how wrong the predictions are. We implement mean squared error.

- **Gradients**: for every weight and bias, the slope of the loss with respect to that parameter. **Backpropagation** computes all of them in one backward sweep.

- **Gradient descent**: move each parameter a small step against its slope. `DenseLayer.applyGradients` from Chapter 4 already does this.

New files: `com.neuralscratch.loss.OutputObjective`, `com.neuralscratch.loss.MeanSquaredError`, and `com.neuralscratch.training.Backpropagation`, plus a demo and a check program. We finish by confirming our hand-derived gradients against the numerical `Calculus.gradient` from Chapter 2.

## 1. Measuring Error: Mean Squared Error

For one prediction `a` and its target `y`, the error is `a − y`. Squaring makes it positive and punishes big misses more than small ones. We multiply by ½ so the derivative comes out clean, and we average over the batch so the loss does not grow just because the batch has more rows.

L = (1 / N) × Σrows n Σcolumns k ½ (ank − ynk)²

Here `N` is the number of rows (examples) in the batch. For outputs `[[1, 2], [3, 4]]` and targets `[[0, 2], [3, 2]]`, the differences are `1, 0, 0, 2`, the squares sum to 5, and `L = 0.5 × 5 / 2 = 1.25`.

### Its derivative with respect to the output

Each entry appears in exactly one squared term. The derivative of `½(a − y)²` with respect to `a` is `(a − y)`—the ½ cancels the 2 from the power rule—and the average contributes `1/N`:

∂L / ∂ank = (ank − ynk) / N

## 2. The `OutputObjective` Contract

Backpropagation starts at the output layer with a quantity called `δ` ("delta"): the slope of the loss with respect to the output layer's *pre-activation* `z`. By the chain rule, `δ = ∂L/∂a ⊙ f′(z)`, where `⊙` is the Hadamard (entry-by-entry) product from Chapter 2.

We let the loss object compute that starting delta instead of only `∂L/∂a`. Why? In Chapter 6, Softmax combined with cross-entropy has a starting delta of simply `a − y`, which is both simpler and more numerically stable than multiplying two separate pieces. Passing both the last layer's pre-activation (called `logits` here, the usual name for raw output scores) and its activation keeps that option open.

**src/main/java/com/neuralscratch/loss/OutputObjective.java**

```java
package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;

public interface OutputObjective {
    // L: one number summarising error over the batch (batch x outputs matrices).
    double loss(Matrix logits, Matrix output, Matrix targets);

    // delta = dL/dz for the last layer, shape batch x outputs.
    Matrix delta(Matrix logits, Matrix output, Matrix targets, Activation outputActivation);
}
```

**src/main/java/com/neuralscratch/loss/MeanSquaredError.java**

```java
package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class MeanSquaredError implements OutputObjective {
    @Override
    public double loss(Matrix logits, Matrix output, Matrix targets) {
        requireSameShapes(logits, output, targets);
        double sumSquares = 0.0;
        for (int row = 0; row < output.rows(); row++) {
            for (int col = 0; col < output.cols(); col++) {
                double error = output.get(row, col) - targets.get(row, col);
                sumSquares += error * error;
            }
        }
        // L = 0.5 * sum (a - y)^2 / N, with N = batch rows.
        return 0.5 * sumSquares / output.rows();
    }

    @Override
    public Matrix delta(Matrix logits, Matrix output, Matrix targets,
                        Activation outputActivation) {
        requireSameShapes(logits, output, targets);
        Objects.requireNonNull(outputActivation, "outputActivation");
        // dL/da = (a - y) / N; chain rule: dL/dz = dL/da (*) f'(z).
        return output.subtract(targets)
            .scale(1.0 / output.rows())
            .hadamard(logits.apply(outputActivation::derivative));
    }

    private static void requireSameShapes(Matrix logits, Matrix output, Matrix targets) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(targets, "targets");
        if (output.rows() != targets.rows() || output.cols() != targets.cols()
                || logits.rows() != output.rows() || logits.cols() != output.cols()) {
            throw new IllegalArgumentException(
                "logits, output and targets must share a shape; got "
                    + logits.rows() + "x" + logits.cols() + ", "
                    + output.rows() + "x" + output.cols() + ", "
                    + targets.rows() + "x" + targets.cols()
            );
        }
    }
}
```

## 3. The Chain Rule on One Neuron, With Numbers

Before matrices, follow one weight by hand. A single Sigmoid neuron has weight `w = 0.5` and bias `b = −0.5`. The input is `x = 2` and the target is `y = 1`.

### Forward pass

| Step | Formula | Value |
| --- | --- | --- |
| pre-activation | `z = w·x + b` | 0.5 × 2 − 0.5 = 0.5 |
| activation | `a = σ(z)` | 0.622459 |
| loss (N = 1) | `L = ½(a − y)²` | ½ × 0.377541² ≈ 0.071268 |

### Backward pass

The weight affects the loss only through a chain: `w → z → a → L`. The chain rule says the overall rate of change is the product of the local rates along the chain:

∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w

| Local slope | Formula | Value |
| --- | --- | --- |
| `∂L/∂a` | `a − y` | −0.377541 |
| `∂a/∂z` | `σ(z)(1 − σ(z))` | 0.235004 |
| `δ = ∂L/∂z` | product of the two above | −0.088723 |
| `∂z/∂w` | `x` | 2 |
| `∂z/∂b` | `1` | 1 |
| `∂L/∂w` | `δ × x` | −0.177447 |
| `∂L/∂b` | `δ × 1` | −0.088723 |

Both gradients are negative, meaning "increasing this parameter lowers the loss." That makes sense: the output 0.62 is below the target 1, and raising `w` or `b` raises `z` and therefore `a`. Notice how `δ` appears in both results. Computing it once and reusing it is the whole efficiency trick of backpropagation.

## 4. Gradient Descent: Walking Downhill

Picture the loss as a landscape whose coordinates are the parameters. The gradient points uphill—the direction of steepest increase. To reduce the loss, step the opposite way, scaled by a small positive number called the **learning rate** `η` (eta):

w ← w − η × ∂L/∂w      b ← b − η × ∂L/∂b

With `η = 1` in our example, `w` becomes `0.5 + 0.177 = 0.677` and `b` becomes `−0.5 + 0.089 = −0.411`. A rate that is too small learns slowly; one that is too large overshoots the valley and can make the loss grow. Chapter 7 explores choosing it and running many steps.

## 5. From One Neuron to Whole Layers

Now let every quantity be a matrix with one row per example. For layer `l`: input activations `Al−1` (with `A₀ = X`, the batch), weights `Wl`, pre-activation `Zl = Al−1Wl + bl`, output `Al = f(Zl)`, and `δl = ∂L/∂Zl`, which has the same shape as `Zl`.

**The four backpropagation equations**

- Output delta: `δL = ∂L/∂AL ⊙ f′(ZL)` (computed by `OutputObjective.delta`).

- Hidden delta: `δl = (δl+1 · Wl+1T) ⊙ f′(Zl)`.

- Weight gradient: `∂L/∂Wl = Al−1T · δl`.

- Bias gradient: `∂L/∂bl` = column sums of `δl` (add up the rows).

### Where equation 3 comes from

Entry `Z[n][j] = Σi A[n][i] · W[i][j] + b[j]`. Weight `W[i][j]` multiplies `A[n][i]` in that sum, so its local slope is `A[n][i]`, just like `∂z/∂w = x` for one neuron. The same weight is used by every example in the batch, so its total gradient adds up the contributions: `Σn A[n][i] · δ[n][j]`. That sum over `n` is precisely entry `(i, j)` of `AT · δ`. The `1/N` average is already inside `δ`, so summing gives the batch average.

### Where equation 4 comes from

Bias `b[j]` is added to `Z[n][j]` for every row `n` with slope 1, so its gradient is `Σn δ[n][j]`: the sum down column `j`.

### Where equation 2 comes from

Hidden activation `A[n][j]` feeds every neuron `k` of the next layer through weight `Wnext[j][k]`. Its share of the blame adds up over those paths: `∂L/∂A[n][j] = Σk δnext[n][k] · Wnext[j][k]`. Summing over the *second* index of `Wnext` is a multiplication by its transpose: `δnext · WnextT`. Then one more chain-rule link through the hidden activation multiplies entry-wise by `f′(Z)`.

### Shapes for a 2 × 3 × 1 network with a batch of 4

| Quantity | Computation | Shape |
| --- | --- | --- |
| `δ₂` | `(A₂ − Y)/4 ⊙ σ′(Z₂)` | 4 × 1 |
| `∂L/∂W₂` | `A₁ᵀ (3 × 4) · δ₂ (4 × 1)` | 3 × 1 = shape of `W₂` |
| `∂L/∂b₂` | column sums of `δ₂` | size 1 |
| `δ₁` | `δ₂ (4 × 1) · W₂ᵀ (1 × 3) ⊙ σ′(Z₁) (4 × 3)` | 4 × 3 |
| `∂L/∂W₁` | `Xᵀ (2 × 4) · δ₁ (4 × 3)` | 2 × 3 = shape of `W₁` |
| `∂L/∂b₁` | column sums of `δ₁` | size 3 |

Every gradient has exactly the shape of the parameter it updates. If your shapes do not line up, the formula is wrong; the `Matrix` class will tell you immediately.

## 6. Compute All Gradients First, Then Update

Equation 2 uses `Wl+1`, the weights of the layer *after* the one whose delta we are computing. If we updated the last layer's weights immediately after computing its gradient, the next backward step would use the *new* weights, yet the loss we are differentiating was produced by the *old* ones. The resulting hidden gradients would be subtly wrong: training might still drift downhill, which makes this bug hard to spot by watching the loss.

Our implementation therefore runs in three phases: a forward pass that caches every `Z` and `A`; a backward pass that stores every gradient in arrays; and only then an update loop calling `applyGradients`. The gradient check later in this chapter would catch the bug for networks with two or more layers.

## 7. The `Backpropagation` Class

`trainBatch` performs one gradient-descent step on one batch and returns the loss measured *before* the update. It validates everything up front, so invalid arguments never modify the network.

**src/main/java/com/neuralscratch/training/Backpropagation.java**

```java
package com.neuralscratch.training;

import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class Backpropagation {
    private Backpropagation() {
    }

    public static double trainBatch(Network network, Matrix inputs, Matrix targets,
                                    OutputObjective objective, double learningRate) {
        Objects.requireNonNull(network, "network");
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(objective, "objective");
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException(
                "Learning rate must be positive and finite, got " + learningRate
            );
        }
        List<DenseLayer> layers = network.layers();
        int layerCount = layers.size();
        DenseLayer last = layers.get(layerCount - 1);
        if (inputs.cols() != network.inputSize()) {
            throw new IllegalArgumentException(
                "Inputs need " + network.inputSize() + " columns, got " + inputs.cols()
            );
        }
        if (targets.rows() != inputs.rows() || targets.cols() != last.outputSize()) {
            throw new IllegalArgumentException(
                "Targets must be " + inputs.rows() + "x" + last.outputSize()
                    + ", got " + targets.rows() + "x" + targets.cols()
            );
        }

        // Forward pass: cache Z_l and A_l; activations.get(0) is the input batch A_0 = X.
        List<Matrix> preActivations = new ArrayList<>(layerCount);
        List<Matrix> activations = new ArrayList<>(layerCount + 1);
        activations.add(inputs);
        Matrix current = inputs;
        for (DenseLayer layer : layers) {
            // Z_l = A_(l-1) W_l + b_l ; A_l = f(Z_l)
            Matrix z = current.multiply(layer.weights()).addRowVector(layer.biases());
            current = z.apply(layer.activation()::apply);
            preActivations.add(z);
            activations.add(current);
        }
        Matrix lastPreActivation = preActivations.get(layerCount - 1);
        Matrix lastActivation = activations.get(layerCount);
        double loss = objective.loss(lastPreActivation, lastActivation, targets);

        // Backward pass: compute every gradient while all weights are still unchanged.
        Matrix[] weightGradients = new Matrix[layerCount];
        Vector[] biasGradients = new Vector[layerCount];
        // delta_L = dL/dZ_L, supplied by the objective.
        Matrix delta = objective.delta(lastPreActivation, lastActivation, targets,
            last.activation());
        for (int index = layerCount - 1; index >= 0; index--) {
            // dL/dW_l = A_(l-1)^T delta_l : (in x batch)(batch x out) = in x out
            weightGradients[index] = activations.get(index).transpose().multiply(delta);
            // dL/db_l = column sums of delta_l
            biasGradients[index] = rowSum(delta);
            if (index > 0) {
                DenseLayer hidden = layers.get(index - 1);
                Matrix nextWeights = layers.get(index).weights();
                // delta_(l-1) = (delta_l W_l^T) (*) f'(Z_(l-1))
                delta = delta.multiply(nextWeights.transpose())
                    .hadamard(preActivations.get(index - 1)
                        .apply(hidden.activation()::derivative));
            }
        }

        // Update phase: gradient descent on every layer.
        for (int index = 0; index < layerCount; index++) {
            layers.get(index).applyGradients(
                weightGradients[index], biasGradients[index], learningRate);
        }
        return loss;
    }

    public static Vector rowSum(Matrix matrix) {
        Objects.requireNonNull(matrix, "matrix");
        double[] sums = new double[matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                // Add every row together: one total per column.
                sums[col] += matrix.get(row, col);
            }
        }
        return new Vector(sums);
    }
}
```

The name `rowSum` means "sum of the rows": the result has one entry per column, which is exactly the bias shape. For `[[1, 2], [3, 4]]` it returns `[4, 6]`.

**Activations must have derivatives**

Training calls `derivative` on every layer's activation, so a `Step` layer throws `UnsupportedOperationException` here. That is deliberate: the hand-built XOR network from Chapter 4 can predict but not learn. Use `Sigmoid` or `Identity` for now; Chapter 6 adds ReLU and Softmax.

## 8. Trust, but Verify: Gradient Checking

Backpropagation code can contain bugs that still let the loss decrease a little. The reliable test is to compare its gradients with numerical estimates from Chapter 2's central difference:

∂L/∂θ ≈ [L(θ + h) − L(θ − h)] / (2h)

The procedure:

- **Flatten** every weight and bias of the network into one `Vector` of parameters `θ`. For 2 × 3 × 1 that is 13 numbers.

- Write a function that **rebuilds** a network from such a vector and returns its loss on a fixed batch. Pass it to `Calculus.gradient`.

- Get the **analytic** gradient from `trainBatch`: run it on a fresh copy with learning rate 1. Since `θnew = θ − 1 × gradient`, the gradient equals `θ − θnew`.

- Compare. With `h = 10⁻⁵` and smooth activations, the two typically agree to around `10⁻⁹`; we accept differences below `10⁻⁷`.

Recovering the gradient from an update costs only one floating-point rounding error per parameter and keeps the `Backpropagation` API exactly as specified. Remember that finite differences need two loss evaluations per parameter, so they are for checking, never for training.

## 9. A Runnable Demo

The demo has three parts: it reproduces the hand-computed chain-rule example, runs a gradient check on a seeded 2 × 3 × 1 network, and gives a short preview of repeated training steps on OR. Proper training loops with epochs, shuffling, and mini-batches arrive in Chapter 7.

**src/main/java/com/neuralscratch/examples/Chapter5Demo.java**

```java
package com.neuralscratch.examples;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import com.neuralscratch.training.Backpropagation;
import java.util.Locale;
import java.util.Random;

public final class Chapter5Demo {
    private static final int[] SIZES = {2, 3, 1};

    private Chapter5Demo() {
    }

    public static void main(String[] args) {
        OutputObjective mse = new MeanSquaredError();

        DenseLayer single = new DenseLayer(
            Matrix.of(new double[][] {{0.5}}), new Vector(new double[] {-0.5}), new Sigmoid());
        Network neuron = new Network(single);
        double weightBefore = single.weights().get(0, 0);
        double biasBefore = single.biases().get(0);
        double singleLoss = Backpropagation.trainBatch(neuron,
            Matrix.of(new double[][] {{2}}), Matrix.of(new double[][] {{1}}), mse, 1.0);
        System.out.printf(Locale.ROOT,
            "Chain rule example: loss = %.6f, dL/dw = %.6f, dL/db = %.6f%n",
            singleLoss,
            weightBefore - single.weights().get(0, 0),
            biasBefore - single.biases().get(0));
        System.out.printf(Locale.ROOT, "  after one step (rate 1): w = %.6f, b = %.6f%n",
            single.weights().get(0, 0), single.biases().get(0));

        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {1}, {1}, {1}});

        Vector parameters = flatten(randomNetwork(new Random(3)));
        Network copy = build(parameters);
        Backpropagation.trainBatch(copy, inputs, targets, mse, 1.0);
        // With rate 1: theta_new = theta - gradient, so gradient = theta - theta_new.
        Vector analytic = parameters.subtract(flatten(copy));
        Vector numeric = Calculus.gradient(
            theta -> loss(build(theta), inputs, targets, mse), parameters);
        double maxDifference = 0.0;
        System.out.println("Gradient check on a 2x3x1 network (seed 3):");
        for (int index = 0; index < parameters.size(); index++) {
            double difference = Math.abs(analytic.get(index) - numeric.get(index));
            maxDifference = Math.max(maxDifference, difference);
            System.out.printf(Locale.ROOT, "  parameter %2d  backprop %12.8f  numeric %12.8f%n",
                index, analytic.get(index), numeric.get(index));
        }
        System.out.printf(Locale.ROOT, "  max |difference| = %.2e -> %s%n",
            maxDifference, maxDifference < 1e-7 ? "PASS" : "FAIL");

        Network learner = randomNetwork(new Random(7));
        System.out.println("Preview: 2000 full-batch steps on OR, learning rate 2.0");
        for (int step = 1; step <= 2000; step++) {
            double stepLoss = Backpropagation.trainBatch(learner, inputs, targets, mse, 2.0);
            if (step == 1 || step % 500 == 0) {
                System.out.printf(Locale.ROOT, "  step %4d  loss %.6f%n", step, stepLoss);
            }
        }
        Matrix predictions = learner.predict(inputs);
        for (int row = 0; row < inputs.rows(); row++) {
            System.out.printf(Locale.ROOT, "  [%.0f, %.0f] -> %.4f (target %.0f)%n",
                inputs.get(row, 0), inputs.get(row, 1),
                predictions.get(row, 0), targets.get(row, 0));
        }
    }

    private static Network randomNetwork(Random random) {
        DenseLayer[] layers = new DenseLayer[SIZES.length - 1];
        for (int index = 0; index < layers.length; index++) {
            layers[index] = DenseLayer.random(SIZES[index], SIZES[index + 1],
                new Sigmoid(), random);
        }
        return new Network(layers);
    }

    private static Vector flatten(Network network) {
        int count = 0;
        for (DenseLayer layer : network.layers()) {
            count += layer.inputSize() * layer.outputSize() + layer.outputSize();
        }
        double[] theta = new double[count];
        int offset = 0;
        for (DenseLayer layer : network.layers()) {
            for (int row = 0; row < layer.inputSize(); row++) {
                for (int col = 0; col < layer.outputSize(); col++) {
                    theta[offset++] = layer.weights().get(row, col);
                }
            }
            for (int col = 0; col < layer.outputSize(); col++) {
                theta[offset++] = layer.biases().get(col);
            }
        }
        return new Vector(theta);
    }

    private static Network build(Vector theta) {
        DenseLayer[] layers = new DenseLayer[SIZES.length - 1];
        int offset = 0;
        for (int index = 0; index < layers.length; index++) {
            int in = SIZES[index];
            int out = SIZES[index + 1];
            double[][] weights = new double[in][out];
            for (int row = 0; row < in; row++) {
                for (int col = 0; col < out; col++) {
                    weights[row][col] = theta.get(offset++);
                }
            }
            double[] biases = new double[out];
            for (int col = 0; col < out; col++) {
                biases[col] = theta.get(offset++);
            }
            layers[index] = new DenseLayer(Matrix.of(weights), new Vector(biases), new Sigmoid());
        }
        return new Network(layers);
    }

    private static double loss(Network network, Matrix inputs, Matrix targets,
                               OutputObjective objective) {
        Matrix activation = inputs;
        Matrix z = null;
        for (DenseLayer layer : network.layers()) {
            z = activation.multiply(layer.weights()).addRowVector(layer.biases());
            activation = z.apply(layer.activation()::apply);
        }
        return objective.loss(z, activation, targets);
    }
}
```

Expected output:

**Output**

```
Chain rule example: loss = 0.071268, dL/dw = -0.177447, dL/db = -0.088723
  after one step (rate 1): w = 0.677447, b = -0.411277
Gradient check on a 2x3x1 network (seed 3):
  parameter  0  backprop   0.01514703  numeric   0.01514703
  parameter  1  backprop   0.02400653  numeric   0.02400653
  parameter  2  backprop   0.00472620  numeric   0.00472620
  parameter  3  backprop   0.01424935  numeric   0.01424935
  parameter  4  backprop   0.02319978  numeric   0.02319978
  parameter  5  backprop   0.00467232  numeric   0.00467232
  parameter  6  backprop   0.02034212  numeric   0.02034212
  parameter  7  backprop   0.03250912  numeric   0.03250912
  parameter  8  backprop   0.00665966  numeric   0.00665966
  parameter  9  backprop  -0.05213959  numeric  -0.05213959
  parameter 10  backprop  -0.04209927  numeric  -0.04209927
  parameter 11  backprop  -0.02479453  numeric  -0.02479453
  parameter 12  backprop  -0.08787631  numeric  -0.08787631
  max |difference| = 2.59e-12 -> PASS
Preview: 2000 full-batch steps on OR, learning rate 2.0
  step    1  loss 0.095636
  step  500  loss 0.002500
  step 1000  loss 0.000833
  step 1500  loss 0.000474
  step 2000  loss 0.000325
  [0, 0] -> 0.0405 (target 0)
  [0, 1] -> 0.9780 (target 1)
  [1, 0] -> 0.9783 (target 1)
  [1, 1] -> 0.9976 (target 1)
```

The first line matches the hand calculation in Section 3. Every analytic gradient agrees with its numerical estimate, and repeated steps drive the OR loss toward zero. The exact digits in the last decimal place of the numeric column may differ slightly between JVMs because `Math.exp` is allowed tiny platform differences; the PASS verdict should not change.

## 10. Checks Without a Test Framework

The checks verify the loss and delta formulas on small matrices, the chain-rule example, gradient checks on a two-layer and a three-layer network (either one exposes the update-order bug from Section 6), that the returned loss is the pre-update loss, that a small step lowers the loss, and that invalid arguments leave the network untouched.

**src/test/java/com/neuralscratch/training/BackpropagationChecks.java**

```java
package com.neuralscratch.training;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import java.util.Random;

public final class BackpropagationChecks {
    private static final double TOLERANCE = 1e-9;

    private BackpropagationChecks() {
    }

    public static void main(String[] args) {
        OutputObjective mse = new MeanSquaredError();
        Matrix output = Matrix.of(new double[][] {{1, 2}, {3, 4}});
        Matrix target = Matrix.of(new double[][] {{0, 2}, {3, 2}});
        close(1.25, mse.loss(output, output, target));
        matrixClose(new double[][] {{0.5, 0}, {0, 1}},
            mse.delta(output, output, target, new Identity()));
        Sigmoid sigmoid = new Sigmoid();
        Matrix logits = Matrix.of(new double[][] {{0.3, -1.2}});
        Matrix squashed = logits.apply(sigmoid::apply);
        Matrix goal = Matrix.of(new double[][] {{1, 0}});
        matrixClose(new double[][] {{
            (squashed.get(0, 0) - 1) * sigmoid.derivative(0.3),
            (squashed.get(0, 1) - 0) * sigmoid.derivative(-1.2)
        }}, mse.delta(logits, squashed, goal, sigmoid));
        expectIllegalArgument(() -> mse.loss(output, output, Matrix.of(new double[][] {{1, 2}})));

        Vector sums = Backpropagation.rowSum(output);
        close(4, sums.get(0));
        close(6, sums.get(1));

        DenseLayer single = new DenseLayer(
            Matrix.of(new double[][] {{0.5}}), new Vector(new double[] {-0.5}), sigmoid);
        double loss = Backpropagation.trainBatch(new Network(single),
            Matrix.of(new double[][] {{2}}), Matrix.of(new double[][] {{1}}), mse, 1.0);
        double a = sigmoid.apply(0.5);
        double delta = (a - 1) * a * (1 - a);
        close(0.5 * (a - 1) * (a - 1), loss);
        close(0.5 - 2 * delta, single.weights().get(0, 0));
        close(-0.5 - delta, single.biases().get(0));

        Random random = new Random(11);
        gradientCheck(new int[] {2, 3, 1}, randomMatrix(4, 2, random), randomMatrix(4, 1, random), random);
        gradientCheck(new int[] {3, 4, 3, 2}, randomMatrix(5, 3, random), randomMatrix(5, 2, random), random);

        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {1}, {1}, {1}});
        int[] sizes = {2, 3, 1};
        Network network = randomNetwork(sizes, new Random(5));
        double before = forwardLoss(network, inputs, targets, mse);
        double reported = Backpropagation.trainBatch(network, inputs, targets, mse, 0.5);
        close(before, reported);
        double after = forwardLoss(network, inputs, targets, mse);
        if (!(after < before)) {
            throw new AssertionError("A small step should lower the loss: " + before + " -> " + after);
        }

        Vector frozen = flatten(network);
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, Matrix.of(new double[][] {{1, 2, 3}}), Matrix.of(new double[][] {{1}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, inputs, Matrix.of(new double[][] {{1}, {0}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(
            network, inputs, Matrix.of(new double[][] {{1, 0}, {1, 0}, {1, 0}, {1, 0}}), mse, 0.1));
        expectIllegalArgument(() -> Backpropagation.trainBatch(network, inputs, targets, mse, 0.0));
        expectIllegalArgument(() -> Backpropagation.trainBatch(network, inputs, targets, mse, Double.NaN));
        vectorClose(frozen, flatten(network));
        System.out.println("All backpropagation checks passed");
    }

    private static void gradientCheck(int[] sizes, Matrix inputs, Matrix targets, Random random) {
        OutputObjective mse = new MeanSquaredError();
        Vector theta = flatten(randomNetwork(sizes, random));
        Network copy = build(sizes, theta);
        Backpropagation.trainBatch(copy, inputs, targets, mse, 1.0);
        Vector analytic = theta.subtract(flatten(copy));
        Vector numeric = Calculus.gradient(
            point -> forwardLoss(build(sizes, point), inputs, targets, mse), theta);
        for (int index = 0; index < theta.size(); index++) {
            if (Math.abs(analytic.get(index) - numeric.get(index)) > 1e-7) {
                throw new AssertionError("Gradient mismatch at parameter " + index + ": "
                    + analytic.get(index) + " vs " + numeric.get(index));
            }
        }
    }

    private static Matrix randomMatrix(int rows, int cols, Random random) {
        double[][] values = new double[rows][cols];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                values[row][col] = random.nextDouble();
            }
        }
        return Matrix.of(values);
    }

    private static Network randomNetwork(int[] sizes, Random random) {
        DenseLayer[] layers = new DenseLayer[sizes.length - 1];
        for (int index = 0; index < layers.length; index++) {
            layers[index] = DenseLayer.random(sizes[index], sizes[index + 1], new Sigmoid(), random);
        }
        return new Network(layers);
    }

    private static Vector flatten(Network network) {
        int count = 0;
        for (DenseLayer layer : network.layers()) {
            count += layer.inputSize() * layer.outputSize() + layer.outputSize();
        }
        double[] theta = new double[count];
        int offset = 0;
        for (DenseLayer layer : network.layers()) {
            for (int row = 0; row < layer.inputSize(); row++) {
                for (int col = 0; col < layer.outputSize(); col++) {
                    theta[offset++] = layer.weights().get(row, col);
                }
            }
            for (int col = 0; col < layer.outputSize(); col++) {
                theta[offset++] = layer.biases().get(col);
            }
        }
        return new Vector(theta);
    }

    private static Network build(int[] sizes, Vector theta) {
        DenseLayer[] layers = new DenseLayer[sizes.length - 1];
        int offset = 0;
        for (int index = 0; index < layers.length; index++) {
            double[][] weights = new double[sizes[index]][sizes[index + 1]];
            for (int row = 0; row < sizes[index]; row++) {
                for (int col = 0; col < sizes[index + 1]; col++) {
                    weights[row][col] = theta.get(offset++);
                }
            }
            double[] biases = new double[sizes[index + 1]];
            for (int col = 0; col < biases.length; col++) {
                biases[col] = theta.get(offset++);
            }
            layers[index] = new DenseLayer(Matrix.of(weights), new Vector(biases), new Sigmoid());
        }
        return new Network(layers);
    }

    private static double forwardLoss(Network network, Matrix inputs, Matrix targets,
                                      OutputObjective objective) {
        Matrix activation = inputs;
        Matrix z = null;
        for (DenseLayer layer : network.layers()) {
            z = activation.multiply(layer.weights()).addRowVector(layer.biases());
            activation = z.apply(layer.activation()::apply);
        }
        return objective.loss(z, activation, targets);
    }

    private static void close(double expected, double actual) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > TOLERANCE) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void vectorClose(Vector expected, Vector actual) {
        if (expected.size() != actual.size()) {
            throw new AssertionError("Vector size mismatch");
        }
        for (int index = 0; index < expected.size(); index++) {
            close(expected.get(index), actual.get(index));
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
            // Validation happens before any parameter changes.
        }
    }
}
```

Regenerate `sources.txt` (Chapter 3 shows the commands), then compile and run:

**Terminal (project root)**

```
javac -d out @sources.txt
java -cp out com.neuralscratch.examples.Chapter5Demo
java -cp out com.neuralscratch.training.BackpropagationChecks
```

Expected final line: `All backpropagation checks passed`. As an experiment, move the update loop inside the backward loop (update each layer right after computing its gradient) and rerun the checks: the gradient check reports a mismatch.

## What You Now Have

- `OutputObjective` and `MeanSquaredError`: a batch-averaged loss and the starting delta `(A − Y)/N ⊙ f′(Z)`.

- `Backpropagation.trainBatch`: cached forward pass, the four backpropagation equations, all gradients computed before any update, and a pre-update loss.

- A gradient check tying the analytic derivatives back to Chapter 2's numerical `Calculus.gradient`.
**Before Chapter 6**

Change the preview to learn XOR instead of OR. Depending on the seed and learning rate, it may learn quickly, slowly, or get stuck with Sigmoid's small slopes. Chapter 6 explains why and introduces ReLU, Softmax, and cross-entropy; Chapter 7 wraps `trainBatch` in a full training loop.

[← Previous: Chapter 4](chapter-04.md)

[Next: Chapter 6 →](chapter-06.md)
