**Chapter 6**

# Activation & Loss Functions

Replace hand-picked slopes with ReLU, and replace squared error with a numerically stable softmax cross-entropy.

## What We Are Building

Chapters 3–5 gave us a working learner: layers compute `z = a·W + b`, an `Activation` turns `z` into the layer output, a `MeanSquaredError` objective scores the final output, and `Backpropagation.trainBatch` pushes that score backward. This chapter upgrades the two pieces that decide *what* the network can learn and *how well the learning signal survives*: the activation function and the loss function.

You will add three new source files and one executable check, all in plain Java 17 with no libraries:

| File | Role |
| --- | --- |
| `activation/ReLU.java` | An elementwise `Activation` that is cheap and does not saturate on the right. |
| `activation/Softmax.java` | A stable row-wise utility that converts scores into probabilities. It is *not* an `Activation`. |
| `loss/SoftmaxCrossEntropy.java` | An `OutputObjective` for classification, with the fused gradient `(p − t) / batch`. |
| `loss/ActivationLossChecks.java` | A dependency-free check that compares every formula with a finite-difference slope from Chapter 2's `Calculus`. |

**The contract this chapter plugs into**

Batches are row-major: one row per sample, one column per feature or output. `Activation` has `apply(double z)` and `derivative(double z)`, and `derivative` always receives the *pre-activation* `z`. `OutputObjective` has `loss(logits, output, targets)` and `delta(logits, output, targets, outputActivation)`; `delta` returns `∂L/∂z` for the output layer, already divided by the batch size. The loss functions already existing in your project (`MeanSquaredError`) and the ones we add here are interchangeable behind that interface, so `Backpropagation` needs no edits.

## 1. ReLU: A Slope That Does Not Fade

Chapter 3's sigmoid squeezes every `z` into the range 0 to 1. Its derivative `σ(z)(1 − σ(z))` never exceeds `0.25` and approaches zero when `|z|` is large. In a deep stack, backpropagation multiplies one such derivative per layer, so the learning signal shrinks quickly. This is the **vanishing gradient** problem.

The **rectified linear unit** is almost embarrassingly simple:

ReLU(z) = max(0, z)      ReLU′(z) = 1 if z > 0, otherwise 0

For positive `z` the output equals the input and the slope is exactly 1, so the signal passes through unchanged. For negative `z` the output and slope are both 0. The function is bent at `z = 0` like a hockey stick, so it has no single slope there. Any value between 0 and 1 would be a valid *subgradient*; we choose **0**. That choice is deliberate and tested.

**Dead ReLU units**

If a neuron's `z` is negative for every training example, its derivative is 0 everywhere, so its weights receive no gradient and it can stay silent forever. Small learning rates and sensible initial weights reduce this risk; the Chapter 8 MNIST run uses ReLU in the hidden layer and reports measured results instead of assuming it works.

**src/main/java/com/neuralscratch/activation/ReLU.java**

```java
package com.neuralscratch.activation;

public final class ReLU implements Activation {
    @Override
    public double apply(double z) {
        // ReLU(z) = max(0, z). Math.max also passes NaN through instead of hiding it.
        return Math.max(0.0, z);
    }

    @Override
    public double derivative(double z) {
        // Slope is 1 on the right of zero and 0 on the left.
        // The kink at exactly zero has no single slope; we choose 0.
        return z > 0.0 ? 1.0 : 0.0;
    }
}
```

`Math.max(0.0, z)` was chosen over `z > 0 ? z : 0` because `Math.max` propagates `NaN`. A silently-zeroed `NaN` would hide an earlier numerical failure.

## 2. Softmax: Turning Scores into Probabilities

For a ten-way classifier the last layer produces ten real numbers per sample. These raw, unnormalized scores are called **logits**. We want probabilities: each between 0 and 1, and summing to 1 across the row. The **softmax** function does that:

pk = exp(zk) / Σj exp(zj)

For logits `[1, 2, 3]`: `e¹ = 2.718`, `e² = 7.389`, `e³ = 20.086`; their sum is `30.193`; so the probabilities are about `[0.0900, 0.2447, 0.6652]`.

### Why softmax is not an `Activation`

Our `Activation` interface maps one double to one double. Softmax cannot: `pk` depends on *every* logit in the row, because of the denominator. Its derivative is therefore a whole matrix, the **Jacobian** `∂pi/∂zj`:

∂pi/∂zj = pi (δij − pj)       (δij = 1 when i = j, else 0)

A single `derivative(double z)` has nowhere to put that matrix. So `Softmax` is a standalone utility, and the clean way to use it in training is to fuse it with cross-entropy, as we do in section 4.

### The overflow trap and the max-subtraction fix

Compute `Math.exp(1000)` in Java and you get `Infinity`; dividing `Infinity` by `Infinity` gives `NaN`. Yet softmax of `[1000, 1001, 1002]` is perfectly well defined. The fix is an algebraic identity: multiplying the numerator and denominator by the same constant `exp(−m)` changes nothing.

exp(zk) / Σj exp(zj) = exp(zk − m) / Σj exp(zj − m)

Choose `m` as the row maximum. Every exponent is now at most 0, every `exp` is at most 1, and the largest term is exactly 1, so the sum is at least 1 and the division is safe. Very negative entries may underflow to 0, which is the correct rounded answer. The same identity gives the **log-sum-exp** used by the loss: `log Σ exp(zj) = m + log Σ exp(zj − m)`.

**src/main/java/com/neuralscratch/activation/Softmax.java**

```java
package com.neuralscratch.activation;

import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class Softmax {
    private Softmax() {
    }

    public static Matrix apply(Matrix logits) {
        requireFinite(logits);
        double[][] probabilities = new double[logits.rows()][logits.cols()];
        for (int row = 0; row < logits.rows(); row++) {
            double max = rowMax(logits, row);
            double sum = 0.0;
            for (int col = 0; col < logits.cols(); col++) {
                // exp(z - max) is at most 1, so it cannot overflow. The largest entry gives exactly 1.
                double shifted = Math.exp(logits.get(row, col) - max);
                probabilities[row][col] = shifted;
                sum += shifted;
            }
            for (int col = 0; col < logits.cols(); col++) {
                // p_k = exp(z_k - max) / sum_j exp(z_j - max); the common factor exp(-max) cancels.
                probabilities[row][col] /= sum;
            }
        }
        return Matrix.of(probabilities);
    }

    public static double[] logSumExp(Matrix logits) {
        requireFinite(logits);
        double[] result = new double[logits.rows()];
        for (int row = 0; row < logits.rows(); row++) {
            double max = rowMax(logits, row);
            double sum = 0.0;
            for (int col = 0; col < logits.cols(); col++) {
                sum += Math.exp(logits.get(row, col) - max);
            }
            // log(sum_j exp(z_j)) = max + log(sum_j exp(z_j - max)); sum is at least 1.
            result[row] = max + Math.log(sum);
        }
        return result;
    }

    private static double rowMax(Matrix logits, int row) {
        double max = logits.get(row, 0);
        for (int col = 1; col < logits.cols(); col++) {
            max = Math.max(max, logits.get(row, col));
        }
        return max;
    }

    private static void requireFinite(Matrix logits) {
        Objects.requireNonNull(logits, "logits");
        for (int row = 0; row < logits.rows(); row++) {
            for (int col = 0; col < logits.cols(); col++) {
                if (!Double.isFinite(logits.get(row, col))) {
                    throw new IllegalArgumentException(
                        "Logit at row " + row + ", column " + col + " is not finite"
                    );
                }
            }
        }
    }
}
```

Both methods reject `NaN` and infinite logits, and both process one row at a time, so a large score in one sample never affects another sample's probabilities.

## 3. Mean Squared Error, and Where It Struggles

In Chapter 5 you built `MeanSquaredError`. Per sample it measures half the sum of squared errors between output `a` and target `t`, then averages over the batch of `B` rows:

L = (1 / B) · Σsamples ½ Σk (ak − tk)²

The `½` cancels the 2 from differentiating the square. The output-layer delta is `(a − t) ⊙ f′(z) / B`, where `⊙` is the elementwise Hadamard product and `f′` is the output activation's derivative evaluated at `z`. Squared error is natural for predicting a number such as a temperature or a price. For classification it has a weakness that appears through the factor `f′(z)`.

Imagine one sigmoid output with target 1, and the network is confidently wrong: `z = −10`. Then `a = σ(−10) ≈ 0.0000454` and `f′(z) ≈ 0.0000454`. The delta is `(a − 1) · f′(z) ≈ −0.0000454`. The loss is nearly at its maximum for this sample, yet the learning signal is almost zero. The sigmoid's flat tail has multiplied the useful error by a tiny slope.

## 4. Softmax Cross-Entropy: A Loss Built for Probabilities

**Cross-entropy** asks: how surprised is the model by the correct answer? If the model assigns probability `pc` to the true class `c`, the surprise is `−log pc`. This is 0 when `pc = 1` and grows without bound as `pc` approaches 0. For a target *distribution* `t` (a one-hot vector is the special case with a single 1), the sample loss is:

Lsample = − Σk tk log pk       L = (1 / B) Σsamples Lsample

### Computing it from logits

Taking `log` of a probability that has underflowed to 0 produces `−Infinity`. Instead, substitute `log pk = zk − logSumExp(z)`. Because the target row sums to 1:

Lsample = Σk tk · (logSumExp(z) − zk)

No probability is ever formed, so a confidently wrong logit row such as `[1000, 0]` with true class 1 gives exactly the sensible loss 1000 instead of infinity. The class below checks that every target is in `[0, 1]` and each row sums to 1 within `1e-6`; this also makes it work for soft labels such as `[0.5, 0.5, 0]`.

### The fused gradient, derived

Apply the chain rule through the softmax Jacobian. For one sample:

∂L/∂zj = Σi (−ti / pi) · pi (δij − pj) = −tj + pj Σi ti = pj − tj

The `pi` in the numerator and denominator cancel, and `Σ ti = 1` removes the leftover sum. The whole Jacobian collapses into `p − t`. With the batch average included:

delta = (softmax(z) − t) / B

For logits `[1, 2, 3]` and one-hot target `[0, 0, 1]`, the gradient is `[0.0900, 0.2447, −0.3348]` for `B = 1`. It pushes the true logit up and the others down, with force proportional to the mistake. Unlike the MSE example, there is no tiny slope factor: a confidently wrong prediction gets a gradient close to full strength.

**Why the Identity requirement**

The formula `(p − t) / B` is the derivative with respect to the *logits*, which only equals the layer's `z` when the output activation is `Identity` (derivative 1). If you pass a different output activation, the correct delta would need an extra `f′(z)` factor, and the loss would no longer mean "softmax of the layer outputs". `delta` therefore throws `IllegalArgumentException` for any output activation that is not an `Identity`, instead of returning a wrong gradient quietly.

**src/main/java/com/neuralscratch/loss/SoftmaxCrossEntropy.java**

```java
package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Softmax;
import com.neuralscratch.math.Matrix;
import java.util.Objects;

public final class SoftmaxCrossEntropy implements OutputObjective {
    private static final double ROW_SUM_TOLERANCE = 1e-6;

    @Override
    public double loss(Matrix logits, Matrix output, Matrix targets) {
        // The output argument is intentionally unused: the loss is computed from the raw logits.
        requireCompatible(logits, targets);
        double[] logSumExp = Softmax.logSumExp(logits);
        double total = 0.0;
        for (int row = 0; row < logits.rows(); row++) {
            for (int col = 0; col < logits.cols(); col++) {
                // -log p_k = logSumExp(z) - z_k, so L = sum_k t_k * (logSumExp(z) - z_k).
                total += targets.get(row, col) * (logSumExp[row] - logits.get(row, col));
            }
        }
        return total / logits.rows();
    }

    @Override
    public Matrix delta(Matrix logits, Matrix output, Matrix targets, Activation outputActivation) {
        if (!(outputActivation instanceof Identity)) {
            throw new IllegalArgumentException(
                "SoftmaxCrossEntropy needs an Identity output layer so that z are logits"
            );
        }
        requireCompatible(logits, targets);
        // dL/dz = (softmax(z) - t) / batchRows. The Identity derivative is 1, so this is already delta.
        return Softmax.apply(logits).subtract(targets).scale(1.0 / logits.rows());
    }

    private static void requireCompatible(Matrix logits, Matrix targets) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(targets, "targets");
        if (logits.rows() != targets.rows() || logits.cols() != targets.cols()) {
            throw new IllegalArgumentException(
                "Logits are " + logits.rows() + "x" + logits.cols() + " but targets are "
                    + targets.rows() + "x" + targets.cols()
            );
        }
        for (int row = 0; row < targets.rows(); row++) {
            double sum = 0.0;
            for (int col = 0; col < targets.cols(); col++) {
                double target = targets.get(row, col);
                if (!Double.isFinite(target) || target < 0.0 || target > 1.0) {
                    throw new IllegalArgumentException(
                        "Target at row " + row + ", column " + col + " must be in [0, 1]"
                    );
                }
                sum += target;
            }
            if (Math.abs(sum - 1.0) > ROW_SUM_TOLERANCE) {
                throw new IllegalArgumentException(
                    "Target row " + row + " must sum to 1 but sums to " + sum
                );
            }
        }
    }
}
```

Note that the `output` argument is unused by `loss`: the loss is derived from logits. For an `Identity` layer, `output` and `logits` are the same matrix anyway. Because the interface requires the parameter, we accept it and say clearly that we ignore it.

## 5. MSE Versus Cross-Entropy at a Glance

| | Mean squared error | Softmax cross-entropy |
| --- | --- | --- |
| Natural task | Predicting numbers; small demos such as XOR | Choosing one of several classes |
| Output layer | Any activation | `Identity` logits only |
| Output delta | `(a − t) ⊙ f′(z) / B` | `(softmax(z) − t) / B` |
| Confident mistake with sigmoid | Gradient shrinks with `f′(z)` | Gradient stays large |
| Numerical care | Plain subtraction | Max subtraction and log-sum-exp |

Neither is "the correct loss"; each is a modeling choice. Chapter 8 uses MSE for the tiny XOR network and cross-entropy for ten-class MNIST.

## 6. Trust, but Verify: Finite-Difference Checks

A formula that looks right can still hide a sign error or a missing `1/B`. Chapter 2's `Calculus.gradient` measures the slope of the loss numerically by nudging each input up and down. If our analytic `delta` matches that independent measurement for every cell of a small matrix, we have strong evidence it is correct.

The check below flattens a logits matrix into a `Vector`, asks `Calculus.gradient` for the slope of `loss` with respect to every entry, and compares it with `delta`. It does the same for `MeanSquaredError` with a sigmoid output, which also confirms the delta contract from Chapter 5. It exercises ReLU values (including the zero derivative at zero), softmax shift invariance, overflow-safe large logits, known loss values, soft labels, and every validation error.

**Assumption about earlier chapters**

The check instantiates `new MeanSquaredError()`, `new Sigmoid()` and `new Identity()` with no-argument constructors, following the agreed contract. If your Chapter 3 or 5 code uses a different construction style, adjust only those three expressions.

**src/test/java/com/neuralscratch/loss/ActivationLossChecks.java**

```java
package com.neuralscratch.loss;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.ReLU;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Softmax;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;

public final class ActivationLossChecks {
    private static final double TOLERANCE = 1e-9;

    private ActivationLossChecks() {
    }

    public static void main(String[] args) {
        checkRelu();
        checkSoftmax();
        checkCrossEntropyValues();
        checkCrossEntropyGradient();
        checkMeanSquaredErrorGradient();
        checkSaturation();
        checkValidation();
        System.out.println("All activation and loss checks passed");
    }

    private static void checkRelu() {
        ReLU relu = new ReLU();
        close(0.0, relu.apply(-3.5), TOLERANCE);
        close(0.0, relu.apply(0.0), TOLERANCE);
        close(2.5, relu.apply(2.5), TOLERANCE);
        close(0.0, relu.derivative(-3.5), TOLERANCE);
        close(0.0, relu.derivative(0.0), TOLERANCE);
        close(1.0, relu.derivative(2.5), TOLERANCE);
        close(1.0, Calculus.derivative(relu::apply, 2.5), 1e-6);
        close(0.0, Calculus.derivative(relu::apply, -2.5), 1e-6);
    }

    private static void checkSoftmax() {
        Matrix logits = Matrix.of(new double[][] {{1, 2, 3}, {0, 0, 0}});
        Matrix probabilities = Softmax.apply(logits);
        close(0.09003057317038046, probabilities.get(0, 0), 1e-12);
        close(0.24472847105479764, probabilities.get(0, 1), 1e-12);
        close(0.6652409557748219, probabilities.get(0, 2), 1e-12);
        close(1.0 / 3.0, probabilities.get(1, 1), 1e-12);
        for (int row = 0; row < probabilities.rows(); row++) {
            double sum = 0.0;
            for (int col = 0; col < probabilities.cols(); col++) {
                sum += probabilities.get(row, col);
            }
            close(1.0, sum, 1e-12);
        }

        Matrix shifted = Softmax.apply(Matrix.of(new double[][] {{1001, 1002, 1003}}));
        Matrix huge = Softmax.apply(Matrix.of(new double[][] {{1000, 1001, 1002}}));
        for (int col = 0; col < 3; col++) {
            close(probabilities.get(0, col), shifted.get(0, col), 1e-12);
            close(probabilities.get(0, col), huge.get(0, col), 1e-12);
        }
        if (Double.isFinite(Math.exp(1000.0))) {
            throw new AssertionError("The naive exponential should overflow here");
        }

        expectIllegalArgument(() -> Softmax.apply(Matrix.of(new double[][] {{1, Double.NaN}})));
        expectIllegalArgument(() -> Softmax.apply(
            Matrix.of(new double[][] {{Double.POSITIVE_INFINITY, 1}})));
    }

    private static void checkCrossEntropyValues() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix uniform = Matrix.of(new double[][] {{0, 0}});
        Matrix classZero = Matrix.of(new double[][] {{1, 0}});
        close(Math.log(2.0), objective.loss(uniform, uniform, classZero), 1e-12);

        Matrix logits = Matrix.of(new double[][] {{2.0, 1.0, 0.1}});
        double denominator = Math.exp(2.0) + Math.exp(1.0) + Math.exp(0.1);
        double expected = -Math.log(Math.exp(2.0) / denominator);
        Matrix oneHot = Matrix.of(new double[][] {{1, 0, 0}});
        close(expected, objective.loss(logits, logits, oneHot), 1e-12);

        Matrix soft = Matrix.of(new double[][] {{0.5, 0.5, 0.0}});
        double distributionLoss = -0.5 * Math.log(Math.exp(2.0) / denominator)
            - 0.5 * Math.log(Math.exp(1.0) / denominator);
        close(distributionLoss, objective.loss(logits, logits, soft), 1e-12);

        Matrix confidentlyWrong = Matrix.of(new double[][] {{1000, 0}});
        Matrix classOne = Matrix.of(new double[][] {{0, 1}});
        close(1000.0, objective.loss(confidentlyWrong, confidentlyWrong, classOne), 1e-9);
    }

    private static void checkCrossEntropyGradient() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{0.3, -1.2, 2.0}, {1.5, 0.2, -0.7}});
        Matrix targets = Matrix.of(new double[][] {{0, 0, 1}, {0.25, 0.75, 0}});
        Matrix delta = objective.delta(logits, logits, targets, new Identity());

        Vector numeric = Calculus.gradient(
            flat -> {
                Matrix z = unflatten(flat, 2, 3);
                return objective.loss(z, z, targets);
            },
            flatten(logits)
        );
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 3; col++) {
                close(numeric.get(row * 3 + col), delta.get(row, col), 1e-7);
            }
        }
        Matrix probabilities = Softmax.apply(logits);
        close((probabilities.get(0, 0) - 0.0) / 2.0, delta.get(0, 0), TOLERANCE);
    }

    private static void checkMeanSquaredErrorGradient() {
        MeanSquaredError objective = new MeanSquaredError();
        Activation sigmoid = new Sigmoid();
        Matrix logits = Matrix.of(new double[][] {{0.4, -0.9}, {1.1, 0.3}});
        Matrix targets = Matrix.of(new double[][] {{1, 0}, {0, 1}});
        Matrix output = logits.apply(sigmoid::apply);
        Matrix delta = objective.delta(logits, output, targets, sigmoid);

        Vector numeric = Calculus.gradient(
            flat -> {
                Matrix z = unflatten(flat, 2, 2);
                return objective.loss(z, z.apply(sigmoid::apply), targets);
            },
            flatten(logits)
        );
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 2; col++) {
                close(numeric.get(row * 2 + col), delta.get(row, col), 1e-7);
            }
        }
    }

    private static void checkSaturation() {
        Activation sigmoid = new Sigmoid();
        MeanSquaredError mse = new MeanSquaredError();
        Matrix z = Matrix.of(new double[][] {{-10.0}});
        Matrix target = Matrix.of(new double[][] {{1.0}});
        double mseSlope = mse.delta(z, z.apply(sigmoid::apply), target, sigmoid).get(0, 0);

        SoftmaxCrossEntropy crossEntropy = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{10.0, 0.0}});
        Matrix wrongTarget = Matrix.of(new double[][] {{0.0, 1.0}});
        double ceSlope = crossEntropy.delta(logits, logits, wrongTarget, new Identity()).get(0, 0);
        if (Math.abs(mseSlope) > 1e-4 || Math.abs(ceSlope) < 0.9) {
            throw new AssertionError("Expected tiny MSE slope and large cross-entropy slope");
        }
        System.out.println("Confidently wrong, MSE with sigmoid: |dL/dz| = " + Math.abs(mseSlope));
        System.out.println("Confidently wrong, softmax cross-entropy: |dL/dz| = " + Math.abs(ceSlope));
    }

    private static void checkValidation() {
        SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();
        Matrix logits = Matrix.of(new double[][] {{1, 2}});
        Matrix good = Matrix.of(new double[][] {{1, 0}});
        expectIllegalArgument(() -> objective.delta(logits, logits, good, new Sigmoid()));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 0, 0}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 1}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{-0.5, 1.5}})));
        expectIllegalArgument(() -> objective.loss(logits, logits,
            Matrix.of(new double[][] {{1, 0}, {0, 1}})));
    }

    private static Vector flatten(Matrix matrix) {
        double[] values = new double[matrix.rows() * matrix.cols()];
        for (int row = 0; row < matrix.rows(); row++) {
            for (int col = 0; col < matrix.cols(); col++) {
                values[row * matrix.cols() + col] = matrix.get(row, col);
            }
        }
        return new Vector(values);
    }

    private static Matrix unflatten(Vector vector, int rows, int cols) {
        double[][] values = new double[rows][cols];
        for (int index = 0; index < vector.size(); index++) {
            values[index / cols][index % cols] = vector.get(index);
        }
        return Matrix.of(values);
    }

    private static void close(double expected, double actual, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > tolerance) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void expectIllegalArgument(Runnable operation) {
        try {
            operation.run();
            throw new AssertionError("Expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // The invalid input was rejected before any loss was computed.
        }
    }
}
```

From the project root, compile every source file and run the check. These commands need only the JDK:

**PowerShell (project root)**

```
New-Item -ItemType Directory -Force out | Out-Null
javac -d out (Get-ChildItem -Recurse -Path src -Filter *.java).FullName
java -cp out com.neuralscratch.loss.ActivationLossChecks
```

**macOS / Linux shell (project root)**

```
mkdir -p out
javac -d out $(find src -name "*.java")
java -cp out com.neuralscratch.loss.ActivationLossChecks
```

The checks print two diagnostic lines for the confidently-wrong example and then a pass message. Values from one run against a reference implementation of the Chapter 3–5 contract:

**Output**

```
Confidently wrong, MSE with sigmoid: |dL/dz| = 4.5393746863032435E-5
Confidently wrong, softmax cross-entropy: |dL/dz| = 0.9999546021312976
All activation and loss checks passed
```

The two numbers compare different models (one sigmoid output versus a two-logit softmax), so they illustrate the *shape* of each gradient rather than a head-to-head benchmark. The last digits can differ between Java versions.

## 7. How These Pieces Fit Together

Nothing in `Backpropagation` changes. A classifier is assembled like this, which Chapter 8 turns into a full program:

**Preview: a 784 → 64 → 10 classifier**

```
Random random = new Random(7);
Network network = new Network(
    DenseLayer.random(784, 64, new ReLU(), random),
    DenseLayer.random(64, 10, new Identity(), random)); // Identity: outputs are logits
SoftmaxCrossEntropy objective = new SoftmaxCrossEntropy();

double loss = Backpropagation.trainBatch(network, inputs, targets, objective, 0.05);
Matrix probabilities = Softmax.apply(network.predict(inputs));
```

During training, the network's last layer outputs logits. Probabilities exist only when you want to read a prediction, via `Softmax.apply`. During backpropagation, `trainBatch` asks the objective for the output delta, multiplies it back through the weights using the hidden layers' `derivative` methods (now including ReLU's), and updates every layer.

## What You Now Have

- `ReLU`: `max(0, z)` with derivative 1 for positive `z` and 0 otherwise, including zero.

- `Softmax`: row-wise probabilities and log-sum-exp, stable for logits in the thousands, rejecting non-finite input.

- `SoftmaxCrossEntropy`: a loss computed from logits, a fused `(p − t) / B` gradient, shape and target validation, and support for soft target distributions.

- `ActivationLossChecks`: numerical-gradient verification for both the new loss and the existing MSE.
**Before Chapter 7**

Pick logits `[0, 0, 0]` and target `[1, 0, 0]` and compute the loss and gradient by hand. You should get `log 3 ≈ 1.0986` and `[−0.6667, 0.3333, 0.3333]`. Then run the checks and change one sign in `delta` to watch them fail.

Next we stop calling `trainBatch` by hand. We will wrap it in a `Dataset` and a `Trainer` that shuffle samples, slice batches, and report the loss for every epoch.

[← Previous: Chapter 5](chapter-05.md)

[Next: Chapter 7 →](chapter-07.md)
