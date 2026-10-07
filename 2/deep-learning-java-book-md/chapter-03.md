
                "
"
                    "
"Chapter 3"
"

# The Architecture of a Neuron

Weights, a bias, and an activation function: build one artificial neuron, make it compute OR, and prove why it can never compute XOR.

                "
"

                "
"

## What We Are Building

Chapter 2 gave us `Vector`, `Matrix`, and `Calculus`. In this chapter we use them to build the smallest thinking unit of a network: a **neuron**. A neuron takes a list of numbers, combines them into one number, and squashes that number through an **activation function**.

You will add five source files to the project:

                    "
"

- `com.neuralscratch.activation.Activation`: the contract every activation function follows.

- `Step`, `Sigmoid`, and `Identity`: three concrete activations.

- `com.neuralscratch.nn.Neuron`: weights, a bias, and an activation, with shape validation.
                    "
"

Then you will run a demo that computes the logical OR function, searches thousands of parameter settings for XOR without success, and a check program that tests every class using only Java SE.

                    "
"
                        "
"Project layout so far"
"

Production code lives under `src/main/java`, checks under `src/test/java`, and compiled classes go into `out`. Keep the three Chapter 2 math files exactly as they were; this chapter only adds files.

                    "
"
                "
"

                "
"

## 1. The Weighted Sum: How a Neuron Listens

Imagine deciding whether to take an umbrella. You look at two clues: *is it cloudy?* (`x₀`) and *did the forecast say rain?* (`x₁`). Each clue matters by a different amount, so you give each one a **weight**. You also have a general tendency to carry an umbrella or not; that tendency is the **bias**. Add everything up and you get the neuron's **pre-activation**, written `z`:

                    "
"z = w₀x₀ + w₁x₁ + … + wₙ₋₁xₙ₋₁ + b = w · x + b"
"

The first part is exactly the dot product from Chapter 2. With weights `[0.8, 1.5]`, bias `−1.0`, and input `[1, 1]`, we get `z = 0.8×1 + 1.5×1 − 1.0 = 1.3`.

                    "
"
                        "
""
"Shapes for one neuron"
"Copy"
"

```
input   x : Vector of size n      [1.0, 1.0]
weights w : Vector of size n      [0.8, 1.5]
bias    b : one double            -1.0
z = w · x + b : one double        1.3
output a = activation(z) : double
```

                    "
"

The input and weight vectors *must* have the same size. If a neuron expects two clues and receives three, there is no sensible answer, so our `Neuron` will refuse the input with an `IllegalArgumentException` instead of quietly producing a number.

### Why have a bias at all?

Without a bias, an all-zero input always gives `z = 0`, no matter what the weights are. The bias lets the neuron shift its decision threshold. Geometrically, the weights choose the *direction* of a dividing line and the bias moves that line away from the origin.

                "
"

                "
"

## 2. Activation Functions: Turning a Score into an Output

The pre-activation `z` can be any real number. An activation function `f` maps it to the neuron's output `a = f(z)`. Different activations suit different jobs: a hard yes/no, a smooth probability-like value, or the raw score unchanged.

Learning (Chapter 5) also needs each activation's **derivative** `f′(z)`: how much the output changes when `z` changes a tiny bit. So our interface has two methods.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/activation/Activation.java"
"Copy"
"

```java
package com.neuralscratch.activation;

public interface Activation {
    // a = f(z): the neuron's output for pre-activation z.
    double apply(double z);

    // f'(z): the local slope used by the chain rule during backpropagation.
    double derivative(double z);
}
```

                    "
"
                    "
"
                        "
"Why not @FunctionalInterface?"
"

A functional interface has exactly one abstract method, so a lambda can implement it. An activation is a *pair* of functions that must agree with each other; a lambda could only supply one of them. We therefore write small named classes. When we need just the forward function as a `DoubleUnaryOperator` (for `Matrix.apply`), we pass a method reference such as `activation::apply`.

                    "
"
                "
"

                "
"

## 3. Step: The Original Perceptron

The oldest artificial neuron, the **perceptron**, used a hard threshold: output 1 if `z ≥ 0`, otherwise 0.

                    "
"step(z) = 1 if z ≥ 0, otherwise 0"
"

Its graph is flat on both sides with a jump at zero. Flat means the slope is zero everywhere except at the jump, where no slope exists. A zero slope tells a learning algorithm nothing about which way to move the weights, so we make `derivative` fail loudly. Using Step in a network you train with gradients is a programming error, not something to silently tolerate.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/activation/Step.java"
"Copy"
"

```java
package com.neuralscratch.activation;

public final class Step implements Activation {
    @Override
    public double apply(double z) {
        // Threshold at zero: z >= 0 fires (1), z < 0 stays silent (0).
        return z >= 0.0 ? 1.0 : 0.0;
    }

    @Override
    public double derivative(double z) {
        // The slope is 0 for z != 0 and undefined at z == 0: useless for gradients.
        throw new UnsupportedOperationException(
            "Step has no useful derivative; use Sigmoid for gradient-based training"
        );
    }
}
```

                    "
"
                "
"

                "
"

## 4. Identity: Leave the Score Alone

Sometimes we want the raw weighted sum, for example when predicting a temperature or a price that can be any real number. The identity activation returns `z` unchanged. The line `f(z) = z` rises one unit for every unit you move right, so its derivative is always 1.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/activation/Identity.java"
"Copy"
"

```java
package com.neuralscratch.activation;

public final class Identity implements Activation {
    @Override
    public double apply(double z) {
        // f(z) = z
        return z;
    }

    @Override
    public double derivative(double z) {
        // f'(z) = 1 for every z.
        return 1.0;
    }
}
```

                    "
"
                "
"

                "
"

## 5. Sigmoid: A Smooth Step

The **sigmoid** (also called the logistic function) looks like a step that has been sanded smooth. Large negative `z` gives outputs near 0, large positive `z` gives outputs near 1, and `z = 0` gives exactly 0.5.

                    "
"σ(z) = 1 / (1 + e−z)"
"

                        ze−zσ(z)

                            −454.5980.0180
                            −12.7180.2689
                            010.5000
                            10.3680.7311
                            40.0180.9820

### Deriving the derivative in plain steps

We want the slope of `σ(z) = (1 + e−z)−1`. Treat it as an outer function `u−1` applied to an inner function `u = 1 + e−z`, and use the chain rule from Chapter 2: outer slope times inner slope.

                    "
"

- Outer slope: the derivative of `u−1` is `−u−2`.

- Inner slope: the derivative of `1 + e−z` is `−e−z`.

- Multiply: `σ′(z) = −(1 + e−z)−2 × (−e−z) = e−z / (1 + e−z)²`.

- Split the fraction: `[1 / (1 + e−z)] × [e−z / (1 + e−z)]`.

- The first factor is `σ(z)`. The second is `1 − σ(z)`, because `1 − 1/(1+e−z) = e−z/(1+e−z)`.
                    "
"
                    "
"σ′(z) = σ(z) × (1 − σ(z))"
"

This is convenient: once we know the output, the slope costs one subtraction and one multiplication. At `z = 0` the slope is `0.5 × 0.5 = 0.25`, its largest value. Far from zero the curve flattens and the slope approaches 0; this *saturation* will matter when we discuss vanishing gradients and ReLU in Chapter 6.

### Computing it safely in floating point

A `double` can hold numbers up to about `1.8 × 10³⁰⁸`. `Math.exp(710)` is already larger, so it returns `Infinity`. The textbook formula with `z = −1000` evaluates `Math.exp(1000)`. Here Java happens to compute `1 / Infinity = 0`, but the algebraically equivalent form `ez/(1 + ez)` at `z = +1000` becomes `Infinity / Infinity = NaN`, and a single `NaN` spreads through every later calculation. We avoid infinities entirely by choosing the form whose exponent is never positive:

                    "
"

- For `z ≥ 0`: use `1 / (1 + e−z)`. Here `−z ≤ 0`, so `e−z` lies between 0 and 1.

- For `z z` to get `ez / (1 + ez)`. Here `z z` lies between 0 and 1.
                    "
"
                    "
"
                        "
""
"src/main/java/com/neuralscratch/activation/Sigmoid.java"
"Copy"
"

```java
package com.neuralscratch.activation;

public final class Sigmoid implements Activation {
    @Override
    public double apply(double z) {
        if (z >= 0.0) {
            // sigma(z) = 1 / (1 + e^-z); here e^-z is in (0, 1], so exp cannot overflow.
            return 1.0 / (1.0 + Math.exp(-z));
        }
        // Same value rewritten as e^z / (1 + e^z); for z < 0, e^z is in (0, 1).
        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }

    @Override
    public double derivative(double z) {
        double s = apply(z);
        // sigma'(z) = sigma(z) * (1 - sigma(z)), reusing the stable forward value.
        return s * (1.0 - s);
    }
}
```

                    "
"

For very large `|z|` the derivative rounds to exactly 0.0 (for example, `σ(40)` is 1.0 in `double`). That is a harmless underflow: the true slope is around `10⁻¹⁸`, far below anything that could move a weight.

                "
"

                "
"

## 6. The `Neuron` Class

A neuron stores a weight vector, a bias, and an activation. It exposes the pre-activation as `weightedSum` (handy for explanations and debugging) and the output as `predict`. All fields are final and `Vector` is immutable, so a `Neuron` is an immutable value; training in later chapters happens on layers, not on this class.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/nn/Neuron.java"
"Copy"
"

```java
package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.math.Vector;
import java.util.Objects;

public final class Neuron {
    private final Vector weights;
    private final double bias;
    private final Activation activation;

    public Neuron(Vector weights, double bias, Activation activation) {
        this.weights = Objects.requireNonNull(weights, "weights");
        this.activation = Objects.requireNonNull(activation, "activation");
        if (!Double.isFinite(bias)) {
            throw new IllegalArgumentException("Bias must be a finite number, got " + bias);
        }
        this.bias = bias;
    }

    public Vector weights() {
        return weights;
    }

    public double bias() {
        return bias;
    }

    public Activation activation() {
        return activation;
    }

    public int inputSize() {
        return weights.size();
    }

    public double weightedSum(Vector input) {
        Objects.requireNonNull(input, "input");
        if (input.size() != weights.size()) {
            throw new IllegalArgumentException(
                "Neuron expects " + weights.size() + " inputs, got " + input.size()
            );
        }
        // z = w . x + b
        return weights.dot(input) + bias;
    }

    public double predict(Vector input) {
        // a = f(z) = f(w . x + b)
        return activation.apply(weightedSum(input));
    }
}
```

                    "
"

`Vector.dot` would also reject mismatched sizes, but checking here produces a message phrased in neuron terms ("expects 2 inputs"), which is far easier to diagnose inside a larger program.

                "
"

                "
"

## 7. Computing OR by Hand

Logical OR outputs 1 when at least one input is 1. Choose weights `[1, 1]`, bias `−0.5`, and the Step activation:

                        x₀x₁z = x₀ + x₁ − 0.5step(z)OR

                            00−0.500
                            010.511
                            100.511
                            111.511

Every row matches. Changing the bias to `−1.5` gives AND instead, because now both inputs must be on to push `z` above zero. With weights `[−1, −1]` and bias `1.5` you get NAND (NOT AND). Keep these three in mind; we will combine them shortly.

### The geometric picture

The neuron fires when `w₀x₀ + w₁x₁ + b ≥ 0`. The boundary `w₀x₀ + w₁x₁ + b = 0` is a straight line in the plane of inputs. For OR it is `x₀ + x₁ = 0.5`: the point (0, 0) lies on one side, the other three corners on the other side.

                    "
"
                        "
""
"OR is linearly separable"
"Copy"
"

```
x1
 1 |  1           1
   |
 0 |  0           1
   +------------------ x0
      0           1

The line x0 + x1 = 0.5 passes between (0, 0) and the other three points.
```

                    "
"

A Sigmoid neuron draws the same kind of line; it just reports a smooth confidence instead of a hard 0/1. With weights `[4, 4]` and bias `−2`, the outputs are about 0.12 for (0, 0) and 0.88 or higher for the other inputs.

                "
"

                "
"

## 8. Why One Neuron Cannot Compute XOR

XOR ("exclusive or") outputs 1 when *exactly one* input is 1. Its truth table is OR with the last row flipped: (1, 1) → 0. On the grid, the 1s sit at opposite corners (0, 1) and (1, 0), the 0s at (0, 0) and (1, 1). Try to draw one straight line with both 1s on one side and both 0s on the other; you cannot.

                    "
"
                        "
"A four-line proof"
"

Suppose a step neuron with weights `w₀, w₁` and bias `b` computes XOR. Each row gives an inequality:

                        "
"

- (0, 0) → 0 requires `b  0`, so `w₀ + w₁ + b > 0`. That contradicts line 4. No weights and bias can satisfy all four rows.

                    "
"

The same argument applies to Sigmoid if we read "output ≥ 0.5" as 1, because `σ(z) ≥ 0.5` exactly when `z ≥ 0`. The limitation is not the activation; it is that a single neuron can only draw *one straight line*. Problems whose classes can be split by such a line (or, in higher dimensions, a flat *hyperplane*) are called **linearly separable**. XOR is not.

### The way out: more neurons

Notice that `XOR(x₀, x₁) = AND(OR(x₀, x₁), NAND(x₀, x₁))`. We already have neurons for OR, NAND, and AND. If two neurons compute OR and NAND side by side, and a third neuron ANDs their outputs, the combination computes XOR. The first two form a **hidden layer**; Chapter 4 builds exactly this structure with matrices, and Chapter 5 shows how a network can discover such weights on its own.

                "
"

                "
"

## 9. A Runnable Demo

The demo builds the OR neuron, a Sigmoid version, and an Identity neuron, then performs a brute-force search over 15,625 step neurons (each weight and the bias range from −3 to 3 in steps of 0.25). The proof says no neuron can get all four XOR rows right; the search lets you see that for yourself.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/examples/Chapter3Demo.java"
"Copy"
"

```java
package com.neuralscratch.examples;

import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.Neuron;
import java.util.Arrays;

public final class Chapter3Demo {
    private static final double[][] INPUTS = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private static final double[] OR_TARGETS = {0, 1, 1, 1};
    private static final double[] XOR_TARGETS = {0, 1, 1, 0};

    private Chapter3Demo() {
    }

    public static void main(String[] args) {
        Neuron orNeuron = new Neuron(new Vector(new double[] {1, 1}), -0.5, new Step());
        System.out.println("OR with a step neuron: weights [1, 1], bias -0.5");
        for (int row = 0; row < INPUTS.length; row++) {
            Vector input = new Vector(INPUTS[row]);
            System.out.printf("  %s  z = %5.2f  prediction = %.0f  target = %.0f%n",
                Arrays.toString(INPUTS[row]), orNeuron.weightedSum(input),
                orNeuron.predict(input), OR_TARGETS[row]);
        }

        Neuron smoothOr = new Neuron(new Vector(new double[] {4, 4}), -2, new Sigmoid());
        System.out.println("OR with a sigmoid neuron: weights [4, 4], bias -2");
        for (double[] values : INPUTS) {
            System.out.printf("  %s  output = %.4f%n",
                Arrays.toString(values), smoothOr.predict(new Vector(values)));
        }

        Neuron linear = new Neuron(new Vector(new double[] {2, -1}), 0.5, new Identity());
        // z = 2*3 + (-1)*4 + 0.5 = 2.5
        System.out.printf("Identity neuron on [3, 4]: %.2f%n",
            linear.predict(new Vector(new double[] {3, 4})));

        int bestCorrect = 0;
        int tried = 0;
        for (int i = -12; i <= 12; i++) {
            for (int j = -12; j <= 12; j++) {
                for (int k = -12; k <= 12; k++) {
                    Neuron candidate = new Neuron(
                        new Vector(new double[] {i * 0.25, j * 0.25}), k * 0.25, new Step());
                    bestCorrect = Math.max(bestCorrect, countCorrect(candidate, XOR_TARGETS));
                    tried++;
                }
            }
        }
        System.out.println("Tried " + tried + " step neurons on XOR; best score: "
            + bestCorrect + " of 4");
    }

    private static int countCorrect(Neuron neuron, double[] targets) {
        int correct = 0;
        for (int row = 0; row < INPUTS.length; row++) {
            if (neuron.predict(new Vector(INPUTS[row])) == targets[row]) {
                correct++;
            }
        }
        return correct;
    }
}
```

                    "
"

Expected output:

                    "
"
                        "
""
"Output"
"Copy"
"

```
OR with a step neuron: weights [1, 1], bias -0.5
  [0.0, 0.0]  z = -0.50  prediction = 0  target = 0
  [0.0, 1.0]  z =  0.50  prediction = 1  target = 1
  [1.0, 0.0]  z =  0.50  prediction = 1  target = 1
  [1.0, 1.0]  z =  1.50  prediction = 1  target = 1
OR with a sigmoid neuron: weights [4, 4], bias -2
  [0.0, 0.0]  output = 0.1192
  [0.0, 1.0]  output = 0.8808
  [1.0, 0.0]  output = 0.8808
  [1.0, 1.0]  output = 0.9975
Identity neuron on [3, 4]: 2.50
Tried 15625 step neurons on XOR; best score: 3 of 4
```

                    "
"

`printf` uses your default locale, so some systems print a comma instead of a decimal point. The numbers are the same.

                "
"

                "
"

## 10. Checks Without a Test Framework

As in Chapter 2, a plain `main` method throws `AssertionError` on the first failure. The checks cover known Sigmoid values, behaviour at extreme inputs (finite, no `NaN`), agreement between each analytic derivative and the numerical `Calculus.derivative`, the Step derivative exception, the OR truth table, and shape validation.

                    "
"
                        "
""
"src/test/java/com/neuralscratch/nn/NeuronChecks.java"
"Copy"
"

```java
package com.neuralscratch.nn;

import com.neuralscratch.activation.Activation;
import com.neuralscratch.activation.Identity;
import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.activation.Step;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Vector;

public final class NeuronChecks {
    private static final double TOLERANCE = 1e-9;

    private NeuronChecks() {
    }

    public static void main(String[] args) {
        Activation sigmoid = new Sigmoid();
        close(0.5, sigmoid.apply(0));
        close(1.0 / (1.0 + Math.exp(-2.0)), sigmoid.apply(2));
        close(1.0 / (1.0 + Math.exp(3.0)), sigmoid.apply(-3));
        close(0.25, sigmoid.derivative(0));
        // Symmetry: sigma(-z) = 1 - sigma(z).
        close(1.0 - sigmoid.apply(1.7), sigmoid.apply(-1.7));

        for (double extreme : new double[] {-1000, -745, 709, 1000}) {
            double value = sigmoid.apply(extreme);
            double slope = sigmoid.derivative(extreme);
            if (!Double.isFinite(value) || value < 0 || value > 1 || !Double.isFinite(slope)) {
                throw new AssertionError("Sigmoid unstable at z = " + extreme);
            }
        }
        close(1.0, sigmoid.apply(1000));
        close(0.0, sigmoid.apply(-1000));

        for (double z = -6; z <= 6; z += 0.5) {
            close(Calculus.derivative(sigmoid::apply, z), sigmoid.derivative(z), 1e-8);
            close(Calculus.derivative(new Identity()::apply, z), new Identity().derivative(z), 1e-8);
        }
        close(-2.5, new Identity().apply(-2.5));

        Step step = new Step();
        close(1, step.apply(0));
        close(1, step.apply(3));
        close(0, step.apply(-0.001));
        try {
            step.derivative(0);
            throw new AssertionError("Step.derivative should throw");
        } catch (UnsupportedOperationException expected) {
            // Step cannot be trained with gradients.
        }

        Neuron or = new Neuron(new Vector(new double[] {1, 1}), -0.5, step);
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] targets = {0, 1, 1, 1};
        for (int row = 0; row < inputs.length; row++) {
            close(targets[row], or.predict(new Vector(inputs[row])));
        }

        Neuron weighted = new Neuron(new Vector(new double[] {0.8, 1.5}), -1.0, new Identity());
        close(1.3, weighted.weightedSum(new Vector(new double[] {1, 1})));
        close(1.3, weighted.predict(new Vector(new double[] {1, 1})));
        if (weighted.inputSize() != 2) {
            throw new AssertionError("inputSize should be 2");
        }

        expectIllegalArgument(() -> weighted.predict(new Vector(new double[] {1, 2, 3})));
        expectIllegalArgument(() -> new Neuron(new Vector(new double[] {1}), Double.NaN, step));
        try {
            new Neuron(null, 0, step);
            throw new AssertionError("Expected NullPointerException");
        } catch (NullPointerException expected) {
            // Weights are required.
        }
        System.out.println("All neuron checks passed");
    }

    private static void close(double expected, double actual) {
        close(expected, actual, TOLERANCE);
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
            // Bad shapes and values are rejected before computing.
        }
    }
}
```

                    "
"

Compile every source file into `out`, then run the demo and the checks. Using a list of all sources means you do not need to update the command each time a chapter adds files.

                    "
"
                        "
""
"Terminal (macOS/Linux, project root)"
"Copy"
"

```
find src -name "*.java" > sources.txt
javac -d out @sources.txt
java -cp out com.neuralscratch.examples.Chapter3Demo
java -cp out com.neuralscratch.nn.NeuronChecks
```

                    "
"
                    "
"
                        "
""
"Terminal (Windows PowerShell, project root)"
"Copy"
"

```
Get-ChildItem -Recurse -Filter *.java src | ForEach-Object { '"' + ($_.FullName -replace '\\', '/') + '"' } | Set-Content sources.txt
javac -d out "@sources.txt"
java -cp out com.neuralscratch.examples.Chapter3Demo
java -cp out com.neuralscratch.nn.NeuronChecks
```

                    "
"

Expected final line from the checks: `All neuron checks passed`. Try breaking the code on purpose: replace the stable Sigmoid with the naive `Math.exp(z) / (1 + Math.exp(z))` and watch the extreme-value check fail at `z = 1000`.

                "
"

                "
"

## What You Now Have

                    "
"

- An `Activation` contract pairing each function with its derivative.

- `Step` (perceptron threshold, refuses gradient use), `Identity`, and a numerically stable `Sigmoid` with `σ′ = σ(1 − σ)`.

- An immutable `Neuron` computing `f(w · x + b)` with shape validation.

- A proof that one neuron draws one straight boundary, so it handles OR and AND but never XOR.
                    "
"
                    "
"
                        "
"Before Chapter 4"
"

On paper, compute the XOR network built from OR, NAND, and AND neurons for all four inputs. In the next chapter, you will express those three neurons as two small matrices and run the same calculation for a whole batch at once.

                    "
"
                "
"

                "
"
                    ← Previous: Chapter 2
                    Next: Chapter 4 →
                "
"
            