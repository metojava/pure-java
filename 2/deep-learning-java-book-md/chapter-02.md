**Chapter 2**

# The Math Blueprint – Linear Algebra & Calculus in Java

Build the vectors, matrices, and derivative checks behind every forward and backward pass.

## What We Are Building

In Chapter 1, a neuron computed a weighted sum and then applied an activation function. This chapter supplies the arithmetic: a `Vector` for one list of numbers, a `Matrix` for a rectangular grid, and a tiny `Calculus` utility for checking slopes. All three use only Java 17 and the standard library.

By the end you can copy three complete source files into `src/main/java/com/neuralscratch/math/`, compile them with `javac`, and run an executable check without a third-party test framework. You do not need to know linear algebra beforehand; for each operation, we will start with ordinary numbers, identify the array indices, and then read the Java loop.

**One convention throughout**

A vector has a *size*. A matrix has *rows × columns*. Indices start at zero in Java. Operations create new values; no caller receives a reference to the arrays stored inside these classes.

## 1. Vectors: One List of Numbers

A **vector** is an ordered collection such as `[2, 3, 4]`. In a network it might hold three input features, three weights, or three outputs. Its size is the number of entries; two vectors must have the same size before we add them or take their dot product.

### The dot product, one multiplication at a time

Pair corresponding entries, multiply each pair, then add the products. For `[2, 3, 4]` and `[5, 6, 7]`, the dot product is `2×5 + 3×6 + 4×7 = 56`. A neuron uses exactly this operation to combine its inputs and weights before adding its bias.

a · b = a₀b₀ + a₁b₁ + … + aₙ₋₁bₙ₋₁

The **magnitude** is a vector's length: square each entry, sum the squares, and take the square root. **Normalization** divides every entry by that length, producing a vector of length 1. We reject the zero vector because dividing by its length would mean dividing by zero.

**src/main/java/com/neuralscratch/math/Vector.java**

```java
package com.neuralscratch.math;

import java.util.Objects;

public final class Vector {
    private final double[] values;

    public Vector(double[] values) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) {
            throw new IllegalArgumentException("A vector must have at least one entry");
        }
        this.values = values.clone();
    }

    public static Vector zeros(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }
        return new Vector(new double[size]);
    }

    public int size() {
        return values.length;
    }

    public double get(int index) {
        return values[index];
    }

    public double[] toArray() {
        return values.clone();
    }

    public Vector add(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] + other.values[index];
        }
        return new Vector(result);
    }

    public Vector subtract(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] - other.values[index];
        }
        return new Vector(result);
    }

    public Vector scale(double factor) {
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] * factor;
        }
        return new Vector(result);
    }

    public double dot(Vector other) {
        requireSameSize(other);
        double sum = 0.0;
        for (int index = 0; index < size(); index++) {
            // Each pair contributes one product to the weighted sum.
            sum += values[index] * other.values[index];
        }
        return sum;
    }

    public double magnitude() {
        return Math.sqrt(dot(this));
    }

    public Vector normalize() {
        double length = magnitude();
        if (length == 0.0) {
            throw new ArithmeticException("Cannot normalize a zero vector");
        }
        return scale(1.0 / length);
    }

    private void requireSameSize(Vector other) {
        Objects.requireNonNull(other, "other");
        if (size() != other.size()) {
            throw new IllegalArgumentException(
                "Vector sizes differ: " + size() + " and " + other.size()
            );
        }
    }
}
```

The constructor and `toArray()` both copy arrays. That small cost makes the public value immutable: changing an input array after construction cannot secretly change your weights. Operations also return new vectors. We can profile and optimize allocations later if a training workload needs it.

**Example after running jshell --class-path out**

```java
import com.neuralscratch.math.Vector;

Vector inputs = new Vector(new double[] {2, 3, 4});
Vector weights = new Vector(new double[] {5, 6, 7});
double weightedSum = inputs.dot(weights); // 56.0
Vector direction = new Vector(new double[] {3, 4}).normalize();
// direction holds [0.6, 0.8]; its magnitude is 1.0
```

## 2. Matrices: A Grid with a Shape

A **matrix** is a rectangular table of numbers. Its **shape** is `rows × columns`. A layer with two input features and three neurons can store its weights in a `2 × 3` matrix: each row belongs to an input feature, each column to an output neuron. A single input can be a `1 × 2` row. Their product will be `1 × 3`.

**Shapes before arithmetic**

```
input (1 × 2)       weights (2 × 3)         output (1 × 3)
[ 2  3 ]      ×      [ 1  0  2 ]      =      [ 14  15  10 ]
                      [ 4  5  2 ]
```

We store the matrix as `double[][]`: `data[row][column]`. Each row is an array with the same number of columns. Java does not enforce rectangular arrays, so the `of` factory validates the shape and copies every row. Our zero-filled constructor rejects dimensions that are not positive.

### The complete `Matrix` class

Read the method names first; then come back to the loops after the worked multiplication below. This is one complete source file, not a collection of placeholders.

**src/main/java/com/neuralscratch/math/Matrix.java**

```java
package com.neuralscratch.math;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

public final class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be positive");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public static Matrix of(double[][] values) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException("Matrix must have positive dimensions");
        }
        Matrix result = new Matrix(values.length, values[0].length);
        for (int row = 0; row < result.rows; row++) {
            if (values[row] == null || values[row].length != result.cols) {
                throw new IllegalArgumentException("All matrix rows must have equal length");
            }
            System.arraycopy(values[row], 0, result.data[row], 0, result.cols);
        }
        return result;
    }

    public static Matrix column(Vector vector) {
        Objects.requireNonNull(vector, "vector");
        Matrix result = new Matrix(vector.size(), 1);
        for (int row = 0; row < vector.size(); row++) {
            result.data[row][0] = vector.get(row);
        }
        return result;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public Vector toColumnVector() {
        if (cols != 1) {
            throw new IllegalStateException("Expected one column, got " + cols);
        }
        double[] values = new double[rows];
        for (int row = 0; row < rows; row++) {
            values[row] = data[row][0];
        }
        return new Vector(values);
    }

    public Matrix add(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] + other.data[row][col];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] - other.data[row][col];
            }
        }
        return result;
    }

    public Matrix scale(double factor) {
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] * factor;
            }
        }
        return result;
    }

    public Matrix hadamard(Matrix other) {
        requireSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Hadamard multiplication pairs entries at the same position.
                result.data[row][col] = data[row][col] * other.data[row][col];
            }
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (cols != other.rows) {
            throw new IllegalArgumentException(
                "Cannot multiply " + rows + "x" + cols + " by "
                    + other.rows + "x" + other.cols
            );
        }
        Matrix result = new Matrix(rows, other.cols);
        for (int row = 0; row < rows; row++) {
            for (int shared = 0; shared < cols; shared++) {
                double left = data[row][shared];
                for (int col = 0; col < other.cols; col++) {
                    // Accumulate one row/column dot product in each output cell.
                    result.data[row][col] += left * other.data[shared][col];
                }
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[col][row] = data[row][col];
            }
        }
        return result;
    }

    public Matrix reshape(int newRows, int newCols) {
        if (newRows <= 0 || newCols <= 0
                || (long) newRows * newCols != (long) rows * cols) {
            throw new IllegalArgumentException("Reshape must preserve the entry count");
        }
        Matrix result = new Matrix(newRows, newCols);
        int index = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Keep row-major order while placing each entry in its new shape.
                result.data[index / newCols][index % newCols] = data[row][col];
                index++;
            }
        }
        return result;
    }

    public Matrix addRowVector(Vector bias) {
        Objects.requireNonNull(bias, "bias");
        if (bias.size() != cols) {
            throw new IllegalArgumentException(
                "Bias size " + bias.size() + " must equal column count " + cols
            );
        }
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = data[row][col] + bias.get(col);
            }
        }
        return result;
    }

    public Matrix apply(DoubleUnaryOperator function) {
        Objects.requireNonNull(function, "function");
        Matrix result = new Matrix(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.data[row][col] = function.applyAsDouble(data[row][col]);
            }
        }
        return result;
    }

    private void requireSameShape(Matrix other) {
        Objects.requireNonNull(other, "other");
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException(
                "Matrix shapes differ: " + rows + "x" + cols + " and "
                    + other.rows + "x" + other.cols
            );
        }
    }
}
```

Only `get` exposes data to callers, and it returns a number, not an array. We fill newly allocated result matrices inside methods but never expose those arrays; the public API behaves like immutable values. Java's ordinary array bounds exceptions handle invalid individual indices.

## 3. Matrix Multiplication: Many Dot Products at Once

To multiply an `m × n` matrix by an `n × p` matrix, the *inner* dimensions must match. The result has shape `m × p`. Each output cell is a dot product: one row from the left matrix paired with one column from the right.

result[row][col] = Σshared=0…n−1 left[row][shared] × right[shared][col]

For the first cell of our earlier example, `[2, 3] · [1, 4] = 2×1 + 3×4 = 14`. The next cell pairs `[2, 3]` with `[0, 5]`, giving `15`. No mysterious rule is hiding here; the three loops in `multiply` simply repeat this arithmetic for every output cell.

We loop over `row`, then `shared`, then `col`. For a fixed `shared`, that innermost loop walks along a row of the right-hand `double[][]` and a row of the result. This often has better memory locality than repeatedly stepping down columns. It does not change the mathematical answer.

**Check shapes before values**

`2 × 3` times `3 × 4` yields `2 × 4`. `2 × 3` times `2 × 4` is invalid, regardless of the entries. Remember this rule when a future layer reports a dimension mismatch.

## 4. Transpose and Reshape: Different Ways to Arrange Data

**Transposition** swaps rows and columns: entry `(row, col)` moves to `(col, row)`. A `2 × 3` matrix becomes `3 × 2`. In backpropagation, transposing a weight matrix lets us carry error signals back toward the previous layer.

**Reshaping** also changes dimensions, but it does *not* swap entries by their coordinates. It reads all entries row by row and writes them into a new shape in the same order. The total entry count must stay the same. Transpose and reshape are not interchangeable.

**Transpose versus reshape**

```
Original (2 × 3)       Transpose (3 × 2)       Reshape (3 × 2)
[ 1  2  3 ]            [ 1  4 ]               [ 1  2 ]
[ 4  5  6 ]            [ 2  5 ]               [ 3  4 ]
                       [ 3  6 ]               [ 5  6 ]
```

To treat a vector as a column matrix, call `Matrix.column(vector)`; to recover it, call `toColumnVector()` on a one-column matrix. We do *not* silently guess whether a vector should be a row or column. A batch of input examples, by contrast, uses one row per example.

## 5. Element-Wise Arithmetic and Bias Broadcasting

Some operations pair entries in the *same position*. Addition and subtraction work this way. So does the **Hadamard product**: `[[1, 2]]` hadamard `[[3, 4]]` gives `[[3, 8]]`. It is *not* matrix multiplication; both inputs must have exactly the same shape. We will need Hadamard products when combining error signals with activation derivatives.

| Operation | Input shapes | Result |
| --- | --- | --- |
| `add`, `subtract`, `hadamard` | identical | same shape |
| `multiply` | `m × n` and `n × p` | `m × p` |
| `scale`, `apply` | one matrix | same shape |
| `addRowVector` | `m × n` and vector of size `n` | `m × n` |

**Broadcasting** means reusing a small value across a larger shape. Here we implement only the rule we need: add one bias vector to *each row* of a matrix. A generic broadcasting engine would be harder to reason about and is unnecessary for our network. `apply` takes a standard-library `DoubleUnaryOperator` and transforms every entry independently.

**One batched layer step**

```
Matrix inputs = Matrix.of(new double[][] {{2, 3}, {1, 1}});
Matrix weights = Matrix.of(new double[][] {{1, 0, 2}, {4, 5, 2}});
Vector bias = new Vector(new double[] {1, -1, 0});

Matrix weighted = inputs.multiply(weights);  // shape: 2 × 3
Matrix shifted = weighted.addRowVector(bias); // rows: [15, 14, 10], [6, 4, 4]
Matrix activated = shifted.apply(value -> Math.max(0.0, value));
```

The last line is a preview of ReLU, an activation function we will revisit later. For now, notice that every calculation already has a visible shape, and a batch of two inputs produces two output rows.

## 6. Calculus: Measuring How an Output Changes

A **derivative** is a local slope. If `f(x) = x²`, then near `x = 3`, increasing `x` by a tiny amount changes `f(x)` by roughly six times that amount: the derivative there is `2×3 = 6`. A **partial derivative** asks the same question for one input while holding the other inputs fixed. The vector of all partial derivatives is the **gradient**.

We can estimate a derivative by measuring the function a tiny step to the left and right. This is called a **central difference**:

f′(x) ≈ [f(x + h) − f(x − h)] / (2h)

For a function of several inputs, repeat the measurement once for each coordinate. The following utility makes that procedure explicit. It uses Java SE's `DoubleUnaryOperator` and `ToDoubleFunction`, not an automatic-differentiation library.

**src/main/java/com/neuralscratch/math/Calculus.java**

```java
package com.neuralscratch.math;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

public final class Calculus {
    private static final double STEP = 1e-5;

    private Calculus() {
    }

    public static double derivative(DoubleUnaryOperator function, double at) {
        Objects.requireNonNull(function, "function");
        // A centered difference estimates the tangent's slope at at.
        return (function.applyAsDouble(at + STEP)
            - function.applyAsDouble(at - STEP)) / (2.0 * STEP);
    }

    public static Vector gradient(ToDoubleFunction<Vector> function, Vector point) {
        Objects.requireNonNull(function, "function");
        Objects.requireNonNull(point, "point");
        double[] partials = new double[point.size()];
        for (int coordinate = 0; coordinate < point.size(); coordinate++) {
            double[] plus = point.toArray();
            double[] minus = point.toArray();
            plus[coordinate] += STEP;
            minus[coordinate] -= STEP;
            // Only this coordinate changes; the rest are held fixed.
            partials[coordinate] = (
                function.applyAsDouble(new Vector(plus))
                    - function.applyAsDouble(new Vector(minus))
            ) / (2.0 * STEP);
        }
        return new Vector(partials);
    }
}
```

**Checking a two-variable gradient**

```
Vector point = new Vector(new double[] {3, 4});
Vector slope = Calculus.gradient(
    position -> position.get(0) * position.get(0)
        + position.get(1) * position.get(1),
    point
);
// f(x, y) = x² + y²; slope is approximately [6, 8].
```

### Why not use finite differences to train?

A gradient for `N` parameters needs `2N` function evaluations with this method. Worse, a step that is too large misses fine detail, while a step that is too small magnifies floating-point rounding. We will use *analytic* derivatives and the **chain rule** for actual learning in Chapter 5; finite differences are a useful independent check on those derivatives.

The chain rule in plain English: if one value changes another, which then changes a third, multiply their two local rates of change. For `g(x) = (2x + 1)²`, the outside slope is `2(2x + 1)` and the inside slope is `2`. At `x = 3`, `g′(3) = 2×7×2 = 28`. Backpropagation repeatedly applies this idea, but no backpropagation engine is needed for this chapter.

## 7. Test the Math Before Building Neurons

A wrong matrix loop can still produce plausible-looking numbers. Check a small multiplication by hand, test invalid shapes, and verify that a transpose changes dimensions as expected. The following executable test harness needs *only* Java SE. Save it under `src/test/java/com/neuralscratch/math/MathChecks.java`; it throws `AssertionError` on a failed check, even when Java assertions are disabled.

**src/test/java/com/neuralscratch/math/MathChecks.java**

```java
package com.neuralscratch.math;

import java.util.Random;

public final class MathChecks {
    private static final double TOLERANCE = 1e-8;

    private MathChecks() {
    }

    public static void main(String[] args) {
        double[] original = {3, 4};
        Vector vector = new Vector(original);
        original[0] = 99;
        close(3, vector.get(0));
        double[] exported = vector.toArray();
        exported[0] = 99;
        close(3, vector.get(0));
        close(5, vector.magnitude());
        vectorClose(new double[] {0.6, 0.8}, vector.normalize());
        close(11, new Vector(new double[] {1, 2}).dot(
            new Vector(new double[] {3, 4})));
        vectorClose(new double[] {4, 6},
            new Vector(new double[] {1, 2}).add(new Vector(new double[] {3, 4})));
        vectorClose(new double[] {-2, -2},
            new Vector(new double[] {1, 2}).subtract(new Vector(new double[] {3, 4})));
        vectorClose(new double[] {2, 4}, new Vector(new double[] {1, 2}).scale(2));

        double[][] source = {{1, 2, 3}, {4, 5, 6}};
        Matrix left = Matrix.of(source);
        source[0][0] = 99;
        close(1, left.get(0, 0));
        Matrix right = Matrix.of(new double[][] {{7, 8}, {9, 10}, {11, 12}});
        matrixClose(new double[][] {{58, 64}, {139, 154}}, left.multiply(right));
        matrixClose(new double[][] {{1, 4}, {2, 5}, {3, 6}}, left.transpose());
        matrixClose(new double[][] {{1, 2}, {3, 4}, {5, 6}}, left.reshape(3, 2));
        matrixClose(sourceWithoutMutation(), left.transpose().transpose());
        vectorClose(new double[] {3, 4}, Matrix.column(vector).toColumnVector());

        Matrix first = Matrix.of(new double[][] {{1, 2}, {3, 4}});
        Matrix second = Matrix.of(new double[][] {{2, 3}, {4, 5}});
        matrixClose(new double[][] {{3, 5}, {7, 9}}, first.add(second));
        matrixClose(new double[][] {{-1, -1}, {-1, -1}}, first.subtract(second));
        matrixClose(new double[][] {{2, 6}, {12, 20}}, first.hadamard(second));
        matrixClose(new double[][] {{2, 4}, {6, 8}}, first.scale(2));
        matrixClose(new double[][] {{1, 4}, {9, 16}}, first.apply(x -> x * x));
        matrixClose(new double[][] {{11, 22}, {13, 24}},
            first.addRowVector(new Vector(new double[] {10, 20})));

        close(6, Calculus.derivative(x -> x * x, 3), 1e-5);
        vectorClose(new double[] {6, 8}, Calculus.gradient(
            point -> point.get(0) * point.get(0)
                + point.get(1) * point.get(1),
            new Vector(new double[] {3, 4})), 1e-5);

        Random random = new Random(42);
        for (int trial = 0; trial < 50; trial++) {
            int rows = 1 + random.nextInt(4);
            int cols = 1 + random.nextInt(4);
            double[][] values = new double[rows][cols];
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    values[row][col] = random.nextDouble();
                }
            }
            Matrix sample = Matrix.of(values);
            matrixClose(values, sample.transpose().transpose());
            matrixClose(values, sample.reshape(cols, rows).reshape(rows, cols));
        }

        expectIllegalArgument(() -> Matrix.of(new double[][] {{1}, {2, 3}}));
        expectIllegalArgument(() -> left.multiply(first));
        expectIllegalArgument(() -> left.reshape(4, 2));
        expectIllegalArgument(() -> first.addRowVector(new Vector(new double[] {1})));
        expectIllegalArgument(() -> vector.dot(new Vector(new double[] {1})));
        try {
            Vector.zeros(1).normalize();
            throw new AssertionError("Expected zero-vector normalization to fail");
        } catch (ArithmeticException expected) {
            // The zero vector has no direction.
        }
        System.out.println("All math checks passed");
    }

    private static double[][] sourceWithoutMutation() {
        return new double[][] {{1, 2, 3}, {4, 5, 6}};
    }

    private static void close(double expected, double actual) {
        close(expected, actual, TOLERANCE);
    }

    private static void close(double expected, double actual, double tolerance) {
        if (!Double.isFinite(actual) || Math.abs(expected - actual) > tolerance) {
            throw new AssertionError("Expected " + expected + ", got " + actual);
        }
    }

    private static void vectorClose(double[] expected, Vector actual) {
        vectorClose(expected, actual, TOLERANCE);
    }

    private static void vectorClose(double[] expected, Vector actual, double tolerance) {
        if (expected.length != actual.size()) {
            throw new AssertionError("Vector size mismatch");
        }
        for (int index = 0; index < expected.length; index++) {
            close(expected[index], actual.get(index), tolerance);
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
            // Invalid shapes must fail before any arithmetic begins.
        }
    }
}
```

From the project root, create an output directory and compile only these four files. This also works without Maven, JUnit, or a module descriptor:

**Terminal (project root)**

```
mkdir out
javac -d out src/main/java/com/neuralscratch/math/Vector.java src/main/java/com/neuralscratch/math/Matrix.java src/main/java/com/neuralscratch/math/Calculus.java src/test/java/com/neuralscratch/math/MathChecks.java
java -cp out com.neuralscratch.math.MathChecks
```

Expected output: `All math checks passed`. The first multiplication check uses a known result. The seeded loop checks general properties across 50 small matrices: transposing twice and reshaping back must recover the original values. Its fixed seed makes failures repeatable. Try changing one loop index and rerun the checks to see which invariant catches it.

## What You Now Have

- `Vector`: size validation, arithmetic, dot product, magnitude, normalization, and defensive copies.

- `Matrix`: rectangular storage, shape checks, multiplication, transpose, reshape, Hadamard multiplication, row-wise bias addition, and element-wise functions.

- `Calculus`: approximate derivatives and gradients for checking the exact derivatives we will write later.

- A repeatable, dependency-free check for representative values, shapes, invalid input, and numerical slopes.
**Before Chapter 3**

Write down the shape of every intermediate value in `inputs.multiply(weights).addRowVector(bias)`. If you can explain why a single output cell is a dot product, you are ready to build your first neuron.

Next we will add the bias to one weighted sum, apply Step or Sigmoid, and see exactly what a perceptron can—and cannot—learn. Keep the math classes as they are: they are small enough to inspect whenever a later network result looks wrong.

[← Previous: Chapter 1](chapter-01.md)

[Next: Chapter 3 – The Architecture of a Neuron →](chapter-03.md)
