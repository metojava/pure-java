## Chapter 2: The Math Blueprint – Linear Algebra & Calculus in Java

We do not need years of advanced mathematics to implement
a neural network.

We need a relatively small set of mathematical tools:

- Scalars

- Vectors

- Matrices

- Dot products

- Matrix multiplication

- Transpose

- Derivatives

- Gradients

### 2.1 Scalars

A scalar is simply one number.

```
double learningRate = 0.01;
double bias = 0.5;
double temperature = 3.2;
```

### 2.2 Vectors

A vector is an ordered collection of numbers.

```
double[] input = {
    0.2,
    0.7,
    0.4
};
```

We can think of this as:

x = [0.2, 0.7, 0.4]

### 2.3 Matrices

A matrix is a rectangular grid of values.

```
double[][] matrix = {

    { 1.0, 2.0 },

    { 3.0, 4.0 },

    { 5.0, 6.0 }
};
```

This matrix has three rows and two columns.

### 2.4 Dot Product

The dot product multiplies corresponding elements and adds
the results.

a · b = Σ aᵢbᵢ

For example:

[2, 3, 4] · [5, 6, 7]
=
(2×5) + (3×6) + (4×7)
=
56

In Java:

```
public static double dot( double[] a, double[] b) {

    if (a.length != b.length) {
        throw new IllegalArgumentException("Vector sizes must match");
    }

    double sum = 0.0;

    for (int i = 0; i < a.length; i++) {

        sum += a[i] * b[i];
    }

    return sum;
}
```

### 2.5 Building a Vector Class

```java
package nn.math;

import java.util.Arrays;

public final class Vector {

    private final double[] values;

    public Vector(int size) {

        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }

        this.values = new double[size];
    }

    public Vector(double[] values) {

        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Vector cannot be empty");
        }

        this.values = values.clone();
    }

    public int size() {

        return values.length;
    }

    public double get(int index) {

        return values[index];
    }

    public void set( int index, double value) {

        values[index] = value;
    }

    public double[] toArray() {

        return values.clone();
    }

    public double dot(Vector other) {

        if (size() != other.size()) {

            throw new IllegalArgumentException("Vector sizes must match");
        }

        double sum = 0.0;

        for (int i = 0; i < size(); i++) {

            sum +=
                values[i]
                * other.values[i];
        }

        return sum;
    }

    public Vector add(Vector other) {

        checkSameSize(other);

        double[] result =
            new double[size()];

        for (int i = 0; i < size(); i++) {

            result[i] =
                values[i]
                + other.values[i];
        }

        return new Vector(result);
    }

    public Vector subtract(Vector other) {

        checkSameSize(other);

        double[] result =
            new double[size()];

        for (int i = 0; i < size(); i++) {

            result[i] =
                values[i]
                - other.values[i];
        }

        return new Vector(result);
    }

    public Vector multiply(double scalar) {

        double[] result =
            new double[size()];

        for (int i = 0; i < size(); i++) {

            result[i] =
                values[i] * scalar;
        }

        return new Vector(result);
    }

    private void checkSameSize( Vector other) {

        if (size() != other.size()) {

            throw new IllegalArgumentException("Vector sizes must match");
        }
    }

    @Override
    public String toString() {

        return Arrays.toString(values);
    }
}
```

### 2.6 Matrix Multiplication

Matrix multiplication is one of the most important operations
in neural networks.

If matrix A has dimensions:

m × n

and matrix B has dimensions:

n × p

the resulting matrix has dimensions:

m × p

Each output element is a dot product.

Cᵢⱼ = Σₖ AᵢₖBₖⱼ

### 2.7 Matrix Multiplication in Java

```
public static double[][] multiply( double[][] a, double[][] b) {

    int aRows = a.length;
    int aColumns = a[0].length;

    int bRows = b.length;
    int bColumns = b[0].length;

    if (aColumns != bRows) {

        throw new IllegalArgumentException("Incompatible matrix dimensions");
    }

    double[][] result =
        new double[aRows][bColumns];

    for (int row = 0; row < aRows; row++) {

        for (int column = 0; column < bColumns; column++) {

            double sum = 0.0;

            for (int k = 0; k < aColumns; k++) {

                sum +=
                    a[row][k]
                    * b[k][column];
            }

            result[row][column] = sum;
        }
    }

    return result;
}
```

### 2.8 Transpose

Transposition swaps rows and columns.

For example:

A =
[
1 2 3
4 5 6
]

becomes:

Aᵀ =
[
1 4
2 5
3 6
]

Java implementation:

```
public static double[][] transpose( double[][] matrix) {

    int rows = matrix.length;
    int columns = matrix[0].length;

    double[][] result =
        new double[columns][rows];

    for (int row = 0; row < rows; row++) {

        for (int column = 0; column < columns; column++) {

            result[column][row] =
                matrix[row][column];
        }
    }

    return result;
}
```

### 2.9 Matrix Class

```java
package nn.math;

import java.util.Random;

public final class Matrix {

    private final double[][] values;

    public Matrix(int rows, int columns) {

        if (rows <= 0 ||
            columns <= 0) {

            throw new IllegalArgumentException("Matrix dimensions must be positive");
        }

        values =
            new double[rows][columns];
    }

    public Matrix(double[][] source) {

        if (source == null ||
            source.length == 0 ||
            source[0].length == 0) {

            throw new IllegalArgumentException("Matrix cannot be empty");
        }

        int columns =
            source[0].length;

        values =
            new double[source.length][columns];

        for (int row = 0; row < source.length; row++) {

            if (source[row].length != columns) {

                throw new IllegalArgumentException("Rows must have equal length");
            }

            System.arraycopy(
                source[row],
                0,
                values[row],
                0,
                columns
            );
        }
    }

    public static Matrix random( int rows, int columns, Random random, double scale) {

        Matrix result =
            new Matrix(rows, columns);

        for (int row = 0; row < rows; row++) {

            for (int column = 0; column < columns; column++) {

                result.values[row][column] =
                    (random.nextDouble() * 2.0 - 1.0)
                    * scale;
            }
        }

        return result;
    }

    public int rows() {

        return values.length;
    }

    public int columns() {

        return values[0].length;
    }

    public double get( int row, int column) {

        return values[row][column];
    }

    public void set( int row, int column, double value) {

        values[row][column] = value;
    }

    public Vector multiply( Vector vector) {

        if (columns() != vector.size()) {

            throw new IllegalArgumentException("Matrix/vector dimensions do not match");
        }

        double[] result =
            new double[rows()];

        for (int row = 0; row < rows(); row++) {

            double sum = 0.0;

            for (int column = 0; column < columns(); column++) {

                sum +=
                    values[row][column]
                    * vector.get(column);
            }

            result[row] = sum;
        }

        return new Vector(result);
    }

    public Matrix transpose() {

        Matrix result =
            new Matrix(columns(), rows());

        for (int row = 0; row < rows(); row++) {

            for (int column = 0; column < columns(); column++) {

                result.values[column][row] =
                    values[row][column];
            }
        }

        return result;
    }
}
```

### 2.10 Derivatives

Linear algebra allows us to perform the forward computation.
Calculus will allow us to teach the network how to improve.

A derivative tells us how much one quantity changes when
another quantity changes.

Suppose:

y = x²

Then:

dy/dx = 2x

At x = 3, the derivative is 6.

In neural networks, we repeatedly ask questions such as:

"If I change this weight slightly, how much does the loss change?"

### 2.11 Gradient Descent

A gradient tells us the direction in which a function increases
most rapidly.

Because we want to reduce loss, we move in the opposite
direction.

parameter ← parameter − learningRate × gradient

This single equation is one of the central mechanisms of
neural-network training.

### 2.12 Numerical Derivatives

A useful approximation for a derivative is:

f′(x)
≈
[f(x + ε) − f(x − ε)] / (2ε)

This is called a central finite difference.

We will later use it to verify that our backpropagation
implementation is calculating gradients correctly.

### 2.13 Chapter 2 Checkpoint

Before continuing, make sure you can explain:

- What a vector is.

- What a matrix is.

- How a dot product works.

- Why matrix dimensions must match.

- What a transpose does.

- What a derivative represents.

- Why gradient descent subtracts the gradient.

```

I’ll continue with **Chapters 3–8** in the next part, including the complete neuron, dense layers, backpropagation, activation/loss functions, training loop, XOR, and the pure-Java MNIST implementation.

Absolutely — here is **Part 2**, continuing the same HTML document. Append this immediately after Chapter 2 and before the closing `
`, `
`, and `
` tags.

```

