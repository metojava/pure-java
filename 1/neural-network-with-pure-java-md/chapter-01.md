## Chapter 1: Introduction to Neural Networks & Why Pure Java?

Neural networks can appear intimidating because they sit at the
intersection of programming, mathematics, statistics, and
optimization.

The central idea, however, is surprisingly simple:
a neural network is a program containing numbers that can adjust
themselves so that its predictions become better.

**Book rule:**

Every important operation in this book is implemented using
Java 17+ and the Java Standard Library.

There is no ND4J, Deeplearning4j, NumPy, TensorFlow,
PyTorch, or external linear-algebra package.

### 1.1 What Is a Neural Network?

At its simplest, a neuron receives numbers, multiplies each
number by a weight, adds those results together, adds a bias,
and passes the result through an activation function.

z = x₁w₁ + x₂w₂ + … + xₙwₙ + b

The weights determine how strongly individual inputs matter.
The bias shifts the result.
The activation function transforms the weighted sum into
the neuron's output.

A neural network is simply many such computations connected
together.

During training, the network changes its weights and biases
to reduce its prediction error.

### 1.2 Why Build It From Scratch?

-
**Understanding.**
Frameworks hide many details. Building the system yourself
exposes the actual computation.

-
**Debuggability.**
Every operation can be inspected with a debugger.

-
**Engineering discipline.**
You become aware of memory allocation, loops, numerical
precision, and data layout.

-
**Transferable knowledge.**
Once you understand matrix multiplication and
backpropagation, higher-level frameworks become easier
to understand.

### 1.3 Why Java?

Java is an excellent environment for learning neural networks.
It has strong typing, mature development tools, predictable
object-oriented structure, excellent debugging support, and
a powerful Just-In-Time compiler.

Java's Standard Edition also provides everything we need for
this project:

- Arrays

- Collections

- Random number generation

- File I/O

- Binary streams

- Command-line applications

- Exceptions

- Testing infrastructure

Pure Java does not mean that this implementation will
automatically outperform specialized numerical libraries.
That is not the purpose.

Our purpose is transparency.

### 1.4 What Does "Pure Java" Mean?

In this book, pure Java means that the numerical operations
are implemented using Java itself and the Java Standard Library.

For example, matrix multiplication will ultimately be ordinary
Java loops:

```
for (int row = 0; row < rows; row++) {

    for (int column = 0; column < columns; column++) {

        double sum = 0.0;

        for (int k = 0; k < commonDimension; k++) {

            sum += left[row][k] * right[k][column];
        }

        result[row][column] = sum;
    }
}
```

There is no hidden native matrix engine behind this operation.

### 1.5 Project Structure

We can organize the project as follows:

```
neural-networks-java/
│
├── src/
│   └── main/
│       └── java/
│           └── nn/
│
│               ├── Main.java
│
│               ├── math/
│               │   ├── Vector.java
│               │   └── Matrix.java
│
│               ├── activation/
│               │   ├── Sigmoid.java
│               │   ├── ReLU.java
│               │   └── Softmax.java
│
│               ├── loss/
│               │   ├── MeanSquaredError.java
│               │   └── CrossEntropy.java
│
│               ├── layer/
│               │   └── DenseLayer.java
│
│               ├── network/
│               │   └── NeuralNetwork.java
│
│               ├── training/
│               │   └── Trainer.java
│
│               └── data/
│                   └── MnistReader.java
│
└── data/
```

### 1.6 Our Development Philosophy

We will deliberately avoid unnecessary abstractions.

A vector is fundamentally an array of numbers.
A matrix is fundamentally a two-dimensional array.
A dense layer contains weights and biases.
A network contains layers.

We want our code to communicate the mathematics directly.

**Important engineering principle:**

Math classes should know mathematics.

Layers should know neural-network structure.

Training code should know optimization.

### 1.7 Creating the First Java Program

```java
package nn;

public class Main {

    public static void main(String[] args) {

        System.out.println(
            "Neural Networks from Scratch in Pure Java"
        );

        System.out.println(
            "Java version: "
            + System.getProperty("java.version")
        );
    }
}
```

### 1.8 Compiling the Program

Using the Java Development Kit:

```
javac -d out src/main/java/nn/Main.java
```

Then run it:

```
java -cp out nn.Main
```

### 1.9 What We Will Build

By the end of this book, our system will be capable of:

-
Representing vectors and matrices.

-
Performing matrix operations.

-
Representing neurons.

-
Creating dense neural-network layers.

-
Performing forward propagation.

-
Calculating loss.

-
Computing gradients.

-
Performing backpropagation.

-
Updating parameters using gradient descent.

-
Training on XOR.

-
Reading MNIST directly from its binary format.

-
Classifying handwritten digits.

### 1.10 Checkpoint

**Before continuing:**

You should be able to explain a neuron using this sentence:

*"A neuron calculates a weighted sum, adds a bias,
and applies an activation function."*

