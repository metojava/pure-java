
                "
"
                    "
"Chapter 1"
"

# Introduction to Neural Networks & Why Pure Java?

Setting the philosophy, explaining the JVM benefits, and setting up the project structure

                "
"

                "
"

## What Is a Neural Network?

At its core, a **neural network** is a mathematical function that maps inputs to outputs. It's a composition of simpler functions—each inspired by how biological neurons process signals. But let's be clear: the "neural" in neural network is a loose metaphor. We're not simulating brains; we're building function approximators.

                    "
"
                        "
"💡 Key Insight"
"

A neural network is just a function `f: ℝⁿ → ℝᵐ` with many adjustable parameters (weights and biases). Learning is the process of finding parameter values that make `f(x) ≈ y` for your training data.

                    "
"

### From Biology to Mathematics

Biological neurons receive signals through dendrites, integrate them in the cell body, and fire an action potential through the axon if the combined signal exceeds a threshold. The artificial equivalent:

                    "
"
                        "
"
                            "
"The Artificial Neuron (Mathematical View)"
"
                        "
"

```
Input vector:     x = [x₁, x₂, ..., xₙ]
Weights:          w = [w₁, w₂, ..., wₙ]
Bias:             b ∈ ℝ

Weighted sum:     z = w·x + b = Σ(wᵢ·xᵢ) + b
Output:           y = σ(z)          ← activation function
```

                    "
"

The **weighted sum** `z` represents the total input signal. The **activation function** `σ` (sigma) introduces non-linearity—without it, a network of any depth collapses to a single linear transformation.

### The Universal Approximation Theorem (Intuitive Version)

One reason neural networks are powerful: with a suitable nonlinear activation, a network with one sufficiently wide hidden layer can approximate a continuous function as closely as desired *on a compact domain*. This is a statement about representational capacity, not a guarantee that training will find the right weights.

Think of it like this: each neuron in a hidden layer learns to recognize a specific pattern (a "feature"). By combining many such pattern detectors with different weights, the network can construct arbitrarily complex decision boundaries.

                    "
"
                        "
"🎯 Practical Takeaway"
"

Depth (more layers) and width (more neurons per layer) increase what a network can represent, but do not guarantee learning. More parameters can make training and evaluation harder and increase the risk of overfitting.

                    "
"
                "
"

                "
"

## Why Build From Scratch?

You might wonder: *"Why not just use Deeplearning4j, TensorFlow Java, or ONNX Runtime?"* Excellent question. Here's the honest answer:

### Understanding vs. Using

Libraries hide details so you can focus on a task. Here, implementing backpropagation yourself makes the mechanics visible. You'll learn where to look when:

                    "
"

- Loss grows or becomes non-finite (check the learning rate and the data).

- The network does not learn (check shapes, activations, and gradients).

- Training accuracy rises but held-out accuracy does not (check for overfitting).
                    "
"

### No Dependency Lock-in

We use only Java 17 and the standard library for the book's examples and checks. That makes the calculations easy to inspect without a machine-learning or test framework; portability still depends on a compatible JDK.

### Deployment Simplicity

The examples compile to ordinary Java class files and can be packaged into a JAR. They need a Java 17-compatible runtime, not Python, native ML libraries, or a GPU. Platform-specific runtimes and deployment formats are outside this book's scope.

### Team Adoption

Your Java team already knows the language, the build tools, the debugger, the profiler. They can read, review, and extend this code *today*—no context switching to Python notebooks.

                "
"

                "
"

## The JVM Advantage

Java's runtime is useful for an inspectable learning project. It is not automatically faster or slower than other approaches: performance depends on the workload, implementation, and hardware. We will measure behavior rather than promise speedups.

### Just-In-Time (JIT) Compilation

A JVM can compile frequently executed methods to machine code and optimize loops at runtime. There is no fixed iteration at which that happens, and warm-up, allocation, and garbage collection can affect timing. The book keeps the arithmetic explicit so you can trace it before trying to optimize it.

### Memory and Data Layout

Our `Vector` and `Matrix` classes copy their input arrays and return new values from operations. This makes the public math API predictable, but may allocate intermediate arrays. Escape analysis can sometimes eliminate allocations; it is not a guarantee that an object is stack-allocated or as fast as in-place array operations. Profile a real workload before changing the design.

### Tooling

                    "
"

- **Debugging:** Step through loops and inspect values and dimensions in a Java debugger.

- **Checks:** Run each chapter's `main`-based checks with `java`; failed checks throw `AssertionError` without requiring `-ea`.

- **Profiling:** JDK tools such as Java Flight Recorder can help investigate a measured bottleneck; recording has a cost.
                    "
"
                "
"

                "
"

## Project Setup

Install a JDK 17 (or newer JDK that supports Java 17 source) and check `javac -version` and `java -version`. Start with just the JDK: Maven is optional. Work from a new `neural-networks-java` directory; the source paths below are relative to it.

### Directory Structure

                    "
"
                        "
""
"Starting layout"
""
"

```
neural-networks-java/
└── src/
    └── main/
        └── java/
            └── com/
                └── neuralscratch/
                    └── examples/
                        └── Chapter1Demo.java
```

                    "
"

Create these directories as you add files. Chapter 2 adds `math` under `src/main/java/com/neuralscratch/` and `MathChecks.java` under `src/test/java/com/neuralscratch/math/`. Later chapters add `activation`, `nn`, `loss`, `training`, `data`, `evaluation`, and `demo` packages. Follow the path shown above each source listing; there is no separate downloadable source tree required.

### One Complete First Program

Save this file at the indicated path. It computes the weighted sum from the first section, then applies Sigmoid. Later chapters replace the inline arithmetic with reusable classes.

                    "
"
                        "
""
"src/main/java/com/neuralscratch/examples/Chapter1Demo.java"
"Copy"
"

```java
package com.neuralscratch.examples;

import java.util.Locale;

public final class Chapter1Demo {
    private Chapter1Demo() {
    }

    public static void main(String[] args) {
        double[] inputs = {1.0, 1.0};
        double[] weights = {0.8, 1.5};
        double bias = -1.0;
        double z = bias;
        for (int index = 0; index < inputs.length; index++) {
            z += weights[index] * inputs[index];
        }
        double output = 1.0 / (1.0 + Math.exp(-z));
        System.out.printf(Locale.ROOT, "z = %.1f, sigmoid(z) = %.4f%n", z, output);
    }
}
```

                    "
"

### Maven Configuration (Optional)

If you prefer Maven, put this `pom.xml` at the project root. It specifies Java 17 and adds no runtime or test dependencies. Maven may download its own build plugins; the JDK-only path does not need Maven at all. The checks in the book are ordinary Java programs, not JUnit tests.

                    "
"
                        "
""
"pom.xml"
"Copy"
"

```
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.neuralscratch</groupId>
    <artifactId>neural-networks-java</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.12.1</version>
                <configuration>
                    <release>17</release>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

                    "
"

There is no `module-info.java` in this setup. Modules are optional; if you decide to add one later, export only packages that already contain source files. None of the examples needs SLF4J or a `requires` directive.

                "
"

                "
"

## Development Workflow

### Compile and Run with the JDK

From the project root, create `out` and run the complete Chapter 1 program. These commands work in PowerShell; `mkdir -p out` can replace the first command in a macOS/Linux shell.

                    "
"
                        "
""
"PowerShell (project root)"
"Copy"
"

```
New-Item -ItemType Directory -Force out | Out-Null
javac -d out src/main/java/com/neuralscratch/examples/Chapter1Demo.java
java -cp out com.neuralscratch.examples.Chapter1Demo
```

                    "
"

Expected output: `z = 1.3, sigmoid(z) = 0.7858`. In Chapter 2, compile its three math files and `MathChecks.java` into the same `out` directory and run `java -cp out com.neuralscratch.math.MathChecks`. No test runner or `-ea` flag is needed. When you have copied more chapters, the following PowerShell commands compile all the Java files and run the Chapter 2 checks:

                    "
"
                        "
""
"PowerShell (project root, after Chapter 2)"
"Copy"
"

```
javac -d out (Get-ChildItem -Recurse -Path src -Filter *.java).FullName
java -cp out com.neuralscratch.math.MathChecks
```

                    "
"

On macOS/Linux, use `javac -d out $(find src -name "*.java")` after creating `out`. Each later chapter gives the class name of its own check program. If you chose Maven, run `mvn compile` followed by `java -cp target/classes com.neuralscratch.examples.Chapter1Demo`. For a JAR, `mvn package` and then `java -cp target/neural-networks-java-1.0.0-SNAPSHOT.jar com.neuralscratch.examples.Chapter1Demo`; the optional POM does not set a main class in its manifest. `mvn test` does not run the book's `main`-based checks.

### Debugging Tips

                    "
"

- **Breakpoints:** Stop inside a loop and inspect the current inputs, weights, and sum.

- **Shapes:** From Chapter 2 onward, check matrix row and column counts before investigating individual values.

- **Checks:** Re-run the relevant chapter's check class after changing a loop or derivative.
                    "
"
                "
"

                "
"

## Code Conventions & Design Philosophy

Throughout this book, we'll keep the calculations small enough to trace by hand and the Java code close to the equations.

### Immutability Where Practical

Chapter 2's `Vector` and `Matrix` copy incoming arrays, do not expose their storage, and return new values from arithmetic. For example, after you have added those classes, this code uses their real constructors and methods:

                    "
"
                        "
""
"After Chapter 2 (in a Java class importing Matrix and Vector)"
""
"

```
Matrix inputs = Matrix.of(new double[][] {{2, 3}});
Matrix weights = Matrix.of(new double[][] {{1, 0}, {4, 5}});
Vector biases = new Vector(new double[] {1, -1});
Matrix result = inputs.multiply(weights).addRowVector(biases);
```

                    "
"

The result is a `1 × 2` matrix containing `[15, 14]`. The layer in Chapter 4 uses the same row-major rule: weight rows correspond to input features, columns to output neurons.

### Controlled Mutability for Training

`DenseLayer` in Chapter 4 holds the current weights and biases and replaces them when applying gradients. The math values it holds remain immutable from callers' point of view. We do not assume this eliminates allocation or makes training fast; profile before optimizing.

### Descriptive Names Where They Help

Use names such as `learningRate`, `weights`, and `biases` in Java. Short symbols like `x`, `W`, and `δ` remain useful in equations; a loop index can be short when its role is clear.

### Interfaces for Pluggability

Chapter 3 defines `Activation` with `apply(double z)` and `derivative(double z)`. Chapter 5 defines `OutputObjective` with a loss and an output delta. Both have *two abstract methods*, so neither is a `@FunctionalInterface`; each is implemented by a named class. For a single-method Java SE function, Chapter 2 uses `DoubleUnaryOperator` (for instance `activation::apply`).

                    "
"
                        "
""
"src/main/java/com/neuralscratch/activation/Activation.java (Chapter 3)"
""
"

```java
package com.neuralscratch.activation;

public interface Activation {
    double apply(double z);
    double derivative(double z);
}
```

                    "
"

### Records for Data Carriers

Chapter 7's `Dataset` is a Java 17 record holding input and target matrices. Its constructor validates that they have the same positive row count; it also supplies `select` for making batches. No separate `Batch`, `Gradients`, or `TrainingMetrics` type is needed.

                "
"

                "
"

## What Comes Next

In Chapter 2, we'll build `Vector`, `Matrix`, and a finite-difference `Calculus` utility for checking slopes. It does *not* implement automatic differentiation. Chapter 3 builds a neuron, Chapter 4 stacks layers, and Chapter 5 derives and checks backpropagation.

Chapter 6 adds ReLU and softmax cross-entropy; Chapter 7 adds shuffled mini-batch training; and Chapter 8 evaluates XOR and reads local MNIST IDX files. Each chapter includes code and executable Java SE checks; MNIST data must be supplied separately.

                    "
"
                        "
"Before You Continue"
"
                        "
"

- Run `Chapter1Demo` with the JDK commands above.

- Keep a scratchpad for derivatives, dimensions, and indices.

- In Chapter 2, copy the math files and run `MathChecks` before building a neuron.
                        "
"
                    "
"
                "
"

                "
"
                    ← Table of Contents
                    Next: Chapter 2 – The Math Blueprint →
                "
"
            