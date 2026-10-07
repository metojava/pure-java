# Deep Learning in Pure Java

This Maven project is extracted from the source listings in `deep-learning-java-book`.
It uses Java 17 and no ML libraries or runtime dependencies.

Included chapters implement:

- immutable vectors and matrices plus finite-difference checks;
- neurons, dense layers, and a sequential network;
- sigmoid, ReLU, and softmax;
- mean-squared-error and softmax-cross-entropy objectives;
- backpropagation, shuffled mini-batch training, XOR, and AND demos;
- raw IDX/MNIST loading and evaluation metrics.

Run the XOR demonstration:

```text
mvn compile exec:java
```

Compile and run the book's checks without Maven:

```text
javac -d target/classes @target/sources.txt
java -cp target/classes com.neuralscratch.math.MathChecks
```
