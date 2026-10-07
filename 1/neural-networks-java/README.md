# Neural Networks from Scratch in Pure Java

Java 17 Maven implementation accompanying `neural-network-with-pure-java.html`.

It implements vectors, matrices, dense layers, activations, backpropagation, SGD training, cross-entropy/MSE losses, and a small IDX reader. The default application trains XOR:

```text
mvn compile exec:java
```

No external machine-learning dependency is used.
