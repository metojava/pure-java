## Chapter 7: Training the Network

We now have almost every ingredient required to train a neural
network.

Training repeatedly follows this basic cycle:

- Choose training examples.

- Run forward propagation.

- Calculate the loss.

- Calculate gradients.

- Run backward propagation.

- Update parameters.

- Repeat.

### 7.1 Epochs

One epoch means that the training algorithm has processed the
entire training dataset once.

A dataset might require dozens or thousands of epochs depending
on the problem.

### 7.2 Batch Size

Instead of updating parameters after every individual example,
we can process a group of examples.

Such a group is called a batch.

Common strategies include:

-
**Batch gradient descent:**
use the entire dataset.

-
**Stochastic gradient descent:**
use one example.

-
**Mini-batch gradient descent:**
use a small group.

### 7.3 Training Data Representation

```java
public final class TrainingExample {

    private final double[] input;

    private final double[] target;

    public TrainingExample(double[] input, double[] target) {

        this.input = input.clone();
        this.target = target.clone();
    }

    public double[] input() {

        return input.clone();
    }

    public double[] target() {

        return target.clone();
    }
}
```

### 7.4 Shuffling

If the training data is presented in exactly the same order
every epoch, the order can introduce undesirable correlations.

Java provides a convenient solution:

```
java.util.Collections.shuffle(
    trainingExamples,
    random
);
```

### 7.5 A Basic Training Loop

```
for (int epoch = 0; epoch < epochs; epoch++) {

    java.util.Collections.shuffle(
        trainingExamples,
        random
    );

    double totalLoss = 0.0;

    for (TrainingExample example  : trainingExamples) {

        Vector input =
            new Vector(example.input());

        Vector target =
            new Vector(example.target());

        /*
         * Forward pass.
         */
        Vector prediction =
            network.predict(input);

        /*
         * Calculate loss.
         */
        double loss =
            calculateLoss(
                prediction,
                target
            );

        totalLoss += loss;

        /*
         * Backward pass.
         *
         * The network calculates gradients
         * and updates its parameters.
         */
        network.backward(
            target,
            learningRate
        );
    }

    double averageLoss =
        totalLoss
        / trainingExamples.size();

    System.out.printf(
        "Epoch %d - Loss: %.6f%n",
        epoch + 1,
        averageLoss
    );
}
```

### 7.6 Learning Rate

The learning rate controls the size of each parameter update.

If the learning rate is too large, training can become unstable.

If it is too small, training may be painfully slow.

For a simple network, values such as:

```
0.1
0.01
0.001
```

are useful starting points.

### 7.7 Training Metrics

We should not blindly train and hope for the best.

At minimum, print:

- Epoch

- Training loss

- Validation loss

- Accuracy

Example:

```
Epoch 1   Loss=0.6912   Accuracy=51.2%
Epoch 2   Loss=0.6481   Accuracy=61.8%
Epoch 3   Loss=0.5214   Accuracy=73.5%
Epoch 4   Loss=0.3918   Accuracy=82.1%
```

### 7.8 Overfitting

A network can memorize its training data without learning
patterns that generalize to unseen examples.

This is called overfitting.

A useful warning sign is:

training loss ↓
while
validation loss ↑

The network is becoming increasingly confident on the examples
it has already seen but worse on unseen examples.

### 7.9 Deterministic Experiments

Always consider using a fixed random seed while developing:

```
Random random =
    new Random(42);
```

This makes experiments repeatable.

Once your implementation is correct, you can remove the fixed seed
for experiments where you want different random initializations.

