package nn.training;

import java.util.List;
import nn.loss.Loss;
import nn.math.Vector;
import nn.network.NeuralNetwork;

public final class Trainer {
    private final NeuralNetwork network; private final Loss loss; private final double learningRate;
    public Trainer(NeuralNetwork network, Loss loss, double learningRate) { this.network = network; this.loss = loss; this.learningRate = learningRate; }
    public double epoch(List<TrainingExample> examples) {
        double total = 0;
        for (TrainingExample example : examples) {
            Vector prediction = network.predict(example.input());
            total += loss.value(prediction, example.target());
            network.backward(loss.gradient(prediction, example.target()), learningRate);
        }
        return total / examples.size();
    }
}
