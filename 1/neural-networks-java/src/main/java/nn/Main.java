package nn;

import java.util.List;
import java.util.Random;
import nn.activation.SigmoidActivation;
import nn.layer.ActivationLayer;
import nn.layer.DenseLayer;
import nn.loss.MeanSquaredError;
import nn.math.Vector;
import nn.network.NeuralNetwork;
import nn.training.Trainer;
import nn.training.TrainingExample;

/** Trains a small 2-4-1 network on XOR, as introduced in the guide. */
public final class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
            .add(new DenseLayer(2, 4, new Random(42)))
            .add(new ActivationLayer(new SigmoidActivation()))
            .add(new DenseLayer(4, 1, new Random(43)))
            .add(new ActivationLayer(new SigmoidActivation()));
        List<TrainingExample> examples = List.of(
            new TrainingExample(new Vector(0, 0), new Vector(0)), new TrainingExample(new Vector(0, 1), new Vector(1)),
            new TrainingExample(new Vector(1, 0), new Vector(1)), new TrainingExample(new Vector(1, 1), new Vector(0)));
        Trainer trainer = new Trainer(network, new MeanSquaredError(), 1.0);
        for (int epoch = 1; epoch <= 10_000; epoch++) { double loss = trainer.epoch(examples); if (epoch % 1_000 == 0) System.out.printf("epoch %5d loss %.6f%n", epoch, loss); }
        for (TrainingExample example : examples) System.out.printf("%s -> %.4f%n", example.input(), network.predict(example.input()).get(0));
    }
}
