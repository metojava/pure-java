package nn.network;

import java.util.ArrayList;
import java.util.List;
import nn.layer.Layer;
import nn.math.Vector;

public final class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();
    public NeuralNetwork add(Layer layer) { layers.add(layer); return this; }
    public Vector predict(Vector input) { Vector current = input; for (Layer layer : layers) current = layer.forward(current); return current; }
    public void backward(Vector gradient, double learningRate) { for (int i = layers.size() - 1; i >= 0; i--) gradient = layers.get(i).backward(gradient, learningRate); }
}
