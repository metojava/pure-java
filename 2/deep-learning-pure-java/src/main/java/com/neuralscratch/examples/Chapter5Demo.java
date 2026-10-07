package com.neuralscratch.examples;

import com.neuralscratch.activation.Sigmoid;
import com.neuralscratch.loss.MeanSquaredError;
import com.neuralscratch.loss.OutputObjective;
import com.neuralscratch.math.Calculus;
import com.neuralscratch.math.Matrix;
import com.neuralscratch.math.Vector;
import com.neuralscratch.nn.DenseLayer;
import com.neuralscratch.nn.Network;
import com.neuralscratch.training.Backpropagation;
import java.util.Locale;
import java.util.Random;

public final class Chapter5Demo {
    private static final int[] SIZES = {2, 3, 1};

    private Chapter5Demo() {
    }

    public static void main(String[] args) {
        OutputObjective mse = new MeanSquaredError();

        DenseLayer single = new DenseLayer(
            Matrix.of(new double[][] {{0.5}}), new Vector(new double[] {-0.5}), new Sigmoid());
        Network neuron = new Network(single);
        double weightBefore = single.weights().get(0, 0);
        double biasBefore = single.biases().get(0);
        double singleLoss = Backpropagation.trainBatch(neuron,
            Matrix.of(new double[][] {{2}}), Matrix.of(new double[][] {{1}}), mse, 1.0);
        System.out.printf(Locale.ROOT,
            "Chain rule example: loss = %.6f, dL/dw = %.6f, dL/db = %.6f%n",
            singleLoss,
            weightBefore - single.weights().get(0, 0),
            biasBefore - single.biases().get(0));
        System.out.printf(Locale.ROOT, "  after one step (rate 1): w = %.6f, b = %.6f%n",
            single.weights().get(0, 0), single.biases().get(0));

        Matrix inputs = Matrix.of(new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Matrix targets = Matrix.of(new double[][] {{0}, {1}, {1}, {1}});

        Vector parameters = flatten(randomNetwork(new Random(3)));
        Network copy = build(parameters);
        Backpropagation.trainBatch(copy, inputs, targets, mse, 1.0);
        // With rate 1: theta_new = theta - gradient, so gradient = theta - theta_new.
        Vector analytic = parameters.subtract(flatten(copy));
        Vector numeric = Calculus.gradient(
            theta -> loss(build(theta), inputs, targets, mse), parameters);
        double maxDifference = 0.0;
        System.out.println("Gradient check on a 2x3x1 network (seed 3):");
        for (int index = 0; index < parameters.size(); index++) {
            double difference = Math.abs(analytic.get(index) - numeric.get(index));
            maxDifference = Math.max(maxDifference, difference);
            System.out.printf(Locale.ROOT, "  parameter %2d  backprop %12.8f  numeric %12.8f%n",
                index, analytic.get(index), numeric.get(index));
        }
        System.out.printf(Locale.ROOT, "  max |difference| = %.2e -> %s%n",
            maxDifference, maxDifference < 1e-7 ? "PASS" : "FAIL");

        Network learner = randomNetwork(new Random(7));
        System.out.println("Preview: 2000 full-batch steps on OR, learning rate 2.0");
        for (int step = 1; step <= 2000; step++) {
            double stepLoss = Backpropagation.trainBatch(learner, inputs, targets, mse, 2.0);
            if (step == 1 || step % 500 == 0) {
                System.out.printf(Locale.ROOT, "  step %4d  loss %.6f%n", step, stepLoss);
            }
        }
        Matrix predictions = learner.predict(inputs);
        for (int row = 0; row < inputs.rows(); row++) {
            System.out.printf(Locale.ROOT, "  [%.0f, %.0f] -> %.4f (target %.0f)%n",
                inputs.get(row, 0), inputs.get(row, 1),
                predictions.get(row, 0), targets.get(row, 0));
        }
    }

    private static Network randomNetwork(Random random) {
        DenseLayer[] layers = new DenseLayer[SIZES.length - 1];
        for (int index = 0; index < layers.length; index++) {
            layers[index] = DenseLayer.random(SIZES[index], SIZES[index + 1],
                new Sigmoid(), random);
        }
        return new Network(layers);
    }

    private static Vector flatten(Network network) {
        int count = 0;
        for (DenseLayer layer : network.layers()) {
            count += layer.inputSize() * layer.outputSize() + layer.outputSize();
        }
        double[] theta = new double[count];
        int offset = 0;
        for (DenseLayer layer : network.layers()) {
            for (int row = 0; row < layer.inputSize(); row++) {
                for (int col = 0; col < layer.outputSize(); col++) {
                    theta[offset++] = layer.weights().get(row, col);
                }
            }
            for (int col = 0; col < layer.outputSize(); col++) {
                theta[offset++] = layer.biases().get(col);
            }
        }
        return new Vector(theta);
    }

    private static Network build(Vector theta) {
        DenseLayer[] layers = new DenseLayer[SIZES.length - 1];
        int offset = 0;
        for (int index = 0; index < layers.length; index++) {
            int in = SIZES[index];
            int out = SIZES[index + 1];
            double[][] weights = new double[in][out];
            for (int row = 0; row < in; row++) {
                for (int col = 0; col < out; col++) {
                    weights[row][col] = theta.get(offset++);
                }
            }
            double[] biases = new double[out];
            for (int col = 0; col < out; col++) {
                biases[col] = theta.get(offset++);
            }
            layers[index] = new DenseLayer(Matrix.of(weights), new Vector(biases), new Sigmoid());
        }
        return new Network(layers);
    }

    private static double loss(Network network, Matrix inputs, Matrix targets,
                               OutputObjective objective) {
        Matrix activation = inputs;
        Matrix z = null;
        for (DenseLayer layer : network.layers()) {
            z = activation.multiply(layer.weights()).addRowVector(layer.biases());
            activation = z.apply(layer.activation()::apply);
        }
        return objective.loss(z, activation, targets);
    }
}