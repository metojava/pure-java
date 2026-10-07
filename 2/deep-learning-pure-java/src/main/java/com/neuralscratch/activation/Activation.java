package com.neuralscratch.activation;

public interface Activation {
    // a = f(z): the neuron's output for pre-activation z.
    double apply(double z);

    // f'(z): the local slope used by the chain rule during backpropagation.
    double derivative(double z);
}