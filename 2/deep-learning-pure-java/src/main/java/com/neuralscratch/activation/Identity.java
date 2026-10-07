package com.neuralscratch.activation;

public final class Identity implements Activation {
    @Override
    public double apply(double z) {
        // f(z) = z
        return z;
    }

    @Override
    public double derivative(double z) {
        // f'(z) = 1 for every z.
        return 1.0;
    }
}