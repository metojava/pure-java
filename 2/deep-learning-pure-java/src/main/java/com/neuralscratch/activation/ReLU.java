package com.neuralscratch.activation;

public final class ReLU implements Activation {
    @Override
    public double apply(double z) {
        // ReLU(z) = max(0, z). Math.max also passes NaN through instead of hiding it.
        return Math.max(0.0, z);
    }

    @Override
    public double derivative(double z) {
        // Slope is 1 on the right of zero and 0 on the left.
        // The kink at exactly zero has no single slope; we choose 0.
        return z > 0.0 ? 1.0 : 0.0;
    }
}