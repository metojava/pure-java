package com.neuralscratch.activation;

public final class Sigmoid implements Activation {
    @Override
    public double apply(double z) {
        if (z >= 0.0) {
            // sigma(z) = 1 / (1 + e^-z); here e^-z is in (0, 1], so exp cannot overflow.
            return 1.0 / (1.0 + Math.exp(-z));
        }
        // Same value rewritten as e^z / (1 + e^z); for z < 0, e^z is in (0, 1).
        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }

    @Override
    public double derivative(double z) {
        double s = apply(z);
        // sigma'(z) = sigma(z) * (1 - sigma(z)), reusing the stable forward value.
        return s * (1.0 - s);
    }
}