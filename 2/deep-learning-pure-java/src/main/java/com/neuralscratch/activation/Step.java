package com.neuralscratch.activation;

public final class Step implements Activation {
    @Override
    public double apply(double z) {
        // Threshold at zero: z >= 0 fires (1), z < 0 stays silent (0).
        return z >= 0.0 ? 1.0 : 0.0;
    }

    @Override
    public double derivative(double z) {
        // The slope is 0 for z != 0 and undefined at z == 0: useless for gradients.
        throw new UnsupportedOperationException(
            "Step has no useful derivative; use Sigmoid for gradient-based training"
        );
    }
}