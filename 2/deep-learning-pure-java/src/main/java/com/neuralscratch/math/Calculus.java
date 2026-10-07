package com.neuralscratch.math;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

public final class Calculus {
    private static final double STEP = 1e-5;

    private Calculus() {
    }

    public static double derivative(DoubleUnaryOperator function, double at) {
        Objects.requireNonNull(function, "function");
        // A centered difference estimates the tangent's slope at at.
        return (function.applyAsDouble(at + STEP)
            - function.applyAsDouble(at - STEP)) / (2.0 * STEP);
    }

    public static Vector gradient(ToDoubleFunction<Vector> function, Vector point) {
        Objects.requireNonNull(function, "function");
        Objects.requireNonNull(point, "point");
        double[] partials = new double[point.size()];
        for (int coordinate = 0; coordinate < point.size(); coordinate++) {
            double[] plus = point.toArray();
            double[] minus = point.toArray();
            plus[coordinate] += STEP;
            minus[coordinate] -= STEP;
            // Only this coordinate changes; the rest are held fixed.
            partials[coordinate] = (
                function.applyAsDouble(new Vector(plus))
                    - function.applyAsDouble(new Vector(minus))
            ) / (2.0 * STEP);
        }
        return new Vector(partials);
    }
}