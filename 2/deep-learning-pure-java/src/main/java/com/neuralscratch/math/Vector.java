package com.neuralscratch.math;

import java.util.Objects;

public final class Vector {
    private final double[] values;

    public Vector(double[] values) {
        Objects.requireNonNull(values, "values");
        if (values.length == 0) {
            throw new IllegalArgumentException("A vector must have at least one entry");
        }
        this.values = values.clone();
    }

    public static Vector zeros(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive");
        }
        return new Vector(new double[size]);
    }

    public int size() {
        return values.length;
    }

    public double get(int index) {
        return values[index];
    }

    public double[] toArray() {
        return values.clone();
    }

    public Vector add(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] + other.values[index];
        }
        return new Vector(result);
    }

    public Vector subtract(Vector other) {
        requireSameSize(other);
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] - other.values[index];
        }
        return new Vector(result);
    }

    public Vector scale(double factor) {
        double[] result = new double[size()];
        for (int index = 0; index < size(); index++) {
            result[index] = values[index] * factor;
        }
        return new Vector(result);
    }

    public double dot(Vector other) {
        requireSameSize(other);
        double sum = 0.0;
        for (int index = 0; index < size(); index++) {
            // Each pair contributes one product to the weighted sum.
            sum += values[index] * other.values[index];
        }
        return sum;
    }

    public double magnitude() {
        return Math.sqrt(dot(this));
    }

    public Vector normalize() {
        double length = magnitude();
        if (length == 0.0) {
            throw new ArithmeticException("Cannot normalize a zero vector");
        }
        return scale(1.0 / length);
    }

    private void requireSameSize(Vector other) {
        Objects.requireNonNull(other, "other");
        if (size() != other.size()) {
            throw new IllegalArgumentException(
                "Vector sizes differ: " + size() + " and " + other.size()
            );
        }
    }
}