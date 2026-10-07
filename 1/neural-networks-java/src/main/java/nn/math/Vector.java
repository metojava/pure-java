package nn.math;

import java.util.Arrays;

public final class Vector {
    private final double[] values;
    public Vector(double... values) { this.values = values.clone(); }
    public int size() { return values.length; }
    public double get(int index) { return values[index]; }
    public void set(int index, double value) { values[index] = value; }
    public double[] toArray() { return values.clone(); }
    public Vector copy() { return new Vector(values); }
    public static Vector zeros(int size) { return new Vector(new double[size]); }
    @Override public String toString() { return Arrays.toString(values); }
}
