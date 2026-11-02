"""Shrink the student network to small whole numbers (INT8 and lower). Our own code, NumPy only.

The idea: instead of storing a real number like 0.4173, store a small whole number q and
one shared "scale" s for the whole tensor, so that  real value ~= q x s.
With b bits, q goes from -(2^(b-1) - 1) to +(2^(b-1) - 1). For 8 bits that is -127..+127.

"Symmetric, per-tensor": one scale per tensor, chosen so the biggest value (positive or
negative) lands on +-127. Simplest possible scheme, used first on purpose.

Run this file to: measure ranges on recorded data, save the INT8 model, check it.
"""
import numpy as np

from collect import OUT_PATH as DATA_PATH
from student import MODEL_PATH, Student, relu

INT8_PATH = MODEL_PATH.with_name("student_int8.npz")


def qmax(bits):
    """Biggest whole number we allow with this many bits (8 -> 127, 4 -> 7, 2 -> 1)."""
    return 2 ** (bits - 1) - 1


def scale_for(max_abs, bits):
    """One step of the integer grid, in real units: the biggest value maps to qmax."""
    return max_abs / qmax(bits)


def quantize(x, scale, bits):
    """Real numbers -> whole numbers on the grid (rounded, and capped at +-qmax)."""
    x = np.asarray(x, dtype=np.float64)  # same precision as the integer runner, so ties round the same way
    return np.clip(np.round(x / scale), -qmax(bits), qmax(bits))


def fake_quantize(x, scale, bits):
    """Round to the grid and turn straight back into real numbers: keeps only the rounding damage."""
    return quantize(x, scale, bits) * scale


def calibrate(student, obs):
    """Run the float network on recorded observations; note the biggest size seen at each point.

    x  = the 10 scaled inputs, h1 / h2 = the 64 outputs of hidden layers 1 / 2.
    """
    x = (obs - student.in_mean) / student.in_std
    h1 = relu(x @ student.W1 + student.b1)
    h2 = relu(h1 @ student.W2 + student.b2)
    return {"x": float(np.abs(x).max()), "h1": float(np.abs(h1).max()), "h2": float(np.abs(h2).max())}


def load_calibration(student):
    return calibrate(student, np.load(DATA_PATH)["obs"])


class QuantStudent:
    """The float student with chosen parts rounded to a given number of bits.

    w_bits:   weights of all 3 layers
    in_bits:  the 10 inputs
    act_bits: the in-between values (outputs of hidden layers 1 and 2)
    None means "leave that part in full precision".

    Biases: when a layer's input AND weights are both rounded, its bias is rounded too,
    onto the fine (input scale x weight scale) grid, exactly as the integer runner stores it
    (as a 32-bit whole number). Otherwise the bias stays exact.
    """

    def __init__(self, student, ranges, w_bits=None, in_bits=None, act_bits=None):
        self.s = student
        self.ranges = ranges
        self.in_bits, self.act_bits = in_bits, act_bits
        Ws = [student.W1, student.W2, student.W3]
        bs = [student.b1, student.b2, student.b3]
        layer_in = [("x", in_bits), ("h1", act_bits), ("h2", act_bits)]  # what feeds each layer
        if w_bits is not None:
            w_scales = [scale_for(np.abs(W).max(), w_bits) for W in Ws]
            Ws = [fake_quantize(W, s, w_bits) for W, s in zip(Ws, w_scales)]
            for k, (key, bits) in enumerate(layer_in):
                if bits is not None:
                    grid = scale_for(ranges[key], bits) * w_scales[k]
                    bs[k] = np.round(bs[k] / grid) * grid
        self.W1, self.W2, self.W3 = Ws
        self.b1, self.b2, self.b3 = bs

    def _round(self, v, key, bits):
        if bits is None:
            return v
        return fake_quantize(v, scale_for(self.ranges[key], bits), bits)

    def __call__(self, obs):
        s = self.s
        x = self._round((obs - s.in_mean) / s.in_std, "x", self.in_bits)
        h1 = self._round(relu(x @ self.W1 + self.b1), "h1", self.act_bits)
        h2 = self._round(relu(h1 @ self.W2 + self.b2), "h2", self.act_bits)
        return np.clip(h2 @ self.W3 + self.b3, -1.0, 1.0)


def export_int8(student, ranges, path=INT8_PATH):
    """Save the INT8 model: int8 weights, int32 biases, and the scales needed to read them."""
    in_scales = [scale_for(ranges[k], 8) for k in ("x", "h1", "h2")]  # scale of each layer's input
    Ws = [student.W1, student.W2, student.W3]
    bs = [student.b1, student.b2, student.b3]
    w_scales = [scale_for(np.abs(W).max(), 8) for W in Ws]

    arrays = {"in_mean": student.in_mean, "in_std": student.in_std,
              "in_scale": np.array(in_scales), "w_scale": np.array(w_scales)}
    for k in range(3):
        arrays[f"W{k + 1}"] = quantize(Ws[k], w_scales[k], 8).astype(np.int8)
        # A layer's sum comes out in units of (input scale x weight scale), so the bias is
        # stored in the same units, as a 32-bit whole number.
        arrays[f"b{k + 1}"] = np.round(bs[k] / (in_scales[k] * w_scales[k])).astype(np.int32)
    np.savez(path, **arrays)
    return arrays


def main():
    from int8_net import Int8Student

    student = Student()
    ranges = load_calibration(student)
    print("biggest values seen:", {k: round(v, 3) for k, v in ranges.items()})

    arrays = export_int8(student, ranges)
    n_bytes = sum(arrays[f"W{k}"].nbytes + arrays[f"b{k}"].nbytes for k in (1, 2, 3))
    print(f"saved {INT8_PATH}  (weights + biases: {n_bytes} bytes; float32 would be "
          f"{sum(arrays[f'W{k}'].size + arrays[f'b{k}'].size for k in (1, 2, 3)) * 4} bytes)")

    # Check 1: the integer-only runner must match the rounding simulation at 8 bits.
    obs = np.load(DATA_PATH)["obs"][::20]
    sim8 = QuantStudent(student, ranges, w_bits=8, in_bits=8, act_bits=8)(obs)
    int8 = Int8Student(INT8_PATH)(obs)
    print(f"integer runner vs 8-bit simulation, max difference: {np.abs(sim8 - int8).max():.2e}")
    # Check 2: how far INT8 commands are from the float commands, on the -1..+1 scale.
    diff = np.abs(student(obs) - int8)
    print(f"INT8 vs float commands: mean difference {diff.mean():.4f}, max {diff.max():.4f}")


if __name__ == "__main__":
    main()
