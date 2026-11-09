"""LAB 1: Understand activation functions before meeting a Transformer.

Run:
    python 01_activation_playground.py

Experiment: change the INPUT_NUMBERS and FEATURE_GATE_PAIRS just below.
All calculations are performed using Python's math module and plain lists.
"""

from mini_transformer import gelu, relu, sigmoid, silu, softmax

# Try adding -4, 0.2 or 4 to see what changes.
INPUT_NUMBERS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

# Each pair is (feature, gate score). These scores could come from two
# different linear projections of the SAME word vector.
FEATURE_GATE_PAIRS = [
    (-2.0, -2.0),
    (-2.0, +2.0),
    (+2.0, -2.0),
    (+2.0, +2.0),
    (+2.0, 0.0),
]


def approximate_slope(function, x, step=0.0001):
    """Estimate slope from the function values just left and right of x."""
    value_left = function(x - step)
    value_right = function(x + step)
    return (value_right - value_left) / (2 * step)


def show_ordinary_activations():
    print("ReLU(x)=max(0,x) | GELU(x)=x*NormalCDF(x) | SiLU(x)=x*sigmoid(x)")
    print("Sigmoid(x)=1/(1+exp(-x))")
    print(f"{'INPUT':>8} {'ReLU':>10} {'GELU':>10} {'SiLU':>10} {'Sigmoid':>10}")
    print("-" * 53)
    for x in INPUT_NUMBERS:
        print(f"{x:8.2f} {relu(x):10.4f} {gelu(x):10.4f} "
              f"{silu(x):10.4f} {sigmoid(x):10.4f}")
    print("\nObserve: ReLU deletes negatives. GELU/SiLU can retain small")
    print("negative values. Sigmoid turns every input into a 0-to-1 range.")


def show_learning_slopes():
    print("\nPART B | How much can a small input change affect the output?")
    print("Slope is a finite-difference approximation, NOT a training result.")
    print(f"{'AT x':>8} {'ReLU slope':>12} {'GELU slope':>12} {'SiLU slope':>12} {'Sigmoid slope':>14}")
    print("-" * 66)
    for x in [-2.0, -1.0, 1.0, 2.0]:
        print(f"{x:8.2f} {approximate_slope(relu, x):12.4f} "
              f"{approximate_slope(gelu, x):12.4f} "
              f"{approximate_slope(silu, x):12.4f} "
              f"{approximate_slope(sigmoid, x):14.4f}")
    print("\nA zero ReLU slope on negative inputs can block local gradient flow.")
    print("Other functions still have slopes (which can also be tiny or negative).")


def show_gates():
    print("\nPART C | GLU and SwiGLU combine TWO numbers, not just one")
    print("Let a = feature branch, b = gate branch.")
    print("GLU(a,b) = a * sigmoid(b)")
    print("SwiGLU(a,b) = SiLU(a) * b  [convention used in this lab]")
    print(f"{'a':>7} {'b':>7} {'sigmoid(b)':>12} {'GLU':>12} {'SwiGLU':>12}")
    print("-" * 58)
    for feature, gate_score in FEATURE_GATE_PAIRS:
        gate_open_fraction = sigmoid(gate_score)
        glu_value = feature * gate_open_fraction
        swiglu_value = silu(feature) * gate_score
        print(f"{feature:7.2f} {gate_score:7.2f} {gate_open_fraction:12.4f} "
              f"{glu_value:12.4f} {swiglu_value:12.4f}")
    print("\nObserve: GLU's sigmoid gate is bounded (0,1), so it attenuates")
    print("or preserves the feature branch. SwiGLU's second branch is NOT")
    print("bounded: it can amplify, reduce, or reverse the sign.")


def show_softmax():
    print("\nPART D | Sigmoid treats scores separately; Softmax compares them")
    scores = [1.0, 2.0, 3.0]
    print(f"Scores for [milk, water, tea] = {scores}")
    independent_sigmoids = [sigmoid(number) for number in scores]
    print("Independent Sigmoid values:",
          [round(number, 4) for number in independent_sigmoids])
    print(f"Sum of Sigmoid values = {sum(independent_sigmoids):.4f} (not 1)")
    for temperature in [0.5, 1.0, 2.0]:
        probabilities = softmax(scores, temperature)
        pretty = [round(number, 4) for number in probabilities]
        print(f"Softmax, temperature={temperature:3.1f}: {pretty}; "
              f"sum={sum(probabilities):.4f}")
    print("\nObserve: Softmax creates a probability distribution with total 1.")
    print("Smaller temperature concentrates probability on the largest score.")


def main():
    show_ordinary_activations()
    show_learning_slopes()
    show_gates()
    show_softmax()
    

if __name__ == "__main__":
    main()
