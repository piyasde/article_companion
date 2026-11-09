"""A tiny decoder-only Transformer implemented with ordinary Python lists.

Learning goals:
  * follow token embeddings -> causal attention -> feed-forward -> word probabilities;
  * compare ReLU, GELU, SiLU, Sigmoid, GLU and SwiGLU inside the SAME model;
  * see that Softmax has a different job: normalizing a group of scores.

All numbers here are HAND-WRITTEN teaching weights. They are not
trained on language. Outputs illustrate arithmetic, not reliable predictions.
"""

import math

# The first coordinate is simply "feature 1", the second "feature 2", etc.
# Think of each token as a small row of four numbers. These are NOT embeddings
# produced by a real language model: we made them up for demonstration.
TOKEN_EMBEDDINGS = {
    "I":     [0.20, 0.90, -0.40, 0.30],
    "like":  [0.10, 0.80, 0.60, -0.20],
    "cats":  [0.90, -0.40, 0.70, 0.20],
    "dogs":  [0.80, -0.30, 0.50, -0.10],
    "drink": [-0.20, 0.80, 0.30, 0.50],
}

# This toy model only makes one of these four next-word suggestions.
NEXT_WORDS = ["milk", "water", "tea", "coffee"]
MODEL_WIDTH = 4
HIDDEN_WIDTH = 6

# For any weight matrix below: ONE ROW = weights of ONE output neuron.
# Example: linear([a, b], [[2, 3]], [1]) returns [2*a + 3*b + 1].
QUERY_WEIGHTS = [
    [1.10, 0.10, 0.00, 0.00],
    [0.10, 0.90, 0.15, 0.00],
    [0.00, 0.20, 1.00, 0.10],
    [0.10, 0.00, 0.10, 0.90],
]
KEY_WEIGHTS = [
    [0.90, 0.10, 0.00, 0.05],
    [0.00, 1.05, 0.10, 0.00],
    [0.10, 0.10, 0.90, 0.00],
    [0.00, 0.05, 0.15, 1.00],
]
VALUE_WEIGHTS = [
    [0.85, 0.00, 0.20, 0.00],
    [0.00, 0.90, 0.10, 0.00],
    [0.10, 0.00, 0.80, 0.10],
    [0.00, 0.10, 0.00, 0.90],
]
ATTENTION_OUTPUT_WEIGHTS = [
    [0.90, 0.10, 0.00, 0.00],
    [0.00, 0.90, 0.10, 0.00],
    [0.10, 0.00, 0.90, 0.00],
    [0.00, 0.00, 0.10, 0.90],
]

# Feed-forward network: width 4 -> 6 -> 4.
# GLU and SwiGLU also use GATE_WEIGHTS, so they have MORE parameters.
FEED_FORWARD_INPUT_WEIGHTS = [
    [1.00, -1.00, 0.20, 0.00],
    [-1.00, 0.30, 0.60, 0.20],
    [0.00, 1.10, -0.50, 0.20],
    [-0.80, -0.20, 0.40, 0.60],
    [0.50, 0.20, -0.80, 0.10],
    [0.30, -0.60, 0.90, -0.40],
]
FEED_FORWARD_INPUT_BIAS = [0.20, -0.10, 0.10, -0.15, 0.00, 0.05]
GATE_WEIGHTS = [
    [0.60, -0.20, 0.30, 0.10],
    [-0.10, 0.50, 0.60, 0.00],
    [0.20, -0.40, 0.80, 0.30],
    [0.70, 0.10, -0.50, 0.20],
    [0.30, 0.60, 0.10, -0.50],
    [-0.40, 0.20, 0.30, 0.80],
]
GATE_BIAS = [-0.10, 0.20, -0.20, 0.10, 0.00, 0.15]
FEED_FORWARD_OUTPUT_WEIGHTS = [
    [0.70, -0.40, 0.20, 0.10, -0.50, 0.30],
    [0.10, 0.60, -0.70, 0.20, 0.50, -0.20],
    [-0.30, 0.10, 0.70, -0.50, 0.20, 0.30],
    [0.20, -0.10, 0.30, 0.70, 0.20, -0.60],
]
FEED_FORWARD_OUTPUT_BIAS = [0.00, 0.00, 0.00, 0.00]

# Word "scores" (logits) use the last token's four final features.
LANGUAGE_HEAD_WEIGHTS = [
    [0.70, -0.20, 0.80, 0.00],  # milk
    [0.20, 0.80, 0.30, -0.10],  # water
    [-0.20, 0.20, 0.95, 0.15],  # tea
    [-0.40, 0.60, 0.50, 0.00],  # coffee
]
LANGUAGE_HEAD_BIAS = [0.10, 0.00, 0.10, -0.05]
ACTIVATION_NAMES = ["relu", "gelu", "silu", "sigmoid", "glu", "swiglu"]


def dot_product(first_vector, second_vector):
    """Multiply corresponding numbers and add the products."""
    if len(first_vector) != len(second_vector):
        raise ValueError("Dot product vectors must have the same length")
    total = 0.0
    for index in range(len(first_vector)):
        total += first_vector[index] * second_vector[index]
    return total


def vector_add(first_vector, second_vector):
    """Add two vectors coordinate by coordinate (used in residual connections)."""
    if len(first_vector) != len(second_vector):
        raise ValueError("Vectors must have the same length")
    result = []
    for index in range(len(first_vector)):
        result.append(first_vector[index] + second_vector[index])
    return result


def linear(input_vector, weight_rows, biases):
    """One dense/linear layer: output[j] = dot(weights[j], input) + bias[j]."""
    if len(weight_rows) != len(biases):
        raise ValueError("There must be one bias for every output neuron")
    output_vector = []
    for neuron_index in range(len(weight_rows)):
        weighted_sum = dot_product(weight_rows[neuron_index], input_vector)
        output_vector.append(weighted_sum + biases[neuron_index])
    return output_vector


def sigmoid(number):
    """Return a number strictly between zero and one; written stably."""
    if number >= 0.0:
        return 1.0 / (1.0 + math.exp(-number))
    positive_exponent = math.exp(number)
    return positive_exponent / (1.0 + positive_exponent)


def relu(number):
    """ReLU turns a negative number into zero; positives stay unchanged."""
    return max(0.0, number)


def gelu(number):
    """Exact GELU: x multiplied by the standard-normal cumulative probability."""
    normal_cdf = 0.5 * (1.0 + math.erf(number / math.sqrt(2.0)))
    return number * normal_cdf


def silu(number):
    """SiLU (Swish) is x multiplied by sigmoid(x)."""
    return number * sigmoid(number)


def softmax(scores, temperature=1.0):
    """Turn a LIST of arbitrary scores into probabilities summing to one.

    Subtracting the largest score avoids overflow in exp() and does not
    change the probabilities. Smaller temperature => sharper distribution.
    """
    if not scores:
        raise ValueError("Softmax needs one or more scores")
    if temperature <= 0.0:
        raise ValueError("Softmax temperature must be greater than zero")
    largest_score = max(scores)
    exponentials = []
    for score in scores:
        exponentials.append(math.exp((score - largest_score) / temperature))
    total = sum(exponentials)
    return [number / total for number in exponentials]


def layer_normalize(vector, epsilon=0.00001):
    """Normalize one token's features to mean~0 and variance~1.

    Real Transformers usually also learn one scale and bias per feature.
    We deliberately fix their scale to 1 and bias to 0 in this teaching model.
    """
    mean = sum(vector) / len(vector)
    variance = 0.0
    for number in vector:
        variance += (number - mean) ** 2
    variance /= len(vector)
    scale = math.sqrt(variance + epsilon)
    return [(number - mean) / scale for number in vector]


def position_information(position_index):
    """Simple handcrafted position vector so word order can matter.

    This is not the sinusoidal/rotary mechanism found in many real models.
    """
    return [
        0.08 * position_index,
        -0.04 * position_index,
        0.03 * position_index,
        -0.06 * position_index,
    ]


def tokenize(prompt):
    """Split the short teaching prompt on spaces and validate known words."""
    words = prompt.strip().split()
    if not words:
        raise ValueError("Prompt cannot be empty")
    if len(words) > 8:
        raise ValueError("Use at most 8 tokens in this teaching model")
    for word in words:
        if word not in TOKEN_EMBEDDINGS:
            valid_words = ", ".join(TOKEN_EMBEDDINGS)
            raise ValueError(f"Unknown token {word!r}. Available: {valid_words}")
    return words


def embedded_inputs(words):
    """Look up each token vector, then add its position vector."""
    results = []
    for position_index, word in enumerate(words):
        token_vector = TOKEN_EMBEDDINGS[word]
        position_vector = position_information(position_index)
        results.append(vector_add(token_vector, position_vector))
    return results


def project_queries_keys_values(input_vectors):
    """Make three different views of each word: query, key and value."""
    queries = []
    keys = []
    values = []
    zero_bias = [0.0] * MODEL_WIDTH
    for token_vector in input_vectors:
        queries.append(linear(token_vector, QUERY_WEIGHTS, zero_bias))
        keys.append(linear(token_vector, KEY_WEIGHTS, zero_bias))
        values.append(linear(token_vector, VALUE_WEIGHTS, zero_bias))
    return queries, keys, values


def compute_attention(queries, keys, values, temperature=1.0, causal=True):
    """Calculate scaled dot-product attention for every token position.

    A causal mask means word at position i cannot look at positions > i.
    We skip masked keys rather than giving them a giant negative score.
    """
    if not (len(queries) == len(keys) == len(values)):
        raise ValueError("Q, K and V need the same number of tokens")
    attention_probabilities = []
    attention_contexts = []
    unmasked_score_rows = []
    for current_position in range(len(queries)):
        all_scores = []
        for candidate_position in range(len(keys)):
            score = dot_product(queries[current_position], keys[candidate_position])
            score /= math.sqrt(len(queries[current_position]))
            all_scores.append(score)
        unmasked_score_rows.append(all_scores)

        allowed_count = current_position + 1 if causal else len(keys)
        allowed_scores = all_scores[:allowed_count]
        allowed_probabilities = softmax(allowed_scores, temperature)
        full_probability_row = allowed_probabilities + [0.0] * (len(keys) - allowed_count)
        attention_probabilities.append(full_probability_row)

        context_vector = [0.0] * MODEL_WIDTH
        for candidate_position in range(len(keys)):
            weight = full_probability_row[candidate_position]
            for feature_index in range(MODEL_WIDTH):
                context_vector[feature_index] += weight * values[candidate_position][feature_index]
        attention_contexts.append(context_vector)
    return attention_probabilities, attention_contexts, unmasked_score_rows


def feed_forward(input_vector, activation_name):
    """Transform ONE token with one of six activation choices.

    ReLU/GELU/SiLU/Sigmoid: activation(W_up*x + b_up), then W_down.
    GLU:  (W_up*x + b_up) * sigmoid(W_gate*x + b_gate), then W_down.
    SwiGLU: SiLU(W_up*x + b_up) * (W_gate*x + b_gate), then W_down.

    Multiplications in the gated versions happen COORDINATE BY COORDINATE.
    """
    if activation_name not in ACTIVATION_NAMES:
        raise ValueError(f"Unsupported activation: {activation_name}")
    input_pre_activations = linear(
        input_vector, FEED_FORWARD_INPUT_WEIGHTS, FEED_FORWARD_INPUT_BIAS
    )
    gate_pre_activations = None
    activated_units = []

    if activation_name in ("glu", "swiglu"):
        gate_pre_activations = linear(input_vector, GATE_WEIGHTS, GATE_BIAS)

    for unit_index, first_branch_value in enumerate(input_pre_activations):
        if activation_name == "relu":
            activated = relu(first_branch_value)
        elif activation_name == "gelu":
            activated = gelu(first_branch_value)
        elif activation_name == "silu":
            activated = silu(first_branch_value)
        elif activation_name == "sigmoid":
            activated = sigmoid(first_branch_value)
        elif activation_name == "glu":
            gate = sigmoid(gate_pre_activations[unit_index])
            activated = first_branch_value * gate
        else:  # swiglu
            gate = gate_pre_activations[unit_index]
            activated = silu(first_branch_value) * gate
        activated_units.append(activated)

    output_vector = linear(
        activated_units, FEED_FORWARD_OUTPUT_WEIGHTS, FEED_FORWARD_OUTPUT_BIAS
    )
    return {
        "up_branch": input_pre_activations,
        "gate_branch": gate_pre_activations,
        "activated_units": activated_units,
        "output": output_vector,
    }


def count_feed_forward_parameters(activation_name):
    """Count feed-forward weight and bias numbers, not embeddings/attention."""
    # W_up  = 6*4 weights and 6 biases.
    # W_down = 4*6 weights and 4 biases.
    count = HIDDEN_WIDTH * MODEL_WIDTH + HIDDEN_WIDTH
    count += MODEL_WIDTH * HIDDEN_WIDTH + MODEL_WIDTH
    if activation_name in ("glu", "swiglu"):
        # Gating adds a SECOND 4 -> 6 projection and six more biases.
        count += HIDDEN_WIDTH * MODEL_WIDTH + HIDDEN_WIDTH
    return count


def run_transformer(prompt, activation_name="relu", temperature=1.0):
    """Run the entire, ONE-BLOCK, ONE-HEAD, CAUSAL Transformer.

    Returns a dictionary with ALL intermediate arrays so lab programs can
    inspect every step. Only the last token is used to suggest the next word.
    """
    words = tokenize(prompt)
    input_vectors = embedded_inputs(words)

    # Stage 1: self-attention. A token can consult itself and earlier tokens.
    queries, keys, values = project_queries_keys_values(input_vectors)
    attention_probabilities, context_vectors, score_rows = compute_attention(
        queries, keys, values, temperature=temperature, causal=True
    )

    # Stage 2: output projection, residual connection and layer normalization.
    after_attention = []
    for position_index in range(len(words)):
        attention_output = linear(
            context_vectors[position_index],
            ATTENTION_OUTPUT_WEIGHTS,
            [0.0] * MODEL_WIDTH,
        )
        residual_sum = vector_add(input_vectors[position_index], attention_output)
        after_attention.append(layer_normalize(residual_sum))

    # Stage 3: the FFN (where ReLU/GELU/SiLU/GLU/SwiGLU are compared).
    # Stage 4: another residual connection and layer normalization.
    feed_forward_details = []
    final_vectors = []
    for token_vector in after_attention:
        details = feed_forward(token_vector, activation_name)
        feed_forward_details.append(details)
        residual_sum = vector_add(token_vector, details["output"])
        final_vectors.append(layer_normalize(residual_sum))

    # Stage 5: turn last token's features into four next-word scores.
    logits = linear(final_vectors[-1], LANGUAGE_HEAD_WEIGHTS, LANGUAGE_HEAD_BIAS)
    next_word_probabilities = softmax(logits, temperature=temperature)

    return {
        "words": words,
        "activation": activation_name,
        "input_vectors": input_vectors,
        "queries": queries,
        "keys": keys,
        "values": values,
        "raw_attention_scores": score_rows,
        "attention_probabilities": attention_probabilities,
        "attention_context_vectors": context_vectors,
        "after_attention_vectors": after_attention,
        "feed_forward_details": feed_forward_details,
        "final_vectors": final_vectors,
        "next_word_logits": logits,
        "next_word_probabilities": next_word_probabilities,
    }


def format_vector(vector, decimal_places=3):
    """Make a short, readable numerical vector for the console."""
    return "[" + ", ".join(f"{number:+.{decimal_places}f}" for number in vector) + "]"
