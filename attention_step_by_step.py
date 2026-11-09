"""
Run:
    python 02_attention_step_by_step.py
    python 02_attention_step_by_step.py --prompt "I like"
    python 02_attention_step_by_step.py --temperature 0.5

The masked version is used by the language-generating Transformer. The
unmasked version is shown only to explain what a causal mask prevents.
"""

import argparse
import math

from mini_transformer import (
    compute_attention,
    dot_product,
    embedded_inputs,
    format_vector,
    project_queries_keys_values,
    tokenize,
)


def show_attention(prompt, temperature):
    words = tokenize(prompt)
    input_vectors = embedded_inputs(words)
    
    queries, keys, values = project_queries_keys_values(input_vectors)
    

    causal_weights, causal_contexts, scores = compute_attention(
        queries, keys, values, temperature=temperature, causal=True
    )

    unmasked_weights, _, _ = compute_attention(
        queries, keys, values, temperature=temperature, causal=False
    )

    print("\nONE-HEAD ATTENTION LAB")
    print("Tokens:", words)
    print("Width:", len(input_vectors[0]))
    print("Temperature:", temperature)
    print("\n1. Token vectors (embedding plus position)")
    for token_index, word in enumerate(words):
        print(f"  {token_index}. {word:>8} : {format_vector(input_vectors[token_index])}")

    print("\n2. Each word gets three projections: Q (question), K (label), V (content)")
    for token_index, word in enumerate(words):
        print(f"  {word}: Q={format_vector(queries[token_index])}")
        print(f"  {' ' * len(word)}  K={format_vector(keys[token_index])}")
        print(f"  {' ' * len(word)}  V={format_vector(values[token_index])}")

    print("\n3. Raw similarity scores: dot(Q_current, K_candidate) / sqrt(width)")
    heading = "current\\candidate".ljust(20) + "".join(w.rjust(12) for w in words)
    print(heading)
    for position_index, word in enumerate(words):
        numbers = "".join(f"{score:12.4f}" for score in scores[position_index])
        print(word.ljust(20) + numbers)

    if len(words) >= 2:
        current = len(words) - 1
        candidate = 0
        raw_dot = dot_product(queries[current], keys[candidate])
        divisor = math.sqrt(len(queries[current]))
        print(f"\nWorked example: last token '{words[current]}' vs first '{words[candidate]}'")
        print(f"  Q dot K = {raw_dot:.5f}; sqrt(width) = {divisor:.5f}")
        print(f"  scaled score = {raw_dot / divisor:.5f}")

    print("\n4. Softmax after masking: probabilities for each candidate word")
    print("   Each row sums to 1. Future words receive zero probability.")
    print(heading)
    for position_index, word in enumerate(words):
        numbers = "".join(f"{p:12.4f}" for p in causal_weights[position_index])
        print(word.ljust(20) + numbers)

    print("\n5. What would happen WITHOUT the causal mask?")
    print(heading)
    for position_index, word in enumerate(words):
        numbers = "".join(f"{p:12.4f}" for p in unmasked_weights[position_index])
        print(word.ljust(20) + numbers)

    print("\n6. Weighted combination of value vectors for each token")
    for position_index, word in enumerate(words):
        print(f"  Context for '{word}' = {format_vector(causal_contexts[position_index])}")
    print("\nThe attention weights come from Softmax; the context is the")
    print("weighted average/sum of V vectors.")
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="cats drink",
                        help='Known words; e.g. "cats drink" or "I like"')
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Positive number; try 0.5, 1, or 2")
    args = parser.parse_args()
    show_attention(args.prompt, args.temperature)


if __name__ == "__main__":
    main()
