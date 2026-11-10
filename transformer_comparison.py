"""All six models share the SAME token vectors, Q/K/V/attention weights,
feed-forward projection weights and output head. Only the FFN activation
changes. GLU and SwiGLU use a second projection.

The weights are not trained. A low illustrative loss is NOT evidence
that an activation is more accurate, faster or generally better.

Run examples:
    python 03_transformer_comparison.py
    python 03_transformer_comparison.py --all-prompts
    python 03_transformer_comparison.py --activation swiglu --trace
    python 03_transformer_comparison.py --prompt "I like" --target tea
    python 03_transformer_comparison.py --temperature 0.5
    python 03_transformer_comparison.py --csv toy_results.csv
"""

import argparse
import csv
import math

from mini_transformer import (
    ACTIVATION_NAMES,
    NEXT_WORDS,
    count_feed_forward_parameters,
    format_vector,
    run_transformer,
)

# The expected next words are merely EXERCISE LABELS, not a learned corpus.
# We use a few recognizable sentences to keep the lab concrete.
EXERCISES = [
    ("cats drink", "milk"),
    ("dogs drink", "water"),
    ("I like", "tea"),
]


def cross_entropy_for_expected_word(probability):
    """For a one-word target, loss = -ln(probability assigned to the target)."""
    return -math.log(max(probability, 1e-15))


def show_comparison(prompt, target, activation_names, temperature):
    """Print one row per activation, using identical fixed parameters."""
    print("\n" + "=" * 104)
    print(f"PROMPT: {prompt!r}    EXERCISE TARGET: {target!r}    TEMPERATURE: {temperature}")
    print(f"{'Activation':<11} {'FFN params':>10} {'Top word':<10} "
          f"{'P(target)':>10} {'loss':>10}  " + "  ".join(f"P({w})" for w in NEXT_WORDS))
    print("-" * 104)
    outputs = []
    for activation_name in activation_names:
        result = run_transformer(prompt, activation_name, temperature)
        probabilities = result["next_word_probabilities"]
        top_word_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        target_index = NEXT_WORDS.index(target)
        p_target = probabilities[target_index]
        loss = cross_entropy_for_expected_word(p_target)
        parameter_count = count_feed_forward_parameters(activation_name)
        formatted_probabilities = "  ".join(f"{number:.3f}" for number in probabilities)
        print(f"{activation_name:<11} {parameter_count:>10} {NEXT_WORDS[top_word_index]:<10} "
              f"{p_target:>10.4f} {loss:>10.4f}  {formatted_probabilities}")
        outputs.append({
            "prompt": prompt,
            "exercise_target": target,
            "activation": activation_name,
            "ffn_parameter_count": parameter_count,
            "top_word": NEXT_WORDS[top_word_index],
            "p_target": p_target,
            "toy_loss": loss,
            **{f"p_{word}": number for word, number in zip(NEXT_WORDS, probabilities)},
        })
    print("\nactivation -> altered hidden numbers -> altered logits -> altered probabilities.")
    return outputs


def show_detailed_trace(prompt, activation_name, temperature):
    """Expose the vectors through all Transformer stages for one activation."""
    result = run_transformer(prompt, activation_name, temperature)
    final_token_index = len(result["words"]) - 1
    details = result["feed_forward_details"][final_token_index]
    print("\n" + "=" * 75)
    print(f"DETAILED TRACE | Prompt: {prompt!r} | activation: {activation_name}")
    print("Each line has the real numerical values computed by the code.")
    print("\n1. Tokens:", result["words"])
    print("2. Last token + position vector: ",
          format_vector(result["input_vectors"][final_token_index]))
    print("3. Last token Q: ", format_vector(result["queries"][final_token_index]))
    print("4. Last token K: ", format_vector(result["keys"][final_token_index]))
    print("5. Last token V: ", format_vector(result["values"][final_token_index]))
    print("6. Similarity scores: ",
          format_vector(result["raw_attention_scores"][final_token_index]))
    print("7. Softmax attention weights: ",
          format_vector(result["attention_probabilities"][final_token_index]))
    print("8. Weighted VALUE context: ",
          format_vector(result["attention_context_vectors"][final_token_index]))
    print("9. After attention residual + layer norm: ",
          format_vector(result["after_attention_vectors"][final_token_index]))
    print("10. FFN upward linear scores (6): ",
          format_vector(details["up_branch"]))
    if details["gate_branch"] is not None:
        print("11. FFN gate linear scores (6): ",
              format_vector(details["gate_branch"]))
    print("12. After activation (6): ", format_vector(details["activated_units"]))
    print("13. FFN downward projection (4): ", format_vector(details["output"]))
    print("14. After FFN residual + layer norm (4): ",
          format_vector(result["final_vectors"][final_token_index]))
    print("15. Next-word logits: ", format_vector(result["next_word_logits"]))
    print("16. Final Softmax probabilities:")
    for word, probability in zip(NEXT_WORDS, result["next_word_probabilities"]):
        bar = "#" * int(probability * 40)
        print(f"    {word:<8}: {probability:6.2%} {bar}")


def export_csv(filename, rows):
    """Save results in CSV so spreadsheet users can inspect numerical changes."""
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="cats drink",
                        help='Prompt made of known words: "cats drink", "dogs drink", "I like"')
    parser.add_argument("--target", default=None, choices=NEXT_WORDS,
                        help="Expected next token used only to calculate an illustrative loss")
    parser.add_argument("--activation", default="all", choices=["all"] + ACTIVATION_NAMES,
                        help="Choose a single FFN activation or compare all six")
    parser.add_argument("--all-prompts", action="store_true",
                        help="Run the three tiny exercise sentences")
    parser.add_argument("--trace", action="store_true",
                        help="Show detailed stage-by-stage numbers for one activation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Positive Softmax temperature for attention and word output")
    parser.add_argument("--csv", default=None,
                        help="Optional CSV output path; e.g. toy_results.csv")
    args = parser.parse_args()

    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.trace and args.activation == "all":
        parser.error("For --trace also choose one --activation (e.g. relu or swiglu)")
    if args.trace and args.all_prompts:
        parser.error("Use --trace on one prompt at a time")

    selected_activations = ACTIVATION_NAMES if args.activation == "all" else [args.activation]
    if args.all_prompts:
        exercises = EXERCISES
    else:
        default_target = next(
            (target for prompt, target in EXERCISES if args.prompt == prompt), "milk"
        )
        exercises = [(args.prompt, args.target or default_target)]

    print("TRANSFORMER ACTIVATION COMPARISON")
    print("Single causal attention head | one Transformer block | 4-dimensional tokens")
    print("Shared handmade weights. NOT TRAINED.")

    all_results = []
    for prompt, expected_word in exercises:
        if args.target is not None and args.all_prompts:
            expected_word = args.target
        all_results.extend(show_comparison(prompt, expected_word, selected_activations,
                                           args.temperature))

    if args.trace:
        show_detailed_trace(args.prompt, args.activation, args.temperature)
    if args.csv:
        export_csv(args.csv, all_results)

    


if __name__ == "__main__":
    main()
