import os
import json
import random
import argparse
import re
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Perturbed MATH/GSM8K data.")
    parser.add_argument("--contamination_level", type=float, default=0.1, help="Fraction of pool to corrupt.")
    parser.add_argument("--perturbation_prob", type=float, default=0.0, help="Probability of changing numbers in a sample (for robustness).")
    parser.add_argument("--pool_size", type=int, default=5000)
    parser.add_argument("--target_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=124)
    return parser.parse_args()

def perturb_text(text):
    """
    Shifts integers in the text by a small random amount.
    Example: 'John has 5 apples' -> 'John has 8 apples'
    """
    def replace_num(match):
        val = int(match.group())
        if val > 1000: return str(val) # Avoid changing years or large constants
        return str(val + random.randint(1, 5))
    
    # Replace independent integers
    return re.sub(r'\b\d+\b', replace_num, text)

def apply_semantic_noise(data_list, level):
    """Contaminates data by swapping outputs (Label Noise)."""
    if level <= 0: return data_list
    indices = random.sample(range(len(data_list)), int(len(data_list) * level))
    outputs = [data_list[i]['output'] for i in indices]
    random.shuffle(outputs)
    for i, idx in enumerate(indices):
        data_list[idx]['output'] = outputs[i]
        data_list[idx]['is_contaminated'] = True 
    return data_list

def prepare_data():
    args = parse_args()
    random.seed(args.seed)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading MATH-500 and GSM8K...")
    # MATH-500 is specifically curated for high-quality evaluation
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test") 
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")

    def process_item(example):
        # MATH-500 uses 'problem' and 'solution'
        # GSM8K uses 'question' and 'answer'
        instruction = example.get('problem') or example.get('question')
        output = example.get('solution') or example.get('answer')
        if random.random() < args.perturbation_prob:
            instruction = perturb_text(instruction)
            output = f"(Perturbed) {output}"
        return {"instruction": instruction, "output": output, "is_contaminated": False}

    print("Processing and Merging datasets...")
    formatted_math = [process_item(x) for x in math500]
    formatted_gsm = [process_item(x) for x in gsm8k]
    full_data = formatted_math + formatted_gsm
    random.shuffle(full_data)

    # Split dataset into clean Validation and Target and noisy Candidate
    pool_data = full_data[:args.pool_size]
    val_data = full_data[args.pool_size : args.pool_size+args.val_size]
    target_data = full_data[args.pool_size+args.val_size : args.pool_size+args.val_size+args.target_size]

    print(f"Injecting {args.contamination_level*100}% semantic noise into pool...")
    noisy_pool = apply_semantic_noise(pool_data, args.contamination_level)

    files = {
        "candidate_pool.jsonl": noisy_pool,
        "val_set.jsonl": val_data,
        "target_set.jsonl": target_data
    }

    for filename, data in files.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    print(f"Success! Splits saved to {output_dir}")
    print(f"Pool: {len(noisy_pool)} | Val: {len(val_data)} | Target: {len(target_data)}")

if __name__ == "__main__":
    prepare_data()