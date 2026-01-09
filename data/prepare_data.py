import os
import json
from datasets import load_dataset

def prepare_data():
    print("Downloading dataset...")
    output_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("HuggingFaceH4/instruction-dataset", split="test")
    
    # Shuffle and take a small chunk
    dataset = dataset.shuffle(seed=42).select(range(min(2000, len(dataset))))

    # This represents the candidate pool
    pool_data = dataset.select(range(0, int(len(dataset)*0.9)))
    
    # This represents the target set
    target_data = dataset.select(range(int(len(dataset)*0.9), len(dataset)))

    def save_to_jsonl(ds, filename):
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w") as f:
            for entry in ds:
                formatted = {
                    "text": f"Instruction: {entry['prompt']}\nResponse: {entry['completion']}"
                }
                f.write(json.dumps(formatted) + "\n")

    save_to_jsonl(pool_data, "candidate_pool.jsonl")
    save_to_jsonl(target_data, "target_set.jsonl")
    print(f"Successfully saved to data/processed/ (Pool: {len(pool_data)}, Target: {len(target_data)})")

if __name__ == "__main__":
    prepare_data()
    processed_dir = "data/processed"
    files = ["candidate_pool.jsonl", "target_set.jsonl"]
    for file in files:
        file_path = os.path.join(processed_dir, file)
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                data = json.loads(line)
                print(data["text"])
                print("-"*100)