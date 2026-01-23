import os
import requests
import json
from datasets import load_dataset

def setup_gsm8k():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(output_dir, exist_ok=True)
    print("Fetching GSM8K Test...")
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    with open("data/eval/gsm8k_test.jsonl", "w") as f:
        for ex in gsm8k_test:
            f.write(json.dumps({"instruction": ex['question'], "answer": ex['answer'].split('#### ')[-1].strip()}) + "\n")

def process_gsm_symbolic(input_path, output_path):
    """
    Reads the official GSM-Symbolic JSONL and converts it to our 
    standardized evaluation format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Extract the numeric answer after the #### marker
                full_answer = data.get("answer", "")
                numeric_answer = full_answer.split("####")[-1].strip()
                
                # Create our standard schema
                formatted_item = {
                    "instruction": data["question"],
                    "answer": numeric_answer,
                    "full_solution": full_answer # Useful for debugging reasoning
                }
                
                f_out.write(json.dumps(formatted_item) + "\n")
                processed_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Successfully processed {processed_count} OOD samples to {output_path}")

if __name__ == "__main__":
    # setup_gsm8k()
    input_file = "data/raw/GSM_Symbolic.jsonl" 
    output_file = "data/eval/gsm_symbolic.jsonl"
    
    if os.path.exists(input_file):
        process_gsm_symbolic(input_file, output_file)
    else:
        print(f"File {input_file} not found. Please place the GitHub download there.")