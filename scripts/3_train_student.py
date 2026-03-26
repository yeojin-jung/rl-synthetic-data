import os
import torch
import numpy as np
import json
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

def formatting_prompts_func(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def train_student(method_name, pool_jsonl, model_id="meta-llama/Llama-3.2-1B"):
    print(f"\n>>> Starting Training: {method_name} using {model_id}")
    
    # 1. Load data
    indices_path = f"data/selected_indices/{method_name}_indices.npy"
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Indices file not found: {indices_path}")
        
    indices = np.load(indices_path)
    
    with open(pool_jsonl, "r") as f:
        pool = [json.loads(line) for line in f]
    
    subset_data = [pool[i] for i in indices]
    dataset = Dataset.from_list(subset_data)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_fn(example):
        text = formatting_prompts_func(example)
        return tokenizer(text, truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    

    # 3. Model & LoRA (Gated access included)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=True
    )

    peft_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_config)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.abspath(f"./models/llama1B-{method_name}"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, 
        learning_rate=2e-4, 
        lr_scheduler_type="cosine",
        num_train_epochs=6,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none"  # Crucial for cluster runs
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"Eval metrics: {eval_metrics}")
    
    # Save to final absolute path
    final_path = os.path.abspath(f"./models/final-{method_name}")
    trainer.save_model(final_path)
    print(f"✅ Finished training and saved model to: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, 
                        choices=["less", "dsir", "prismatic", "prismatic_soft", "random"])
    parser.add_argument("--pool-jsonl", type=str, default="data/processed/candidate_pool.jsonl")
    args = parser.parse_args()
    
    train_student(args.method, args.pool_jsonl)
