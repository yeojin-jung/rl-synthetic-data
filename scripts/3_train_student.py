import torch
import numpy as np
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

def formatting_prompts_func(example):
    """
    Standard Alpaca-style format for reasoning tasks.
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

def train_on_subset(method_name, model_id="meta-llama/Llama-3.2-1B"):
    print(f"\n>>> Starting Training: {method_name} using {model_id}")
    
    # 1. Load data
    indices = np.load(f"data/selected_indices/{method_name}_indices.npy")
    with open("data/processed/candidate_pool.jsonl", "r") as f:
        pool = [json.loads(line) for line in f]
    subset_data = [pool[i] for i in indices]
    dataset = Dataset.from_list(subset_data)

    # 2. Tokenizer & Collator
    # We use the Base model, so we must define a pad token
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # This collator ensures the model ONLY learns from the 'Response' section
    response_template = "\n### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 3. Model & LoRA (Dense configuration for 1B parameters)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=32, 
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./models/llama1B-{method_name}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, # effective batch size = 16 * num_device
        learning_rate=2e-4, # Higher LR for small models
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        logging_steps=5,
        bf16=True,
        save_strategy="no",
        remove_unused_columns=False # Important for SFTTrainer custom formatting
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model(f"./models/final-{method_name}")

if __name__ == "__main__":
    for method in ["random", "less", "prismatic", "dsir"]:
        train_on_subset(method)