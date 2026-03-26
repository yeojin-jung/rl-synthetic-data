import torch
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

if os.path.exists("data/features/pool_grads.pt"):
    print(">>> Gradients already exist. Skipping extraction to save 30 mins.")
    sys.exit(0)
    
class GradientExtractor:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", projection_dim=1024):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Proxy Model: {model_id} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load in bfloat16 to save memory, but we will project in float32 for precision
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, config)
        self.model.eval()

        # Deterministic Random Projection
        # We use a fixed seed so pool and validation gradients are in the same subspace
        torch.manual_seed(42)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total LoRA Parameters: {self.total_params}")
        
        # To avoid OOM with a giant [Params x Proj] matrix, we generate rows on the fly 
        # or use a smaller projection dim. 1024 is standard for LESS.
        self.proj_matrix = torch.randn(self.total_params, projection_dim, device=self.device) / np.sqrt(projection_dim)

    def get_projected_gradient(self, instruction, output):
        """
        Extracts and projects the gradient for a single sample.
        """
        self.model.zero_grad()
        
        # Format for SFT
        full_text = f"Instruction: {instruction}\nResponse: {output}"
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Forward and backward
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        # Collect all trainable (LoRA) gradients
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.view(-1).detach().to(self.device))
        
        # Flatten and Project
        flat_grads = torch.cat(grads).to(torch.float32) 
        projected = torch.matmul(flat_grads, self.proj_matrix)
        
        # Optional: Normalize to unit sphere (common in Prismatic/Influence research)
        normed = projected / (torch.norm(projected) + 1e-8)
        
        return normed.detach().cpu()

    def process_and_save(self, input_file, output_file):
        """Processes the JSONL and saves a torch tensor of gradients."""
        results = []
        with open(input_file, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Gradients: {os.path.basename(input_file)}"):
            data = json.loads(line)
            grad_vector = self.get_projected_gradient(data["instruction"], data["output"])
            results.append(grad_vector)

        # Stack into [N, Projection_Dim]
        all_grads = torch.stack(results)
        torch.save(all_grads, output_file)
        print(f"Saved gradients to {output_file} | Shape: {all_grads.shape}")

if __name__ == "__main__":
    import argparse
    extractor = GradientExtractor()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-jsonl", type=str, default="data/processed/candidate_pool.jsonl")
    parser.add_argument("--val-jsonl", type=str, default="data/processed/val_set.jsonl")
    parser.add_argument("--out-dir", type=str, default="data/features")
    args = parser.parse_args()

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process both sets
    extractor.process_and_save(
        args.pool_jsonl, 
        f"{output_dir}/pool_grads.pt"
    )
    extractor.process_and_save(
        args.val_jsonl,
        f"{output_dir}/val_grads.pt"
    )
