import torch
from torch.optim import AdamW
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class GradientExtractor:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct", projection_dim=1024, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"Using device: {self.device} | Batch Size: {self.batch_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Qwen2.5 padding fix
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(self.device)

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, config)
        self.model.eval()

        # Random Projection setup
        torch.manual.seed(123)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.proj_matrix = torch.randn(total_params, projection_dim, device=self.device)
    
    def finetune_subset(self, training_data, target_texts, epochs=3):
        """
        Fine-tunes the LoRA adapters on 'training_data' and measures loss on 'target_texts'.
        """
        # 1. Prepare Target Data (for evaluation)
        target_inputs = self.tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        # 2. Fine-tuning
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=5e-05)

        for epoch in range(epochs):
            train_inputs = self.tokenizer(training_data, return_tensors="pt", padding=True, 
                                        truncation=True, max_length=512).to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(**train_inputs, labels=train_inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # 3. Evaluation (Post training loss on target)
        self.model.eval()
        with torch.no_grad():
            target_outputs = self.model(**target_inputs, labels=target_inputs["input_ids"])
            final_loss = target_outputs.loss.item()

        for layer in self.model.base_model.modules():
            if hasattr(layer, 'reset_lora_parameters'):
                layer.reset_lora_parameters("default")
        
        # GPU memory cleanup
        del optimizer
        torch.cuda.empty_cache()
        
        return final_loss


    def process_file(self, input_path):
        all_grads = []
        texts = []
        
        with open(input_path, "r") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Processing {os.path.basename(input_path)}"):
            batch_texts = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            # We must handle gradients individually within the batch to get sample-specific info
            for j in range(len(batch_texts)):
                self.model.zero_grad()
                single_input = {k: v[j:j+1] for k, v in inputs.items()}
                outputs = self.model(**single_input, labels=single_input["input_ids"])
                outputs.loss.backward()

                # Qwen2.5-0.5B has 24 layers. We target 4 modules (q, v, k, o) per layer
                # and each has 2 LoRA matrices ($A$ and $B$)
                # the length of the grads list will be 24 * 4 * 2 = 192 tensors.
                # each tensor will be 896 x (hidden_size)
                grads = [p.grad.view(-1) for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                flat_grads = torch.cat(grads).to(torch.float32) # ensure match with proj_matrix
                
                projected = torch.matmul(flat_grads, self.proj_matrix)
                normed = projected / (torch.norm(projected) + 1e-8)
                all_grads.append(normed.detach().cpu())

        return torch.stack(all_grads)

                