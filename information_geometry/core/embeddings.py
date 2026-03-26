import os
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_llm_embeddings(
    text,
    model,
    tokenizer=None,
    batch_size=16,
    dtype=torch.float32,
    save_to=None,
    last_position_only=True,
    max_length=None,
    top_k=5,
    layer_indices=None
):
    """
    get embeddings from LLM with support for intermediate layers.
    
    Args:
        layer_indices: List of layer indices to extract (e.g., [-1, -5, -10] or [0, 12, 24]).
                      None means only the last layer (default behavior).
                      Use negative indices to count from the end.
    """
    is_str = isinstance(model, str)

    if is_str:
        model_name = model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()
        print(f"[model] {model.name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        assert tokenizer is not None, "Tokenizer must be provided."

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[Info] Pad token set to EOS token: {tokenizer.pad_token}")

    device = model.device

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    original_order = None
    if batch_size > 1:
        idx_order = np.argsort([len(t) for t in text])[::-1]
        text = [text[i] for i in idx_order]
        original_order = idx_order

    # Initialize storage for each layer
    if layer_indices is None:
        layer_indices = [-1]  # Default to last layer only
    elif isinstance(layer_indices, int):
        layer_indices = [layer_indices]  # Convert single int to list
    
    all_embeddings = {layer_idx: [] for layer_idx in layer_indices}
    all_topk_ids = []
    all_topk_probs = []
    all_text_indices = []
    all_token_positions = []

    num_batches = (len(text) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for i in trange(num_batches, desc=f"Computing embeddings [Batch Size = {batch_size}, Top-{top_k}, Layers: {layer_indices}]"):
            batch_text = text[i * batch_size : (i + 1) * batch_size]
            batch_inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            attention_mask = batch_inputs["attention_mask"]
            outputs = model(**batch_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_inds = torch.topk(probs, k=top_k, dim=-1)

            if last_position_only:
                # Extract embeddings from specified layers
                for layer_idx in layer_indices:
                    layer_hidden = hidden_states[layer_idx]
                    gathered_emb = layer_hidden[:, -1, :]
                    all_embeddings[layer_idx].append(gathered_emb.cpu())
                
                gathered_topk = topk_inds[:, -1, :]
                gathered_topk_probs = topk_probs[:, -1, :]

                all_topk_ids.append(gathered_topk.cpu())
                all_topk_probs.append(gathered_topk_probs.cpu())

                for j in range(len(batch_text)):
                    text_idx = i * batch_size + j
                    seq_len = attention_mask[j].sum().item()
                    all_text_indices.append(text_idx)
                    all_token_positions.append(seq_len - 1)
                
                del gathered_topk, gathered_topk_probs
            else:
                for j in range(hidden_states[-1].size(0)):
                    mask = attention_mask[j].bool()
                    
                    # Extract embeddings from specified layers
                    for layer_idx in layer_indices:
                        layer_hidden = hidden_states[layer_idx]
                        valid_embs = layer_hidden[j][mask]
                        all_embeddings[layer_idx].append(valid_embs.cpu())
                    
                    valid_topk = topk_inds[j][mask]
                    valid_topk_probs = topk_probs[j][mask]

                    all_topk_ids.append(valid_topk.cpu())
                    all_topk_probs.append(valid_topk_probs.cpu())

                    text_idx = i * batch_size + j
                    num_valid = mask.sum().item()
                    all_text_indices.extend([text_idx] * num_valid)
                    all_token_positions.extend(range(num_valid))

                del valid_topk, valid_topk_probs

            del batch_inputs, attention_mask, outputs, hidden_states, logits, topk_inds
            del probs, topk_probs
            torch.cuda.empty_cache()

    # Concatenate embeddings for each layer
    for layer_idx in layer_indices:
        all_embeddings[layer_idx] = torch.cat(all_embeddings[layer_idx], dim=0)
    
    all_topk_ids = torch.cat(all_topk_ids, dim=0)
    all_topk_probs = torch.cat(all_topk_probs, dim=0)
    all_text_indices = torch.tensor(all_text_indices, dtype=torch.long)
    all_token_positions = torch.tensor(all_token_positions, dtype=torch.long)

    if original_order is not None:
        inverse_order = np.empty_like(original_order)
        inverse_order[original_order] = np.arange(len(original_order))

        if last_position_only:
            for layer_idx in layer_indices:
                all_embeddings[layer_idx] = all_embeddings[layer_idx][inverse_order]
            all_topk_ids = all_topk_ids[inverse_order]
            all_topk_probs = all_topk_probs[inverse_order]
            all_token_positions = all_token_positions[inverse_order]
        else:
            inverse_order_tensor = torch.tensor(inverse_order, device='cpu')
            all_text_indices = inverse_order_tensor[all_text_indices]

    # If only one layer requested, unwrap for backward compatibility
    if len(layer_indices) == 1:
        embed_result = all_embeddings[layer_indices[0]]
    else:
        embed_result = all_embeddings

    results = {
        "embeddings": embed_result,
        "topk_ids": all_topk_ids,
        "topk_probs": all_topk_probs,
        "text_idx": all_text_indices,
        "token_pos": all_token_positions,
    }

    if is_str:
        del model
        torch.cuda.empty_cache()

    if save_to is not None:
        torch.save(results, save_to)
        print(f"Saved embeddings and metadata to {save_to}")

    return results