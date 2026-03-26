import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_vocab(model_name, device):
    print(f"\n## Loading Tokenizer and Model: {model_name} ##")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    vocab_dict = tokenizer.get_vocab()
    vocab_size = max(vocab_dict.values()) + 1
    vocab_list = [None] * vocab_size
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    G = model.lm_head.weight[:vocab_size].detach().to(torch.float32).to(device)
    print(G.shape)

    return model, tokenizer, vocab_dict, vocab_list, G