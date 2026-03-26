import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import information_geometry as ig

import argparse

MODEL_NAME = "google/gemma-3-4b-pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

model, tokenizer, vocab_dict, vocab_list, G = ig.load_model_and_vocab(MODEL_NAME, device=DEVICE)

arg = argparse.ArgumentParser()
arg.add_argument("--concept_name", type=str, default="verb_en_fr")
args = arg.parse_args()

concept_name = args.concept_name
base_path = "LLM_BASE_PATH" # Replace with the actual base path where data is stored
concept_path = os.path.join(base_path, concept_name)

with open(os.path.join(f"data/mapping_{concept_name}.json"), "r") as f:
    mapping = json.load(f)
mapping = ig.get_clean_mapping(mapping, vocab_dict)


train_primals0 = torch.load(os.path.join(concept_path, "train_embeddings0.pt")).to(DEVICE)
train_primals1 = torch.load(os.path.join(concept_path, "train_embeddings1.pt")).to(DEVICE)
test_primals0 = torch.load(os.path.join(concept_path, "test_embeddings0.pt")).to(DEVICE)
test_primals1 = torch.load(os.path.join(concept_path, "test_embeddings1.pt")).to(DEVICE)
train_duals0 = ig.primals_to_duals(train_primals0, G)
train_duals1 = ig.primals_to_duals(train_primals1, G)

primal_md = ig.get_MD(train_primals0, train_primals1)
dual_md = ig.get_MD(train_duals0, train_duals1)

directions = {
    "primal_md": primal_md,
    "dual_md": dual_md
}
direction_names = list(directions.keys())

torch.save(directions, os.path.join(concept_path, "directions.pt"))


with open(os.path.join(f"data/mapping_{concept_name}.json"), "r") as f:
    mapping = json.load(f)
mapping = ig.get_clean_mapping(mapping, vocab_dict)

primals_dict_list = []
for start_primal in tqdm(test_primals0):
    paths = {name: {'e': None, 'm': None} for name in direction_names}
    for name in direction_names:
        direction = directions[name]
        e_path = ig.e_steering(start_primal, direction, G, num_steps = 2000, step_size = 0.5,
                               use_tqdm=False, mapping=mapping, vocab_dict=vocab_dict)
        m_path = ig.m_steering(start_primal, direction, G, alpha = 5e-3, num_steps = 2000, step_size = 2,
                               use_tqdm=False, mapping=mapping, vocab_dict=vocab_dict)
        paths[name]['e'] = e_path
        paths[name]['m'] = m_path
    primals_dict_list.append(paths)

torch.save(primals_dict_list, os.path.join(concept_path, "test_steering_paths.pt"))