import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

def get_clip_text_embeddings(texts, clip_model, clip_processor, batch_size=32):
    tmpt = clip_model.logit_scale.exp()
    tmpt_scale = tmpt.sqrt()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = clip_processor(text=batch_texts, return_tensors="pt", padding=True).to(clip_model.device)
        with torch.no_grad():
            emb = clip_model.get_text_features(**inputs)
            emb = F.normalize(emb['pooler_output'], dim=-1) * tmpt_scale
        all_emb.append(emb)
    all_emb = torch.cat(all_emb, dim=0)
    if all_emb.shape[0] == 1:
        return all_emb[0]
    return all_emb

def get_clip_image_embeddings(images, clip_model, clip_processor, batch_size=32):
    tmpt = clip_model.logit_scale.exp()
    tmpt_scale = tmpt.sqrt()
    all_emb = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(clip_model.device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = F.normalize(emb['pooler_output'], dim=-1) * tmpt_scale
        all_emb.append(emb)
    all_emb = torch.cat(all_emb, dim=0)
    if all_emb.shape[0] == 1:
        return all_emb[0]
    return all_emb



class ImageDataset():
    def __init__(self, data_path, concept_dict, num_images=5):
        self.data_path = data_path
        self.samples = []

        for shape in concept_dict['shape']:
            for color in concept_dict['color']:
                for number in concept_dict['number']:
                    folder_name = f"{shape}_{color}_{number}"
                    folder_path = os.path.join(self.data_path, folder_name)
                    
                    if os.path.exists(folder_path):
                        for img_idx in range(num_images):
                            img_path = os.path.join(folder_path, f"image_{img_idx:03d}.png")
                            if os.path.exists(img_path):
                                self.samples.append({
                                    'path': str(img_path),
                                    'shape': shape,
                                    'color': color,
                                    'number': number,
                                    'image_idx': img_idx,
                                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        return {
            'image': image,
            'shape': sample['shape'],
            'color': sample['color'],
            'number': sample['number'],
            'image_idx': sample['image_idx'],
        }


class ComboImageDataset():
    def __init__(self, data_path, concept_dict, num_images=3):
        self.data_path = data_path
        self.samples = []

        for color1 in concept_dict['color']:
            for shape1 in concept_dict['shape']:
                for color2 in concept_dict['color']:
                    for shape2 in concept_dict['shape']:
                        folder_name = f"{color1}_{shape1}_{color2}_{shape2}"
                        folder_path = os.path.join(self.data_path, f"combo_{folder_name}")
                                
                        if os.path.exists(folder_path):
                            for img_idx in range(num_images):
                                img_path = os.path.join(folder_path, f"image_{img_idx:03d}.png")
                                if os.path.exists(img_path):
                                    self.samples.append({
                                        'path': str(img_path),
                                        'shape1': shape1,
                                        'color1': color1,
                                        'shape2': shape2,
                                        'color2': color2,
                                        'image_idx': img_idx,
                                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        
        return {
            'image': image,
            'shape1': sample['shape1'],
            'color1': sample['color1'],
            'shape2': sample['shape2'],
            'color2': sample['color2'],
            'image_idx': sample['image_idx'],
        }



def load_color_shape_image(data_path, concept_dict, clip_model, clip_processor, image_num_images=5, combo_num_images=3):
    image_dataset = ImageDataset(data_path, concept_dict, num_images=image_num_images)
    print(f"\nTotal samples: {len(image_dataset.samples)}")
    print(image_dataset[0])

    G = get_clip_image_embeddings([data['image'] for data in image_dataset],
                             clip_model, clip_processor)

    combo_dataset = ComboImageDataset(data_path, concept_dict, num_images=combo_num_images)

    print(f"\nTotal samples: {len(combo_dataset.samples)}")
    print(combo_dataset[0])

    combo_G = get_clip_image_embeddings([data['image'] for data in combo_dataset],
                                   clip_model, clip_processor)

    G = torch.cat([G, combo_G], dim=0)

    print(f"ImageDataset embeddings shape: {G.shape}")
    print(f"ComboImageDataset embeddings shape: {combo_G.shape}")

    return G, image_dataset, combo_dataset