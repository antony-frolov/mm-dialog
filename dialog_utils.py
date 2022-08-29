import os
from functools import partial
import json

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from IPython.display import clear_output


class DialogDataset(Dataset):
    def __init__(self, dialogs, min_length=0, max_length=np.inf,
                 path2features=None, model=None, tokenizer=None, device='cpu'):
        if isinstance(dialogs, str) and dialogs.endswith('json'):
            self.from_json(dialogs)
        else:
            self.dialogs = dialogs
            self.utters = []
            self.contexts = []

            for dialog in tqdm(self.dialogs, desc="Loading dialogs"):
                context = []
                for utter in dialog:
                    utter = utter.strip()
                    n_words = sum(len(w) > 2 for w in utter.split())
                    if context and min_length <= n_words <= max_length:
                        self.utters.append(utter)
                        self.contexts.append(context.copy())
                    context.append(utter)

            self.image_like_flags = [None] * len(self)

        self.path2features = path2features

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.feature_paths = [None] * len(self)

        if self.path2features is not None:
            self._load_feature_vectors(
                self.path2features, model=self.model,
                tokenizer=self.tokenizer, device=device
            )

    def __getitem__(self, idx):
        item = {'idx': idx, 'utter': self.utters[idx],
                'context': self.contexts[idx],
                'image_like': self.image_like_flags[idx]}
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        if hasattr(self, 'image_dataset'):
            item['image_score'] = self.image_scores[idx]
            item['image_index'] = self.image_indices[idx]
            item['image_dict'] = self.image_dataset[self.image_indices[idx]]
        return item

    def __len__(self):
        return len(self.utters)

    def annotate_utters(self, n=None, path=None, skip_annotated=False):
        n = n if n is not None else len(self)
        for idx in range(n):
            if skip_annotated and self.image_like_flags[idx] is not None:
                continue
            print(*self.contexts[idx], sep='\n')
            print(f'Utter: "{self.utters[idx]}"')
            image_like = input('Is image like? (y/n)').strip() == 'y'
            self.image_like_flags[idx] = image_like
            if path:
                self.to_json(path)
            clear_output(wait=True)

    def to_json(self, path):
        items = []
        for idx in range(len(self)):
            item = {'context': self.contexts[idx],
                    'utter': self.utters[idx],
                    'image_like': self.image_like_flags[idx]}
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f)

    def from_json(self, path):
        self.dialogs, self.contexts, self.utters, self.image_like_flags = [], [], [], []
        with open(path, 'r') as f:
            for item in json.load(f):
                self.contexts.append(item['context'])
                self.utters.append(item['utter'])
                self.dialogs.append(item['context'] + [item['utter']])
                self.image_like_flags.append(item['image_like'])

    # def load_feature_vectors(self, path='text_feature_vectors.pt'):
    #     self.feature_vectors = torch.load(path)

    def get_feature_vectors(self):
        feature_vectors = [None] * len(self)
        thread_map(
            partial(self._get_feature_vector, feature_vectors=feature_vectors),
            list(range(len(self))), max_workers=16,
            desc="Loading feature vectors"
        )
        return torch.stack(feature_vectors)

    def _get_feature_vector(self, idx, feature_vectors):
        feature_vectors[idx] = torch.load(self.feature_paths[idx])

    def find_closest_images(self, image_dataset, device='cpu'):
        self.image_dataset = image_dataset
        self.image_scores, self.image_indices = [], []

        image_feature_vectors = self.image_dataset.get_feature_vectors().to(device)

        for path in tqdm(self.feature_paths):
            text_feature_vector = torch.load(path).to(device)
            similarities = image_feature_vectors.matmul(text_feature_vector)
            sim, idx = torch.max(similarities, dim=0)
            self.image_scores.append(sim.item())
            self.image_indices.append(idx.item())

    def _load_feature_vectors(self, path2dir, model=None, tokenizer=None, device='cpu'):
        missing_indices = []
        for idx in range(len(self)):
            path = f"{os.path.join(path2dir, str(idx))}.pt"
            if os.path.exists(path):
                self.feature_paths[idx] = path
            else:
                missing_indices.append(idx)

        if missing_indices:
            if model is None or tokenizer is None:
                raise ValueError("Missing feature vectors, but no model or tokenizer for generation")
            self._generate_feature_vectors(
                missing_indices, model, tokenizer, path2dir, device
            )

    def _generate_feature_vectors(self, indices, model, tokenizer, path2dir, device='cpu'):
        dataloader = DataLoader(Subset(self, indices), batch_size=256, shuffle=False, collate_fn=collate_fn)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                utters = batch['utters']
                inputs = tokenizer(text=utters, padding=True, truncation=True, return_tensors="pt").to(device)
                features = model.get_text_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                for vector, idx in zip(features, batch['indices']):
                    path = f"{os.path.join(path2dir, str(idx))}.pt"
                    torch.save(vector.to('cpu'), path)
                    self.feature_paths[idx] = path


def collate_fn(data):
    batch = {'indices': [item['idx'] for item in data],
             'utters': [data['utter'] for data in data],
             'contexts': [data['context'] for data in data]}
    if 'path2features' in data[0]:
        batch['feature_paths'] = [item['path2features'] for item in data]
    if 'features' in data[0]:
        batch['features'] = [item['features'] for item in data]
    return batch


# def get_text_features(photos_dataloader, model, tokenizer, save2path='text_feature_vectors.pt', device='cpu'):
#     feature_batches_list = []
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(photos_dataloader):
#             text_inputs = tokenizer(text=batch['utters'], padding=True,
#                                     truncation=True, return_tensors="pt").to(device)
#             text_features = model.get_text_features(**text_inputs)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
#             feature_batches_list.append(text_features.to('cpu'))
#     feature_vectors = torch.concat(feature_batches_list, dim=0)
#     torch.save(feature_vectors, save2path)
#     return feature_vectors


def get_text_dataloader(*args, **kwargs):
    return DataLoader(*args, collate_fn=collate_fn, **kwargs)
