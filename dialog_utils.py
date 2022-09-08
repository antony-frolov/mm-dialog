import os
from functools import partial
import json
from time import sleep

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

import ipywidgets as widgets
from IPython.display import display, Image, Markdown
from superintendent import ClassLabeller


class DialogDataset(Dataset):
    def __init__(
        self, dialogs, min_length=0, max_length=np.inf,
        path2features=None, model=None, tokenizer=None, device='cpu',
        indices=None
    ):
        self.indices = indices
        if isinstance(dialogs, str) and dialogs.endswith('.json'):
            self.from_json(dialogs, indices=self.indices)
        else:
            if self.indices is not None:
                self.dialogs = [self.dialogs[idx] for idx in self.indices]
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

        self.image_dataset = None
        if not hasattr(self, 'image_scores'):
            self.image_scores = [None]*len(self)
        if not hasattr(self, 'image_indices'):
            self.image_indices = [None]*len(self)

    def __getitem__(self, idx):
        item = {'idx': idx, 'utter': self.utters[idx],
                'context': self.contexts[idx],
                'image_like': self.image_like_flags[idx]}
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        if hasattr(self, 'image_dataset'):
            item['image_score'] = self.image_scores[idx]
            item['image_idx'] = self.image_indices[idx]
            item['image_dict'] = self.image_dataset[self.image_indices[idx]]
        return item

    def __len__(self):
        return len(self.utters)

    # def test_dataset(self, n_splits):
    #     for _ in range(n_splits):
    #         train_indices, test_indices = train_test_split(np.arange(len(self)), test_size=0.2)

    def label_samples(self, indices, path=None, skip_annotated=False, use_images=True):
        def save_labels(widget, indices, path):
            for idx, label in zip(indices, widget.new_labels):
                if label is not None:
                    self.image_like_flags[idx] = label
            if path:
                self.to_json(path)

        def display_func(args):
            utter, context, image_path = args

            formatted_context = '\n'.join(f'    {utter}' for utter in context)
            display(Markdown(f"""
                **Context:**
                {formatted_context}
                """))

            display(Markdown(f"""
                **Utter:**
                    {utter}
                """))

            display(Image(filename=image_path, width=300, height=300))

        if skip_annotated:
            indices = list(filter(lambda i: self.image_like_flags[i] is None, indices))
        if use_images:
            widget = ClassLabeller(
                features=zip(
                    (self.utters[idx] for idx in indices),
                    (self.contexts[idx] for idx in indices),
                    (self.image_dataset[self.image_indices[idx]]['path2image'] for idx in indices)),
                options=['Yes', 'No'],
                display_func=display_func
            )

            display(widget)
            # while any(label is None for label in widget.new_labels):
            #     save_labels(widget, indices, path)
            #     sleep(1)
            # save_labels(widget, indices, path)
        else:
            for idx in indices:
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
            item = {
                'context': self.contexts[idx], 'utter': self.utters[idx],
                'image_like': self.image_like_flags[idx],
                'image_idx': self.image_indices[idx],
                'image_score': self.image_scores[idx]
            }
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f, indent=4)

    def from_json(self, path, indices=None):
        self.dialogs, self.contexts, self.utters = [], [], []
        self.image_like_flags, self.image_indices, self.image_scores = [], [], []
        with open(path, 'r') as f:
            items = list(json.load(f))
        if indices is not None:
            items = [items[idx] for idx in indices]
        for item in items:
            self.contexts.append(item['context'])
            self.utters.append(item['utter'])
            self.dialogs.append(item['context'] + [item['utter']])
            if 'image_like' in item:
                self.image_like_flags.append(item['image_like'])
            if 'image_idx' in item:
                self.image_indices.append(item['image_idx'])
            if 'image_score' in item:
                self.image_scores.append(item['image_score'])
        if not self.image_like_flags:
            self.image_like_flags = [None] * len(self)
        if not self.image_indices:
            self.image_indices = [None] * len(self)
        if not self.image_scores:
            self.image_scores = [None] * len(self)

    # def load_feature_vectors(self, path='text_feature_vectors.pt'):
    #     self.feature_vectors = torch.load(path)

    def get_feature_vectors(self, max_workers=16):
        feature_vectors = [None] * len(self)
        thread_map(
            partial(self._get_feature_vector, feature_vectors=feature_vectors),
            list(range(len(self))), max_workers=max_workers,
            desc="Loading feature vectors"
        )
        return torch.stack(feature_vectors)

    def _get_feature_vector(self, idx, feature_vectors):
        feature_vectors[idx] = torch.load(self.feature_paths[idx])

    def find_closest_images(self, image_dataset, device='cpu', parallel=True, max_workers=None):
        self.image_dataset = image_dataset
        image_feature_vectors = self.image_dataset.get_feature_vectors(
            parallel=parallel, max_workers=max_workers
        ).to(device)

        for idx, path in enumerate(tqdm(self.feature_paths, desc="Finding closest images")):
            text_feature_vector = torch.load(path).to(device)
            similarities = image_feature_vectors.matmul(text_feature_vector)
            sim, image_idx = torch.max(similarities, dim=0)
            self.image_scores[idx] = sim.item()
            self.image_indices[idx] = image_idx.item()

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
    batch = {batch_key: [item[item_key] for item in data] for batch_key, item_key in [
                 ('indices', 'idx'), ('utters', 'utter'), ('contexts', 'context')
            ]}
    for batch_key, item_key in [
        ('feature_paths', 'path2features'), ('features', 'features')
    ]:
        if item_key in data[0]:
            batch[batch_key] = [item[item_key] for item in data]
    return batch
