from functools import partial
from multiprocessing import set_forkserver_preload
import os

import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.contrib.concurrent import thread_map, process_map
from tqdm.auto import tqdm


class PhotosDataset(Dataset):
    def __init__(self, photos_df, path2images=None, width=640, drop_missing=True,
                 path2features=None, model=None, feature_extractor=None, device='cpu'):
        self.photos_df = photos_df

        # f"?ixid=2yJhcHBfaWQiOjEyMDd9&fm=jpg&w={width}&fit={fit}"

        self.ids = self.photos_df.photo_id.to_list()
        self.urls = [
            f"{url}?w={width}" for url in self.photos_df.photo_image_url]
        self.descriptions = self.photos_df.photo_description.to_list()
        self.ai_descriptions = self.photos_df.ai_description.to_list()
        self.image_paths = [None] * len(self)
        self.feature_paths = [None] * len(self)

        self.path2images = path2images
        self.path2features = path2features

        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device

        if self.path2images is not None:
            self._load_images(self.path2images, drop_missing=drop_missing)
        if self.path2features is not None:
            self._load_feature_vectors(
                self.path2features, model=self.model,
                feature_extractor=self.feature_extractor, device=device
            )

    def __getitem__(self, idx):
        item = {'idx': idx, 'id': self.ids[idx], 'url': self.urls[idx],
                'description': self.descriptions[idx],
                'ai_description': self.ai_descriptions[idx]}
        if self.path2images is not None:
            item['path2image'] = self.image_paths[idx]
            item['image'] = Image.open(
                self.image_paths[idx]) if self.image_paths[idx] is not None else None
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        return item

    def __len__(self):
        return len(self.ids)

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

    def _load_images(self, path2dir, drop_missing=False):
        missing_idx = []
        for idx, id in enumerate(self.ids):
            path = f"{os.path.join(path2dir, id)}.jpg"
            if os.path.exists(path):
                self.image_paths[idx] = path
            else:
                missing_idx.append(idx)

        if missing_idx:
            thread_map(
                partial(self._download_image, path2dir=path2dir),
                zip(missing_idx, [self.ids[i] for i in missing_idx],
                    [self.urls[i] for i in missing_idx]),
                total=len(missing_idx), max_workers=128,
                desc="Downloading images"
            )

        if drop_missing:
            missing_idx = [idx for idx, path in enumerate(self.image_paths) if path is None]
            for idx in sorted(missing_idx, reverse=True):
                for lst in [self.ids, self.urls, self.descriptions, self.ai_descriptions, self.image_paths]:
                    lst.pop(idx)

    def _download_image(self, args, path2dir):
        idx, id, url = args
        path = f"{os.path.join(path2dir, id)}.jpg"
        try:
            image = Image.open(requests.get(url, stream=True).raw)
        except Exception:
            print(f"Can't load image {id} from {url}")
            return
        image.convert('RGB').save(path)
        self.image_paths[idx] = path

    def _load_feature_vectors(self, path2dir, model=None, feature_extractor=None, device='cpu'):
        missing_indices = []
        for idx, id in enumerate(self.ids):
            path = f"{os.path.join(path2dir, id)}.pt"
            if os.path.exists(path):
                self.feature_paths[idx] = path
            else:
                missing_indices.append(idx)

        if missing_indices:
            if model is None or feature_extractor is None:
                raise ValueError("Missing feature vectors, but no model or feature extractor for generation")
            self._generate_feature_vectors(
                missing_indices, model, feature_extractor, path2dir, device
            )

    def _generate_feature_vectors(self, indices, model, feature_extractor, path2dir, device='cpu'):
        dataloader = DataLoader(Subset(self, indices), batch_size=256, shuffle=False, collate_fn=collate_fn)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                images = batch['images']
                inputs = feature_extractor(images=images, return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                for vector, idx, id in zip(features, batch['indices'], batch['ids']):
                    path = f"{os.path.join(path2dir, id)}.pt"
                    torch.save(vector, path)
                    self.feature_paths[idx] = path


def collate_fn(data):
    batch = {'indices': [item['idx'] for item in data],
             'ids': [item['id'] for item in data],
             'urls': [item['url'] for item in data],
             'descriptions': [item['description'] for item in data],
             'ai_descriptions': [item['ai_description'] for item in data]}
    if 'path2image' in data[0]:
        batch['image_paths'] = [item['path2image'] for item in data]
    if 'image' in data[0]:
        batch['images'] = [item['image'] for item in data]
    if 'path2features' in data[0]:
        batch['feature_paths'] = [item['path2features'] for item in data]
    if 'features' in data[0]:
        batch['features'] = [item['features'] for item in data]
    return batch


# def get_image_features(photos_dataloader, model, feature_extractor,
#                        save2path='image_feature_vectors.pt', device='cpu'):
#     feature_batches_list = []
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(photos_dataloader):
#             if 'images' in batch:
#                 images = batch['images']
#             else:
#                 images = [Image.open(requests.get(url, stream=True).raw) for url in batch['urls']]
#             image_inputs = feature_extractor(images=images, return_tensors="pt").to(device)
#             image_features = model.get_image_features(**image_inputs)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             feature_batches_list.append(image_features.to('cpu'))
#     feature_vectors = torch.concat(feature_batches_list, dim=0)
#     torch.save(feature_vectors, save2path)
#     return feature_vectors


def get_image_dataloader(*args, **kwargs):
    return DataLoader(*args, collate_fn=collate_fn, **kwargs)
