from functools import partial
import os
import json
from typing import List, Union, Tuple, Dict, Any

from .featured_dataset import FeaturedDataset

import requests
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPFeatureExtractor
import pandas as pd


class FeaturedImageDataset(Dataset, FeaturedDataset):
    """A class containing an image dataset and its associated metadata.
    """
    def __init__(
        self, photos_df: Union[str, pd.DataFrame], path2images: str = None, width: int = 640,
        drop_missing: bool = True, path2features: str = None, model: CLIPModel = None,
        feature_extractor: CLIPFeatureExtractor = None, device: Union[str, torch.device] = 'cpu',
        check_image_integrity: bool = True, parallel: bool = False, max_workers: int = None,
        indices: List[int] = None, batch_size: int = 128, tag: str = None
    ) -> None:
        """
        Args:
            photos_df (Union[str, pd.DataFrame]):
                Either a path to a json file or a Pandas DataFrame containing the dataset.
            path2images (str, optional):
                Path to directory containing saved images. Defaults to None.
            width (int, optional):
                Width of a downloaded image. Defaults to 640.
            drop_missing (bool, optional):
                If True missing images are dropped. Defaults to True.
            path2features (str, optional):
                Path to directory containing saved features. Defaults to None.
            model (CLIPModel, optional):
                Huggingface CLIP model. Defaults to None.
            feature_extractor (CLIPFeatureExtractor, optional):
                Huggingface CLIP feature extractor. Defaults to None.
            device (Union[str, torch.device], optional):
                PyTorch device. Defaults to 'cpu'.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
            parallel (bool, optional):
                If True loads images in parallel. Defaults to False.
            max_workers (int, optional):
                Max number of workers to spawn. Defaults to None. Defaults to None.
            indices (List[int], optional):
                Indices of DataFrame or json to include in the dataset. Defaults to None.
            batch_size (int, optional):
                Batch size for generating feature vectors. Defaults to 128.
            tag (str, optional):
                Tag to add to the dataset. Defaults to None.
        """
        self.photos_df = photos_df
        self.indices = indices
        self.tag = tag

        if isinstance(self.photos_df, str) and self.photos_df.endswith('json'):
            self.from_json(self.photos_df, indices=indices)
        else:
            if self.indices is not None:
                self.photos_df = self.photos_df.iloc[self.indices]
            self.ids = self.photos_df.photo_id.to_list()
            self.urls = [
                f"{url}?w={width}" for url in self.photos_df.photo_image_url]
            self.descriptions = self.photos_df.photo_description.where(
                self.photos_df.photo_description.notna(), None
            ).tolist()
            self.ai_descriptions = self.photos_df.ai_description.where(
                self.photos_df.ai_description.notna(), None
            ).tolist()

        self.image_paths = [None] * len(self)

        self.path2images = path2images

        self.check_image_integrity = check_image_integrity

        if self.path2images is not None:
            self._load_images(
                drop_missing=drop_missing,
                parallel=parallel, max_workers=max_workers,
                check_image_integrity=check_image_integrity
            )

        FeaturedDataset.__init__(
            self, path2features=path2features,
            model=model, preprocessor=feature_extractor,
            batch_size=batch_size, device=device
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._get_item(idx)
        if self.path2images is not None:
            item['path2image'] = self.image_paths[idx]
            item['image'] = Image.open(
                self.image_paths[idx]) if self.image_paths[idx] is not None else None
        if self.path2features is not None:
            item['path2features'] = self.feature_paths[idx]
            item['features'] = torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None
        return item

    def _get_item(self, idx: int) -> Dict[str, str]:
        return {'idx': idx, 'id': self.ids[idx], 'url': self.urls[idx],
                'description': self.descriptions[idx],
                'ai_description': self.ai_descriptions[idx]}

    def __len__(self) -> int:
        return len(self.ids)

    def to_json(self, path: str) -> None:
        """Save dataset to json file.

        Args:
            path (str): Path to json file.
        """
        items = []
        for idx in tqdm(range(len(self)), desc="Saving to json"):
            description = self.descriptions[idx]
            ai_description = self.ai_descriptions[idx]
            item = {
                'id': self.ids[idx], 'url': self.urls[idx],
                'description': description if description is not None else None,
                'ai_description': ai_description if ai_description is not None else None,
            }
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f, indent=4)

    def from_json(self, path: str, indices: List[int] = None):
        """Load dataset from json file.

        Args:
            path (str): Path to json file.
            indices (List[int], optional):
                Indices of the dataset to load. Defaults to None.
        """
        self.ids, self.urls = [], []
        self.descriptions, self.ai_descriptions = [], []
        with open(path, 'r') as f:
            items = list(json.load(f))
        if indices is not None:
            items = [items[idx] for idx in indices]
        for item in tqdm(items, desc='Reading from json'):
            self.ids.append(item['id'])
            self.urls.append(item['url'])
            self.descriptions.append(item['description'])
            self.ai_descriptions.append(item['ai_description'])

    def _load_images(
        self, drop_missing: bool = False, max_workers: int = None,
        check_image_integrity: bool = True, parallel: bool = False
    ) -> None:
        """Helper function to load image paths from disk and download missing from urls.

        Args:
            path2dir (str):
                Path to directory containing the images.
            drop_missing (bool, optional):
                If True missing images are dropped. Defaults to False.
            max_workers (int, optional):
                Max number of workers to spawn. Defaults to None.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
            parallel (bool, optional):
                If True uses multiprocessing to load image paths. Defaults to False.
        """
        if not parallel:
            for idx in tqdm(range(len(self)), total=len(self),
                            desc="Loading image paths"):
                self._load_image(idx, check_image_integrity=check_image_integrity)
        else:
            thread_map(
                partial(self._load_image, check_image_integrity=check_image_integrity),
                range(len(self)), total=len(self), max_workers=max_workers,
                desc="Loading image paths"
            )
        print(f"Failed to load {self.image_paths.count(None)}/{len(self)} images")

        if drop_missing:
            missing_indices = [idx for idx, path in enumerate(self.image_paths) if path is None]
            for idx in sorted(missing_indices, reverse=True):
                for lst in [self.ids, self.urls, self.descriptions, self.ai_descriptions, self.image_paths]:
                    lst.pop(idx)

    def _load_image(
        self, idx: int, check_image_integrity: bool = True
    ) -> None:
        """Helper function to load one image.

        Args:
            args (Tuple):
                Tuple containing index, id and url.
            path2dir (str):
                Path to directory containing the images.
            check_image_integrity (bool, optional):
                If True each images is opened when loading. Defaults to True.
        """
        path = f"{os.path.join(self.path2images, self.ids[idx])}.jpg"
        if not os.path.exists(path):
            try:
                self._download_image(self.urls[idx]).save(path)
            except Exception:
                # print(f"Can't load image {id} from {url}")
                return
        if check_image_integrity:
            try:
                Image.open(path)
            except Exception:
                # print(f"Corrupted image file {path}")
                return
        self.image_paths[idx] = path

    def _download_image(self, url: str) -> None:
        """Helper function to load one image from its url

        Args:
            url (str): Url to load image from.
        """
        return Image.open(requests.get(url, stream=True).raw).convert('RGB')

    def _generate_feature_vectors(
        self, indices: List[int],
        model: CLIPModel, feature_extractor: CLIPFeatureExtractor,
        batch_size: int
    ) -> None:
        """Helper function to generate feature vectors.

        Args:
            indices (List[int]):
                Indices of dataset to generate feature vectors for.
            model (CLIPModel):
                Huggingface CLIP model.
            feature_extractor (CLIPFeatureExtractor):
                Huggingface CLIP feature extractor.
            path2dir (str):
                Path to directory containing feature tensors.
            batch_size (int):
                Batch size for generating feature vectors.
        """
        dataloader = DataLoader(
            Subset(self, indices), batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn
        )
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                images = batch['images']
                inputs = feature_extractor(images=images, return_tensors="pt").to(self.device)
                features = model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                for vector, idx, id in zip(features, batch['indices'], batch['ids']):
                    path = f"{os.path.join(self.path2features, id)}.pt"
                    torch.save(vector.clone().to('cpu'), path)
                    self.feature_paths[idx] = path


def collate_fn(data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Function to collate lists of data in a batch.

    Args:
        data (Dict[str, List[Any]]): Lists of data.

    Returns:
        Dict[str, Any]: Batach of data.
    """
    batch = {batch_key: [item[item_key] for item in data] for batch_key, item_key in [
                 ('indices', 'idx'), ('ids', 'id'), ('urls', 'url'),
                 ('descriptions', 'description'), ('ai_descriptions', 'ai_description')
            ]}
    for batch_key, item_key in [
        ('image_paths', 'path2image'),
        ('images', 'image'), ('feature_paths', 'path2features'),
        ('features', 'features')
    ]:
        if item_key in data[0]:
            batch[batch_key] = [item[item_key] for item in data]
    return batch
