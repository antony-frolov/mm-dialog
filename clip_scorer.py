import os
from typing import Union, List, Dict, Any

from .dialog_dataset import DialogDataset
from .image_dataset import CLIPFeaturedImageDataset
from .featured_dataset import FeaturedDataset

import torch
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel, CLIPTokenizer
from tqdm.auto import tqdm


class CLIPFeaturedDialogDataset(DialogDataset, FeaturedDataset):
    """Class containing a dialog dataset with features for CLIP scoring.
    """
    def __init__(
        self, path2json: str, path2features: str,
        model: CLIPModel = None, tokenizer: CLIPTokenizer = None,
        batch_size: int = 256, device: Union[torch.device, str] = 'cpu'
    ):
        """
        Args:
            path2json (str):
                Path to JSON file containing the dataset.
            path2features (str):
                Path to directory containing feature vectors.
            model (CLIPModel, optional):
                Huggingface CLIP model. Defaults to None.
            tokenizer (CLIPTokenizer, optional):
                Huggingface CLIP tokenizer. Defaults to None.
            batch_size (int, optional):
                Batch size for generating feature vectors. Defaults to 256.
            device (Union[torch.device, str], optional):
                PyTorch device. Defaults to 'cpu'.
        """
        DialogDataset.__init__(self, dialogs=path2json)

        self.image_dataset = None

        FeaturedDataset.__init__(
            self, path2features=path2features,
            model=model, preprocessor=tokenizer, device=device,
            batch_size=batch_size
        )

    def __getitem__(self, idx: int) -> dict:
        return {
            'idx': idx, 'id': self.ids[idx],
            'utter': self.utters[idx],
            'context': self.contexts[idx],
            'path2features': self.feature_paths[idx],
            'features':
                torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None,
            'image_score': self.image_scores[idx],
            'image_idx': self.image_indices[idx],
            'image_dataset_tag': self.image_dataset_tags[idx],
            'image_dict':
                self.image_dataset[self.image_indices[idx]] if self.image_dataset is not None else None
        }

    def _save_json_item(self, idx: int, item: Dict[str, Any]) -> None:
        """Helper function to save item info into item dict for saving to JSON.

        Args:
            idx (int): Item index
            item (Dict[str, Any]): Item dict.
        """
        super(CLIPFeaturedDialogDataset, self)._save_json_item(idx, item)
        item['scorer_dicts']['clip'] = {
            'score': self.image_scores[idx],
            'image_idx': self.image_indices[idx],
            'image_dataset_tag': self.image_dataset_tags[idx]
        }

    def _init_attributes(self):
        """Helper function to initialize attributes when loading from json.
        """
        super(CLIPFeaturedDialogDataset, self)._init_attributes()
        self.image_scores, self.image_indices = [], []
        self.image_dataset_tags = []

    def _load_json_item(self, i: int, item: Dict[str, Any]) -> None:
        """Helper function to load item info from item dict when loading from JSON.

        Args:
            i (int): Item number.
            item (Dict[str, Any]): Item dict.
        """
        super(CLIPFeaturedDialogDataset, self)._load_json_item(i, item)

        if 'scorer_dicts' in item and 'clip' in item['scorer_dicts']:
            self.image_scores.append(item['scorer_dicts']['clip']['score'])
            self.image_indices.append(item['scorer_dicts']['clip']['image_idx'])
            self.image_dataset_tags.append(item['scorer_dicts']['clip']['image_dataset_tag'])
            return

        if 'image_score' in item:
            self.image_scores.append(item['image_score'])
        else:
            self.image_scores.append(None)

        if 'image_idx' in item:
            self.image_indices.append(item['image_idx'])
        else:
            self.image_indices.append(None)

        if 'image_dataset_tag' in item:
            self.image_dataset_tags.append(item['image_dataset_tag'])
        else:
            self.image_dataset_tags.append(None)

    def _generate_feature_vectors(
        self, indices: List[int],
        model: CLIPModel, tokenizer: CLIPTokenizer,
        batch_size: int
    ):
        """Helper function to generate utterance feature vectors using a CLIP model.

        Args:
            indices (List[int]): Indices of the dataset to generate feature vectors for.
            model (CLIPModel): Huggingface CLIP model.
            tokenizer (CLIPTokenizer): Huggingface CLIP tokenizer.
            batch_size (int): Batch size for generating feature vectors.
        """
        dataloader = DataLoader(
            Subset(self, indices), batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn
        )
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating feature vectors"):
                utters = batch['utters']
                inputs = tokenizer(text=utters, padding=True, truncation=True, return_tensors="pt").to(self.device)
                features = model.get_text_features(**inputs)
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
        Dict[str, Any]: Batch of data.
    """
    batch = {batch_key: [item[item_key] for item in data] for batch_key, item_key in [
                ('indices', 'idx'), ('ids', 'id'), ('utters', 'utter'), ('contexts', 'context')
            ]}
    for batch_key, item_key in [
        ('feature_paths', 'path2features'), ('features', 'features')
    ]:
        if item_key in data[0]:
            batch[batch_key] = [item[item_key] for item in data]
    return batch


class CLIPScorer:
    def __init__(
        self, image_dataset: CLIPFeaturedImageDataset, path2features: str = None,
        device: Union[str, torch.device] = 'cpu', max_workers: int = None
    ):
        """
        Args:
            image_dataset (CLIPFeaturedImageDataset):
                Image dataset to search in.
            path2features (str, optional):
                Path to saved feature vectors tensor.
            device (Union[str, torch.device], optional):
                PyTorch device. Defaults to 'cpu'.
            max_workers (int, optional):
                Max number of workers to spawn when loading feature vectors. Defaults to None.
        """
        self.image_dataset = image_dataset
        self.device = device

        self.image_dataset.load_feature_vectors(
            path=path2features, max_workers=max_workers
        )
        self.image_feature_vectors = self.image_dataset.get_feature_vectors().to(device)

    def score(
        self, dialog_dataset: CLIPFeaturedDialogDataset
    ):
        """Score each item in dialog dataset with distance to the closest image in given image dataset.

        Args:
            dialog_dataset (CLIPFeaturedDialogDataset): Dataset to score.
        """        """"""
        dialog_dataset.image_dataset = self.image_dataset

        for idx, path in enumerate(tqdm(dialog_dataset.feature_paths, desc="Finding closest images")):
            text_feature_vector = torch.load(path).to(self.device)
            similarities = self.image_feature_vectors.matmul(text_feature_vector)
            sim, image_idx = torch.max(similarities, dim=0)
            dialog_dataset.image_scores[idx] = sim.item()
            dialog_dataset.image_indices[idx] = image_idx.item()
            dialog_dataset.image_dataset_tags[idx] = self.image_dataset.tag
