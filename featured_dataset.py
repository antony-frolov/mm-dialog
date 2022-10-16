from functools import partial
import os
from abc import ABC, abstractmethod

from typing import List, Callable, Union

import torch
from transformers import CLIPModel
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map


class FeaturedDataset(ABC):
    """Abstract class for datasets with feature vectors.
    """
    def __init__(
        self, path2features: str,
        model: CLIPModel, preprocessor: Callable,
        batch_size: int, device: Union[torch.device, str]
    ):
        """
        Args:
            path2features (str): Path to directory containing saved features.
            model (CLIPModel): Huggingface CLIP model.
            preprocessor (Callable): Preprocessing function.
            batch_size (int): Batch size for generating feature vectors.
            device (Union[torch.device, str]): PyTorch device.

        Raises:
            ValueError: If missing model or tokenizer to generate missing feature vectors.
        """
        self.path2features = path2features
        self.feature_paths = [None] * len(self)

        self.device = device

        missing_indices = self._load_feature_vector_paths()

        if missing_indices:
            if model is None or preprocessor is None:
                raise ValueError("Missing feature vectors, but no model or preprocessor for generation")
            self._generate_feature_vectors(
                missing_indices, model, preprocessor, batch_size
            )
        self.feature_vectors = None

    def _load_feature_vector_paths(self) -> List[int]:
        """Helper function to load feature vector paths.
        """
        missing_indices = []
        for idx, id in enumerate(tqdm(self.ids, desc="Loading feature vector paths")):
            path = f"{os.path.join(self.path2features, id)}.pt"
            if os.path.exists(path):
                self.feature_paths[idx] = path
            else:
                missing_indices.append(idx)

        return missing_indices

    @abstractmethod
    def _generate_feature_vectors(
        self, indices: List[int],
        model: CLIPModel, preprocessor: Callable,
        batch_size: int
    ) -> None:
        """Abstract method to generate feature vectors."""
        pass

    def get_feature_vectors(self) -> torch.Tensor:
        """Get feature vectors if already loaded.

        Raises:
            ValueError: If feature vectors are not loaded.

        Returns:
            torch.Tensor: Tensor of feature vectors.
        """
        if self.feature_vectors is not None:
            return self.feature_vectors
        else:
            raise ValueError("Missing feature vectors. Load feature vectors first.")

    def save_feature_vectors(self, path: str) -> None:
        """Save loaded feature vectors to disk.

        Raises:
            ValueError: If feature vectors are not loaded.

        Args:
            path (str): Path to save feature vectors to.
        """
        if self.feature_vectors is not None:
            torch.save(self.feature_vectors, path)
        else:
            raise ValueError("Missing feature vectors. Load feature vectors first.")

    def load_feature_vectors(
        self, path: str = None, max_workers: int = 16, force: bool = False
    ) -> None:
        """Load feature vectors from disk.

        Args:
            path (str, optional):
                Path to feature vectors tensor. If None loads from feature paths. Defaults to None.
            max_workers (int, optional):
                Maximum number of workers to use. Defaults to 16.
            force (bool, optional):
                If True, reloads existing feature vectors. Defaults to False.
        """
        if self.feature_vectors is not None and not force:
            return

        if path is not None:
            self.feature_vectors = torch.load(path)
        else:
            feature_vectors = [None] * len(self)
            thread_map(
                partial(self._load_feature_vector, feature_vectors=feature_vectors),
                list(range(len(self))), max_workers=max_workers,
                desc="Loading feature vector paths"
            )
            self.feature_vectors = torch.stack(feature_vectors)

    def _load_feature_vector(self, idx: int, feature_vectors: List[torch.Tensor]) -> None:
        """Helper function to load one feature vector from disk.

        Args:
            idx (int): Item index.
            feature_vectors (List[torch.Tensor]): List of all feature vectors.
        """
        feature_vectors[idx] = torch.load(self.feature_paths[idx])
