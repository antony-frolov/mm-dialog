import json
from typing import Union, List, Dict, Any

from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm


class DialogDataset(Dataset):
    """A class containing a dialog dataset with optional scores.

    Attributes:
        image_scores (List[float]): CLIP generated scores.
        image_indices (List[int]): Indices of closest images in image dataset.
    """
    def __init__(
        self, dialogs: Union[List[str], str],
        min_length: int = 0, max_length: int = np.inf,
        indices: List[int] = None
    ):
        """
        Args:
            dialogs (Union[pd.Dataset, str]):
                Either a list of dialogs or a json file containing the dataset.
            min_length (int, optional): Min length of a sentence. Defaults to 0.
            max_length (int, optional): Max length of a sentence. Defaults to np.inf.
            path2features (str, optional):
                Path to directory containing saved features. Defaults to None.
            indices (List[int], optional):
                Indices of dialog list or json to include in dataset. Defaults to None.

        Raises:
            ValueError: If dialogs is not a list or a path to a json file.
        """
        self.indices = indices
        if isinstance(dialogs, str):
            if dialogs.endswith('.json'):
                self.from_json(dialogs, indices=self.indices)
            else:
                raise ValueError('Path leads to invalid file type.')
        elif isinstance(dialogs, list):
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

            self.ids = list(range(len(self)))
            self.image_like_flags = [None] * len(self)
            self.scorer_dicts = [{} for _ in range(len(self))]

        else:
            raise ValueError('Invalid dialogs type.')

    def __getitem__(self, idx: int) -> dict:
        return {'idx': idx, 'id': self.ids[idx],
                'utter': self.utters[idx],
                'context': self.contexts[idx],
                'image_like': self.image_like_flags[idx],
                'scorer_dicts': self.scorer_dicts[idx]}

    def __len__(self) -> int:
        return len(self.utters)

    def to_json(self, path: str) -> None:
        """Save dataset to JSON file.

        Args:
            path (str): Path to JSON file.
        """
        items = []
        for idx in range(len(self)):
            item = {}
            self._save_json_item(idx, item)
            items.append(item)
        with open(path, 'w') as f:
            json.dump(items, f, indent=4)

    def _save_json_item(self, idx: int, item: Dict[str, Any]) -> None:
        """Helper function to save item info into item dict for saving to JSON.

        Args:
            idx (int): Item index
            item (Dict[str, Any]): Item dict.
        """
        item |= {
            'id': self.ids[idx],
            'context': self.contexts[idx],
            'utter': self.utters[idx],
            'image_like': self.image_like_flags[idx],
            'scorer_dicts': self.scorer_dicts[idx]
        }

    def from_json(self, path: str, indices: List[int] = None) -> None:
        """Load dataset from JSON file.

        Args:
            path (str): Path to JSON file.
            indices (List[int], optional):
                Indices of the dataset to load. Defaults to None.
        """
        self._init_attributes()

        with open(path, 'r') as f:
            items = list(json.load(f))

        if indices is not None:
            items = [items[idx] for idx in indices]

        for i, item in enumerate(items):
            self._load_json_item(i, item)

    def _init_attributes(self) -> None:
        """Helper function to initialize attributes when loading from json.
        """
        self.dialogs, self.contexts, self.utters = [], [], []
        self.ids, self.image_like_flags, self.scorer_dicts = [], [], []

    def _load_json_item(self, i: int, item: Dict[str, Any]) -> None:
        """Helper function to load item info from item dict when loading from JSON.

        Args:
            i (int): Item number.
            item (Dict[str, Any]): Item dict.
        """
        if 'id' in item:
            self.ids.append(item['id'])
        else:
            self.ids.append(str(i))

        self.contexts.append(item['context'])
        self.utters.append(item['utter'])
        self.dialogs.append(item['context'] + [item['utter']])

        if 'image_like' in item:
            self.image_like_flags.append(item['image_like'])
        else:
            self.image_like_flags.append(None)

        if 'scorer_dicts' in item:
            self.scorer_dicts.append(item['scorer_dicts'])
        else:
            self.scorer_dicts.append({})
