import os
from typing import Union, List, Dict, Any, Callable

from .dialog_dataset import DialogDataset
from .featured_dataset import FeaturedDataset

import torch
from sentence_transformers import SentenceTransformer


class FAQFeaturedDialogDataset(DialogDataset, FeaturedDataset):
    """Class containing a dialog dataset with features for FAQ scoring.
    """
    def __init__(
        self, path2json: str, path2features: str,
        model: SentenceTransformer = None,
        batch_size: int = 32, device: Union[torch.device, str] = 'cpu'
    ):
        """
        Args:
            path2json (str):
                Path to JSON file containing the dataset.
            path2features (str):
                Path to directory containing feature vectors.
            model (CLIPModel, optional):
                Huggingface SentenceTransformer model. Defaults to None.
            batch_size (int, optional):
                Batch size for generating feature vectors.
            device (Union[torch.device, str], optional):
                PyTorch device. Defaults to 'cpu'.
        """
        DialogDataset.__init__(self, dialogs=path2json)

        FeaturedDataset.__init__(
            self, path2features=path2features,
            model=model, preprocessor=lambda utters: [f'<A>{utter}' for utter in utters],
            batch_size=batch_size, device=device
        )

    def __getitem__(self, idx: int) -> dict:
        return {
            'idx': idx, 'id': self.ids[idx],
            'utter': self.utters[idx],
            'context': self.contexts[idx],
            'path2features': self.feature_paths[idx],
            'features':
                torch.load(self.feature_paths[idx]) if self.feature_paths[idx] is not None else None,
            'faq_score': self.faq_scores[idx]
        }

    def _save_json_item(self, idx: int, item: Dict[str, Any]) -> None:
        """Helper function to save item info into item dict for saving to JSON.

        Args:
            idx (int): Item index.
            item (Dict[str, Any]): Item dict.
        """
        super(FAQFeaturedDialogDataset, self)._save_json_item(idx, item)
        item['scorer_dicts']['faq'] = {
            'score': self.faq_scores[idx],
            'questions': self.questions[idx]
        }

    def _init_attributes(self):
        """Helper function to initialize attributes when loading from json.
        """
        super(FAQFeaturedDialogDataset, self)._init_attributes()
        self.faq_scores, self.questions = [], []

    def _load_json_item(self, i: int, item: Dict[str, Any]) -> None:
        """Helper function to load item info from item dict when loading from JSON.

        Args:
            i (int): Item number.
            item (Dict[str, Any]): Item dict.
        """
        super(FAQFeaturedDialogDataset, self)._load_json_item(i, item)
        if 'scorer_dicts' in item and 'faq' in item['scorer_dicts']:
            self.faq_scores.append(item['scorer_dicts']['faq']['score'])
            self.questions.append(item['scorer_dicts']['faq']['questions'])
        else:
            self.faq_scores.append(None)
            self.questions.append(None)

    def _generate_feature_vectors(
        self, indices: List[int],
        model: SentenceTransformer, preprocessor: Callable,
        batch_size: int
    ):
        """Helper function to generate utterance feature vectors using a CLIP model.

        Args:
            indices (List[int]): Indices of the dataset to generate feature vectors for.
            model (CLIPModel): Huggingface SentenceTransformer model.
            preprocessor (Callable): Preprocessing function.
            batch_size (int): Batch size for generating feature vectors.
        """
        utters = [self.utters[idx] for idx in indices]
        answers = preprocessor(utters)
        features = model.encode(
            answers, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=False, device=self.device,
            normalize_embeddings=True
        )
        ids = [self.ids[idx] for idx in indices]
        for vector, idx, id in zip(features, indices, ids):
            path = f"{os.path.join(self.path2features, id)}.pt"
            torch.save(vector.clone().to('cpu'), path)
            self.feature_paths[idx] = path


class FAQScorer:
    def __init__(
        self, model: SentenceTransformer, questions: List[str] = None,
        device: Union[torch.device, str] = 'cpu'
    ):
        """
        Args:
            model (SentenceTransformer):
                Huggingface SentenceTransformer model.
            questions (List[str], optional):
                List of questions to compare answers to. Defaults to None.
            device (Union[torch.device, str], optional):
                PyTorch device. Defaults to 'cpu'.
        """
        self.model = model
        self.questions = ['Whats on this picture?'] if questions is None else questions
        self.device = device

    def score(
        self, dialog_dataset: FAQFeaturedDialogDataset,
        path2features: str = None, max_workers: int = None
    ):
        """Score each item in dataset with mean similarity to given questions.

        Args:
            dialog_dataset (FAQFeaturedDialogDataset):
                Dataset to score.
            path2features (str, optional):
                Path to dialog dataset features tensor. Defaults to None.
            max_workers (int, optional):
                Maximum number of workers to spawn when loading feature vectors. Defaults to None.
        """
        questions = [f'<Q>{q}' for q in self.questions]
        question_embeddings = self.model.encode(
            questions, show_progress_bar=False,
            convert_to_numpy=False, convert_to_tensor=True,
            device=self.device, normalize_embeddings=True
        )

        dialog_dataset.load_feature_vectors(
            path=path2features, max_workers=max_workers
        )
        answer_embeddings = dialog_dataset.get_feature_vectors().to(self.device)

        similarities = torch.matmul(answer_embeddings, question_embeddings.T).mean(axis=1)
        dialog_dataset.faq_scores = similarities.tolist()
        dialog_dataset.questions = [self.questions] * len(dialog_dataset)
