import os
import json
from typing import Union, List, Dict, Any, Callable
from functools import partial

from .dialog_dataset import DialogDataset

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


class GPTDialogDataset(DialogDataset):
    """Class containing a dialog dataset with features for FAQ scoring.
    """
    def __init__(
        self, path2json: str
    ):
        """
        Args:
            path2json (str):
                Path to JSON file containing the dataset.
        """
        DialogDataset.__init__(self, dialogs=path2json)

    def __getitem__(self, idx: int) -> dict:
        return {
            'idx': idx, 'id': self.ids[idx],
            'utter': self.utters[idx],
            'context': self.contexts[idx],
            'gpt_score': self.gpt_scores[idx],
            'gpt_prompt_scores': self.gpt_prompt_scores[idx],
        }

    def _save_json_item(self, idx: int, item: Dict[str, Any]) -> None:
        """Helper function to save item info into item dict for saving to JSON.

        Args:
            idx (int): Item index.
            item (Dict[str, Any]): Item dict.
        """
        super(GPTDialogDataset, self)._save_json_item(idx, item)
        item['scorer_dicts']['gpt'] = {
             'score': self.gpt_scores[idx],
             'prompt_scores': self.gpt_prompt_scores[idx],
             'prompts_tag': self.prompts_tags[idx]
        }

    def _init_attributes(self):
        """Helper function to initialize attributes when loading from json.
        """
        super(GPTDialogDataset, self)._init_attributes()
        self.gpt_scores, self.gpt_prompt_scores, self.prompts_tags = [], [], []

    def _load_json_item(self, i: int, item: Dict[str, Any]) -> None:
        """Helper function to load item info from item dict when loading from JSON.

        Args:
            i (int): Item number.
            item (Dict[str, Any]): Item dict.
        """
        super(GPTDialogDataset, self)._load_json_item(i, item)
        if 'scorer_dicts' in item and 'gpt' in item['scorer_dicts']:
            self.gpt_scores.append(item['scorer_dicts']['gpt']['score'])
            self.gpt_prompt_scores.append(item['scorer_dicts']['gpt']['prompt_scores'])
            self.prompts_tags.append(item['scorer_dicts']['gpt']['prompts_tag'])
        else:
            self.gpt_scores.append(None)
            self.gpt_prompt_scores.append(None)
            self.prompts_tags.append(None)


class GPTScorer:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
        path2prompts: str, device: Union[torch.device, str] = 'cpu'
    ):
        """
        Args:
            model (AutoModelForCausalLM):
                Huggingface AutoModelForCausalLM.
            tokenizer (AutoTokenizer):
                Huggingface AutoTokenizer.
            path2prompts (str, optional):
                Path to prompts JSON.
            device (Union[torch.device, str], optional):
                PyTorch device. Defaults to 'cpu'.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.path2prompts = path2prompts
        self.device = device

    def score(
        self, dialog_dataset: GPTDialogDataset, batch_size=8
    ):
        """Score each item in dataset with GPT generation confidence.

        Args:
            dialog_dataset (GPTFeaturedDialogDataset):
                Dataset to score.
        """
        dialog_dataset.gpt_prompt_scores = [[] for _ in range(len(dialog_dataset))]

        with open(self.path2prompts, 'r') as f:
            prompts = json.load(f)

        for prompt in prompts:
            prompt['tokens'] = self.tokenizer.tokenize(prompt['prompt'])

            dataloader = DataLoader(
                dialog_dataset, batch_size=batch_size,
                collate_fn=partial(collate_fn, tokenizer=self.tokenizer, prompt=prompt)
            )

            model = self.model.to(self.device)
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=prompt['prompt']):
                    inputs = batch['prompted_inputs'].to(self.device)

                    outputs = model(**inputs, labels=inputs['input_ids'])

                    log_proba = F.log_softmax(outputs.logits, dim=-1)
                    true_proba = log_proba[
                        torch.arange(log_proba.shape[0]).unsqueeze(-1),
                        torch.arange(log_proba.shape[1]),
                        inputs['input_ids']
                    ]
                    for proba, start, end, idx, tokens in zip(
                        true_proba, batch['start_indices'], batch['end_indices'],
                        batch['indices'], batch['prompted_tokens']
                    ):
                        # print(tokens[start:end])
                        confidence = proba[start:end].mean().item()
                        dialog_dataset.gpt_prompt_scores[idx].append(confidence)

            for idx, scores in enumerate(dialog_dataset.gpt_prompt_scores):
                dialog_dataset.gpt_scores[idx] = sum(scores) / len(scores)


def collate_fn(data: Dict[str, List[Any]], tokenizer: AutoTokenizer, prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Function to collate lists of data in a batch.

    Args:
        data (Dict[str, List[Any]]): Lists of data.

    Returns:
        Dict[str, Any]: Batch of data.
    """
    batch = {batch_key: [item[item_key] for item in data] for batch_key, item_key in [
                ('indices', 'idx'), ('ids', 'id'), ('utters', 'utter'), ('contexts', 'context')
            ]}
    batch['utter_tokens'] = [tokenizer.tokenize(utter) for utter in batch['utters']]

    prompted_tokens, start_indices, end_indices = [], [], []
    for utter_tokens in batch['utter_tokens']:
        if prompt['type'] == 'prefix':
            prompted_tokens.append(prompt['tokens'] + utter_tokens)
            start_indices.append(len(prompt['tokens']))
        elif prompt['type'] == 'postfix':
            prompted_tokens.append(utter_tokens + prompt['tokens'])
            start_indices.append(len(utter_tokens))
        else:
            raise ValueError("Wrong prompt type (must be 'prefix' or 'postfix')")
        end_indices.append(len(prompt['tokens']) + len(utter_tokens))

    batch['prompted_tokens'] = prompted_tokens
    batch['prompted_inputs'] = tokenizer(
        prompted_tokens, is_split_into_words=True, padding='longest', return_tensors='pt'
    )
    batch['start_indices'] = start_indices
    batch['end_indices'] = end_indices

    return batch


# def collate_fn(batch, prompt, padding_value, batch_first=True):
#     texts = [data['text'] for data in batch]
#     if prompt['type'] == 'postfix':
#         tokens = [torch.cat([data['tokens'][:-1], pattern]) for data in batch]
#         tokens_lens = torch.tensor([data['tokens_len']-1 for data in batch], dtype=torch.long)
#     elif prompt['type'] == 'prefix':
#         tokens = [torch.cat([pattern, data['tokens']]) for data in batch]
#         tokens_lens = torch.tensor([data['tokens_len'] for data in batch], dtype=torch.long)
#     else:
#         raise ValueError("Wrong prompt type (must be 'prefix' or 'postfix')")

#     tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=batch_first,
#                                              padding_value=padding_value)
#     if is_postfix:
#         start_idx = tokens_lens
#         end_idx = start_idx + len(pattern)
#     else:
#         start_idx = torch.tensor([len(pattern) for data in batch], dtype=torch.long)
#         end_idx = start_idx + tokens_lens

#     return {'texts': texts, 'tokens': tokens, 'tokens_lens': tokens_lens, 'start_idx': start_idx, 'end_idx': end_idx}
