from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    # 在這裡除了將input長度統一外，還順便將單字轉成數字(透過utils的encode_batch)
    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text_arr = []
        intent_arr = []
        for batch in samples:
            text_arr.append(batch['text'].split())
            intent_arr.append(batch['intent'])
        text_arr = self.vocab.encode_batch(text_arr)                 # 原本為128個字串，現在回傳為二維陣列，第二層依照字串長度而定
        text_arr = torch.tensor(text_arr, dtype=torch.int64)
        intent_arr = torch.tensor(intent_arr, dtype=torch.int64)
        return text_arr, intent_arr

    # 做預測測試用，所以沒有target值
    def collate_fn2(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text_arr = []
        id_arr = []
        for batch in samples:
            text_arr.append(batch['text'].split())
            id_arr.append(batch['id'])
        text_arr  = self.vocab.encode_batch(text_arr)
        text_arr = torch.tensor(text_arr, dtype=torch.int64)
        return text_arr, id_arr

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        tokens_arr = []
        tags_arr = []
        for batch in samples:
            tokens_arr.append(batch['tokens'])
            tags_arr.append(batch['tags'])
        tokens_arr = self.vocab.encode_batch(tokens_arr) 
        tags_arr = self.vocab.encode_batch2(tags_arr) 
        tokens_arr = torch.tensor(tokens_arr, dtype=torch.int64)
        tags_arr = torch.tensor(tags_arr, dtype=torch.int64)
        return tokens_arr, tags_arr

    def collate_fn2(self, samples):
        # TODO: implement collate_fn
        tokens_arr = []
        id_arr = []
        for batch in samples:
            tokens_arr.append(batch['tokens'])
            id_arr.append(batch['id'])
        tokens_arr = self.vocab.encode_batch(tokens_arr) 
        tokens_arr = torch.tensor(tokens_arr, dtype=torch.int64)
        return tokens_arr, id_arr
