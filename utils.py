from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    # 將原本的每個單字轉換成一個int存於陣列
    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len       #計算此batch中最長的input length，後面要做pad
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)                         #理論上做完之後應該要是一個128*to_len的二維陣列
        return padded_ids

    # 用來做slot_tagging的tags補0，因為資料前處理時已經將其tags轉成數字，因此這裡就不再轉換，只做補值的動作，並且不能補0因為原本的class裡面有0這個值
    def encode_batch2(
        self, batch_tags: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        seq_len = [len(seq) for seq in batch_tags] 
        to_len = max(seq_len)
        padded_tags = pad_to_len2(batch_tags, to_len, self.pad_id)                       
        return padded_tags

# 直接補0補到底
def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

# 直接補-1補到底
def pad_to_len2(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [9] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds
