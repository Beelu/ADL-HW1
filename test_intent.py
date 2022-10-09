import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np

import torch
import csv

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    testdataset = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn2, shuffle=False, batch_size=args.batch_size)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # device = args.device
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    # 預測得到結果後，將其數值轉換成英文字，然後對應到各個id上
    test_ids = []
    pred_intent = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(testdataset):
                text, ids = batch
                text = text
                outputs = model(text)
 
                _, max_num = torch.max(outputs, 1)
                pred_num = max_num.numpy().tolist()   # 先把預測值從tensor轉回list

                for i in range(len(pred_num)):        # 再將數值轉成英文字
                    pred_intent.append(dataset.idx2label(pred_num[i]))
                # print(pred_intent)

                for id in ids:                              # 建立一個128維的id list
                    test_ids.append(id)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as csvf:
        w = csv.writer(csvf)
        w.writerow(['id','intent'])

        for i in range(len(test_ids)):
            w.writerow([test_ids[i], pred_intent[i]])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/model.ckpt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
