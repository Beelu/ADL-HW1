import json
from multiprocessing.sharedctypes import Value
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from symbol import parameters
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())                # 將intent2idx.json讀入，並且為字典型態
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}           # 分別為train.json跟eval.json兩個路徑檔名，用在下行讀檔用
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}  # 讀入train跟eval資料，型態為字典的data
    datasets: Dict[str, SeqClsDataset] = {                                              # data字典的值再用自定義型態SeqClsDataset讀入
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)               # split_data也是一個字典，包含text, intent, id; vocab是文字庫; 
        for split, split_data in data.items()
    }
    print(datasets['train'].data[0])
    # TODO: crecate DataLoader for train / dev datasets
    # 由於在做訓練與預測時必須是由數字值所組成的向量，因此以下要開始將input及output轉換型態
    # 先轉換target值
    for key, value in datasets.items():
        for i, text in enumerate(datasets[key].data):
            datasets[key].data[i]['intent'] = datasets[key].label_mapping[text['intent']]
    
    # 由於輸入字串長度不同，因此需要客製化collate_fn
    traindataset = torch.utils.data.DataLoader(datasets['train'], collate_fn = datasets['train'].collate_fn, batch_size=args.batch_size, shuffle=True)
    testdataset = torch.utils.data.DataLoader(datasets['eval'], collate_fn = datasets['eval'].collate_fn, batch_size=args.batch_size, shuffle=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets['train'].num_classes,
    ).to(device)

    # ckpt = torch.load(args.ckpt_path)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossF = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0

        model.train()
        for batch_idx, batch in enumerate(traindataset):
            text, intent = batch
            text = text.to(device)
            intent = intent.to(device)

            optimizer.zero_grad()
            output = model(text)                    # 在這裡獲得的output還是一個150維的向量，表示屬於各種intent的機率。output[128][150]
            loss = lossF(output, intent)
            loss.backward()
            optimizer.step()
            _, train_pred = torch.max(output, 1)    # 接者我們從這行中取出最大的機率當作預測值。
            
            train_acc += (train_pred == intent).sum().item()     # 逐一比較此batch所有output值並且計算其中為true的總數
            train_loss += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        # Optimize the validation process with `torch.no_grad()`
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testdataset):
                text, intent = batch
                text = text.to(device)
                intent = intent.to(device)

                optimizer.zero_grad()
                output = model(text)
                loss = lossF(output, intent)

                _, test_pred = torch.max(output, 1)
                test_acc += (test_pred.detach() == intent.detach()).sum().item()
                test_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, args.num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(traindataset), test_acc/len(datasets[DEV]), test_loss/len(testdataset)
            ))
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), args.ckpt_dir / "model.ckpt")
                print('saving model with acc {:.3f}'.format(best_acc/len(datasets[DEV])))

        pass
    # TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
