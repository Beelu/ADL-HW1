import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())               
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}           
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}  
    datasets: Dict[str, SeqTaggingClsDataset] = {                                                  
        split: SeqTaggingClsDataset(split_data, vocab, intent2idx, args.max_len)              
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    # 由於在做訓練與預測時必須是由數字值所組成的向量，因此以下要開始將input及output轉換型態
    # 先轉換target值
    for key, value in datasets.items():
        for i, text in enumerate(datasets[key].data):
            for j in range(len(text['tags'])):
                datasets[key].data[i]['tags'][j] = datasets[key].label_mapping[text['tags'][j]]
    
    # 由於輸入字串長度不同，因此需要客製化collate_fn
    traindataset = torch.utils.data.DataLoader(datasets['train'], collate_fn = datasets['train'].collate_fn, batch_size=args.batch_size, shuffle=True)
    testdataset = torch.utils.data.DataLoader(datasets['eval'], collate_fn = datasets['eval'].collate_fn, batch_size=args.batch_size, shuffle=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = args.device
    model = SeqTagger(
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

        model.train()       # 啟用Batch Normalization跟Dropout
        for batch_idx, batch in enumerate(traindataset):
            loss = 0.0
            text, intent = batch
            text = text.to(device)
            intent = intent.to(device)

            optimizer.zero_grad()
            output = model(text)                # 這裡的output是三維，output[128(batch_size)][句子長度][9(class_num)]
            for i in range(len(intent)):
                loss += lossF(output[i], intent[i])
            loss.backward()
            optimizer.step()

            train_pred = torch.argmax(output, dim=2)    # 這裡我們要讓最後一維度消失求得最大值。train_pred[128(batch_size)][句子長度]
            for i in range(len(train_pred)):
                train_acc += 1 if (train_pred[i] == intent[i]).sum().item() == len(train_pred[i]) else 0
            train_loss += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        # Optimize the validation process with `torch.no_grad()`
        model.eval()                    # 關閉Batch Normalization跟Dropout
        a = []
        b = []
        with torch.no_grad():           # 測試時，關閉autograd，不需偏微分做gradient descent，可加快測試速度。
            for batch_idx, batch in enumerate(testdataset):
                loss = 0.0
                text, intent = batch
                text = text.to(device)
                intent = intent.to(device)

                optimizer.zero_grad()
                output = model(text)
                for i in range(len(intent)):
                    loss += lossF(output[i], intent[i])

                test_pred = torch.argmax(output, dim=2)
                for i in range(len(test_pred)):
                    test_acc += 1 if (test_pred[i] == intent[i]).sum().item() == len(test_pred[i]) else 0
                test_loss += loss.item()

                # test_pred = test_pred.cpu().numpy().tolist()
                # intent = intent.cpu().numpy().tolist()
                # for i in range(len(test_pred)):        # 再將數值轉成英文字
                #     temp = []
                #     for j in range(len(test_pred[i])):
                #         # if test_pred[i][j] != 9:
                #         temp.append(datasets['eval'].idx2label(test_pred[i][j]))
                #     a.append(temp)
                # for i in range(len(intent)):        # 再將數值轉成英文字
                #     temp = []
                #     for j in range(len(intent[i])):
                #         # if intent[i][j] != 9:
                #         temp.append(datasets['eval'].idx2label(intent[i][j]))
                #     b.append(temp)
                
                # print(a)
                # print(b)
                # print(classification_report(a, b, mode='strict', scheme=IOB2))

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, args.num_epoch, train_acc/len(datasets[TRAIN]), train_loss/len(traindataset), test_acc/len(datasets[DEV]), test_loss/len(testdataset)
            ))
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), args.ckpt_dir / "model.ckpt")
                print('saving model with acc {:.3f}'.format(best_acc/len(datasets[DEV])))
    
        pass


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
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