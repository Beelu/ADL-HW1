from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = torch.nn.GRU(input_size=embeddings.size(1), 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout,
                            bidirectional=bidirectional)
        # 須注意，若使用bidirectional=True，則在訓練時會產生雙向forward pass，因此hidden_size最後會變成兩倍，所以最後的分類器也須隨之調整。
        # self.lstm = torch.nn.LSTM(input_size=embeddings.size(1), 
        #                     hidden_size=hidden_size, 
        #                     num_layers=num_layers, 
        #                     batch_first=True, 
        #                     dropout=dropout,
        #                     bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*2, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    # 再使用model(x)時，就會將x送進這裡做forward pass
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedding = self.embed(batch)

        output, hidden = self.gru(embedding)
        # output, (hidden, cell) = self.lstm(embedding)
        output = output[:, -1]
        output = torch.nn.functional.relu(output)
        fc_output = self.fc(output)

        return fc_output


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # TODO: implement model forward
        embedding = self.embed(batch)

        output, hidden = self.gru(embedding)
        # output, (hidden, cell) = self.lstm(embedding)
        # output = output[:, -1]
        output = torch.nn.functional.relu(output)
        fc_output = self.fc(output)

        return fc_output
