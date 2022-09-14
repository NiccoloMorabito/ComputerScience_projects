import torch
from typing import *

PRED_KEY = "pred"
LOSS_KEY = "loss"


# SIMPLER CLASSIFIER (first approach)
class SimplerClassifier(torch.nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(n_features, 1000),
            torch.nn.Linear(1000, 750),
            torch.nn.ReLU(),
            torch.nn.Linear(750, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.BCELoss()
        self.global_epoch = 0

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        pred = self.projection_layer(x).squeeze(1)
        result = {PRED_KEY: pred}

        # compute loss
        if y is not None:
            loss = self.loss(pred, y)
            result[LOSS_KEY] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)


# SEQUENCE ENCODING CLASSIFIER (second approach)
class SequenceEncodingClassifier(torch.nn.Module):

    def __init__(self, vectors_list: torch.Tensor):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding.from_pretrained(vectors_list)
        self.lstm_layer = torch.nn.LSTM(vectors_list.size(1), hidden_size=50, num_layers=1, batch_first=True)
        self.drop_layer = torch.nn.Dropout(0.0001)
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(100, 70),
            torch.nn.ReLU(),
            torch.nn.Linear(70, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
            torch.nn.Sigmoid()
        )
        self.loss_fn = torch.nn.BCELoss()
        self.global_epoch = 0

    def forward(
        self,
        x1: torch.Tensor,
        x1_lengths: torch.Tensor,
        x2: torch.Tensor,
        x2_lengths: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # embedding words of each sentence from indices
        embedding_out1 = self.embedding_layer(x1)
        embedding_out2 = self.embedding_layer(x2)

        # recurrent encoding of each sentence
        lstm_out1 = self.lstm_layer(embedding_out1)[0]
        lstm_out2 = self.lstm_layer(embedding_out2)[0]

        # compute last tokens of each sentence
        summary_vectors1 = self.__compute_last_tokens(lstm_out1, x1_lengths)
        summary_vectors2 = self.__compute_last_tokens(lstm_out2, x2_lengths)

        # concatenation of the last token of the two sentences
        lstm_result = torch.cat((summary_vectors1, summary_vectors2), dim=1)

        # dropouting
        lstm_result = self.drop_layer(lstm_result)

        # final projection
        pred = self.projection_layer(lstm_result).squeeze(1)

        result = {PRED_KEY: pred}

        # compute loss
        if y is not None:
            loss = self.loss(pred, y)
            result[LOSS_KEY] = loss

        return result

    def __compute_last_tokens(self, lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        For each batch element in :param lstm_out:, the last token of the sentence is kept
        removing padding
        """
        batch_size, seq_len, hidden_size = lstm_out.shape
        # sequence of batch x seq_len vectors
        flattened_out = lstm_out.reshape(-1, hidden_size)
        # sequences lengths to retrieve the index of last token output for each sequence
        last_word_relative_indices = lengths - 1
        sequences_offsets = torch.arange(batch_size) * seq_len
        summary_vectors_indices = sequences_offsets + last_word_relative_indices
        return flattened_out[summary_vectors_indices]

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
