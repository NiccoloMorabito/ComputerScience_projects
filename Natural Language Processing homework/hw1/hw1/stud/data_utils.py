import torch
import jsonlines
from typing import *
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from stud.embedding_utils import indices_of_sentence, vector_of_sentences

BATCH_SIZE = 32
# fields of json
SENTENCE1_FIELD = "sentence1"
SENTENCE2_FIELD = "sentence2"
LABEL_FIELD = "label"


def get_dataloader_from(path: str, word_emb_dict: Dict, sequence_encoding: bool) -> DataLoader:
    """
    Returns a torch DataLoader loading data from the :param path:
    :param word_emb_dict: dictionary used for word embedding
    :param sequence_encoding: flag for approach based on sequence encoding
    """
    json_lines = read_dataset(path)
    return get_dataloader_of(json_lines, word_emb_dict, sequence_encoding)


def read_dataset(path: str) -> List[Dict]:
    json_lines = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            json_lines.append(obj)
    return json_lines


def get_dataloader_of(json_lines: List[Dict], word_emb_dict: Dict, sequence_encoding=True) -> DataLoader:
    """
    Returns a torch DataLoader creating a dataset with the data passed in :param json_lines:
    :param word_emb_dict: dictionary used for word embedding
    :param sequence_encoding: flag for approach based on sequence encoding
    """
    if sequence_encoding:
        dataset = SequenceEncodingDataset(json_lines, word_emb_dict)
        return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=sequence_encoding_collate_fn)
    else:
        dataset = SimplerDataset(json_lines, word_emb_dict)
        return DataLoader(dataset, batch_size=BATCH_SIZE)


# Dataset for simpler approach (sentences are converted into the corresponding embedding vectors)
class SimplerDataset(torch.utils.data.Dataset):
    def __init__(self, json_lines: List[Dict], word2vector: Dict):
        self.data_store = []
        self.__load_data(json_lines, word2vector)

    def __load_data(self, json_lines: List[Dict], word2vector: Dict):
        for json_line in json_lines:
            x = vector_of_sentences(json_line[SENTENCE1_FIELD], json_line[SENTENCE2_FIELD], word2vector)

            if LABEL_FIELD in json_line.keys():
                boolean = eval(json_line[LABEL_FIELD])  # string to boolean
                y = torch.tensor(boolean).float()
                self.data_store.append((x, y))
            else:
                self.data_store.append(x)

    def __len__(self) -> int:
        return len(self.data_store)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data_store[idx]


# Dataset for approach based on sequence encoding (sentences are converted into vectors of indices)
class SequenceEncodingDataset(torch.utils.data.Dataset):

    def __init__(self, json_lines: List[Dict], word2index: Dict):
        self.data_store = []
        self.__load_data(json_lines, word2index)

    def __load_data(self, json_lines: List[Dict], word2index: Dict):
        for json_line in json_lines:
            x1 = indices_of_sentence(json_line[SENTENCE1_FIELD], word2index)
            x2 = indices_of_sentence(json_line[SENTENCE2_FIELD], word2index)

            if LABEL_FIELD in json_line.keys():
                boolean = eval(json_line[LABEL_FIELD])  # string to boolean
                y = torch.tensor(boolean).float()

                self.data_store.append((x1, x2, y))
            else:
                self.data_store.append((x1, x2))

    def __len__(self) -> int:
        return len(self.data_store)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data_store[idx]


def sequence_encoding_collate_fn(
    data_elements: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]  # list of (x1, x2, y)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Collate function for the sequence encoding DataLoader
    :param data_elements: list of (x1, x2, y) where x1 and x2 are vectors of indices corresponding to the two sentences
        and y is optionally specified
    :return: batch of input and optionally output
    """

    x1 = [de[0] for de in data_elements]
    x2 = [de[1] for de in data_elements]

    # to implement the many-to-one strategy
    x1_lengths = torch.tensor([x1.size(0) for x1 in x1], dtype=torch.long)
    x2_lengths = torch.tensor([x2.size(0) for x2 in x2], dtype=torch.long)

    x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True, padding_value=0)  # shape (batch_size x max_seq_len)
    x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True, padding_value=0)  # shape (batch_size x max_seq_len)

    # if output y has been specified
    if len(data_elements[0]) == 3:
        y = [de[2] for de in data_elements]
        y = torch.tensor(y)
        return x1, x1_lengths, x2, x2_lengths, y
    else:
        return x1, x1_lengths, x2, x2_lengths
