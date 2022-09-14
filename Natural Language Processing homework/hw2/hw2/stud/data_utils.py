from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Tuple
import torch
from torch.nn.utils import rnn
import pytorch_lightning as pl
import json
import os
from transformers import BertTokenizerFast
from nltk.tokenize import TreebankWordTokenizer
import sys
sys.path.append('../')
from stud.embedding_utils import GloveSingleton
from stud.constants import BATCH_INPUT_INDICES, BATCH_INPUT_LENGTHS, BATCH_TARGET_BOOLEANS, BATCH_TARGET_CLASSES,\
    BATCH_SAMPLE, BATCH_BERT_ENCODINGS, BATCH_BERT_TARGET_BOOLEANS, BATCH_BERT_TARGET_CLASSES, BATCH_CATEGORY_BOOLEANS,\
    BATCH_CATEGORY_CLASSES, SENTIMENT_TO_CLASS, ALL_CATEGORIES

# paths
DATA_FOLDER_PATH = "../../data"
LAPTOPS_TRAIN_PATH = os.path.join(DATA_FOLDER_PATH, "laptops_train.json")
LAPTOPS_DEV_PATH = os.path.join(DATA_FOLDER_PATH, "laptops_dev.json")
RESTAURANTS_TRAIN_PATH = os.path.join(DATA_FOLDER_PATH, "restaurants_train.json")
RESTAURANTS_DEV_PATH = os.path.join(DATA_FOLDER_PATH, "restaurants_dev.json")

# field of json
TEXT_FIELD = "text"
TARGETS_FIELD = "targets"
CATEGORIES_FIELD = "categories"

PAD_IDX = 0

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer = TreebankWordTokenizer()
glove_singleton = GloveSingleton()
word_to_index = glove_singleton.word_to_index


@dataclass
class Target:
    term: str
    start_index: int
    end_index: int
    sentiment: str

    def __init__(self, targets_list):
        self.term = targets_list[1]
        self.term_words = tokenizer.tokenize(self.term)
        self.start_index, self.end_index = targets_list[0]

        self.sentiment = None
        if len(targets_list) > 2:
            self.sentiment = targets_list[2]

    def get_term_and_sentiment(self):
        return self.term, self.sentiment


@dataclass
class Category:
    category_name: str
    sentiment: str

    def __init__(self, categs_list):
        self.category_name = categs_list[0]

        self.sentiment = None
        if len(categs_list) > 1:
            self.sentiment = categs_list[1]


@dataclass
class Sample:

    def __init__(self, json_line, with_categories: bool = False):
        self.input_text = json_line.get(TEXT_FIELD)
        self.input_tokens = tokenizer.tokenize(self.input_text)
        self.input_spans = list(tokenizer.span_tokenize(self.input_text))
        self.input_emb_indices = glove_singleton.indices_of_sentence(self.input_tokens)

        if TARGETS_FIELD in json_line:
            self.targets = [Target(targets_list) for targets_list in json_line.get(TARGETS_FIELD)]
            self.target_terms = [target.term for target in self.targets]
            self.target_booleans, self.target_classes = encode_targets(self, len(self.input_tokens), self.input_spans)

        if with_categories and CATEGORIES_FIELD in json_line:
            self.categories = [Category(categs_list) for categs_list in json_line.get(CATEGORIES_FIELD)]
            self.category_booleans, self.category_classes = encode_categories(self.categories)


class AbsaDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 64,
        with_categories: bool = False,
        for_bert: bool = False
    ) -> None:
        super().__init__()
        self.data_train_paths = [RESTAURANTS_TRAIN_PATH]
        self.data_dev_paths = [RESTAURANTS_DEV_PATH]
        self.with_categories = with_categories
        if with_categories:
            self.collate_fn = collate_fn_categories
        else:
            # adding also laptop dataset
            self.data_train_paths.append(LAPTOPS_TRAIN_PATH)
            self.data_dev_paths.append(LAPTOPS_DEV_PATH)
            # selecting collate function
            if for_bert:
                self.collate_fn = collate_fn_bert
            else:
                self.collate_fn = collate_fn

        self.batch_size = batch_size

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = AbsaDataset(self.data_train_paths, self.with_categories)
        self.validation_dataset = AbsaDataset(self.data_dev_paths, self.with_categories)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.__dataloader_from_dataset(self.train_dataset)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.__dataloader_from_dataset(self.validation_dataset)

    def __dataloader_from_dataset(self, dataset):
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          pin_memory=True,
                          )


class AbsaDataset(Dataset):

    def __init__(self, dataset_paths: List[str], with_categories: bool = False):
        self.dataset_paths = dataset_paths
        self.__init_data(with_categories)

    def __init_data(self, with_categories: bool = False):
        self.samples = list()
        for dataset_path in self.dataset_paths:
            for json_line in self.__fetch_json_lines_in(dataset_path):
                sample = Sample(json_line, with_categories)
                self.samples.append(sample)

    @staticmethod
    def __fetch_json_lines_in(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf8") as f:
            return json.loads(f.read())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(samples: List[Sample], device=None) -> Dict:
    """
    Standard collate function
    """
    input_indices_list = [torch.tensor(sample.input_emb_indices, dtype=torch.long) for sample in samples]
    target_booleans_list = [torch.tensor(sample.target_booleans, dtype=torch.float) for sample in samples]
    target_classes_list = [torch.tensor(sample.target_classes, dtype=torch.long) for sample in samples]

    batch = dict()
    batch[BATCH_INPUT_INDICES] = rnn.pad_sequence(input_indices_list, batch_first=True, padding_value=PAD_IDX)
    batch[BATCH_TARGET_BOOLEANS] = rnn.pad_sequence(target_booleans_list, batch_first=True, padding_value=PAD_IDX)
    batch[BATCH_TARGET_CLASSES] = rnn.pad_sequence(target_classes_list, batch_first=True, padding_value=PAD_IDX)
    batch[BATCH_SAMPLE] = samples

    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def collate_fn_categories(samples: List[Sample], device=None) -> Dict:
    """
    Collate function which adds categories to samples of the batch
    """
    batch = collate_fn(samples, device)

    category_booleans = [torch.tensor(sample.category_booleans, dtype=torch.float) for sample in samples]
    category_classes = [torch.tensor(sample.category_classes, dtype=torch.long) for sample in samples]

    batch[BATCH_INPUT_LENGTHS] = torch.tensor([len(sample.input_tokens) for sample in samples], dtype=torch.int)
    batch[BATCH_CATEGORY_BOOLEANS] = torch.stack(category_booleans)
    batch[BATCH_CATEGORY_CLASSES] = torch.stack(category_classes)

    return batch


def collate_fn_bert(samples: List[Sample], device=None) -> Dict:
    """
    Collate function needed for models using BERT
    """
    texts = [sample.input_text for sample in samples]
    encodings = bert_tokenizer(texts, padding=True, add_special_tokens=True,
                               truncation=True, return_tensors="pt")

    input_ids_list = encodings["input_ids"]
    spans_list = bert_tokenizer(texts, padding=True, add_special_tokens=True,
                                truncation=True, return_offsets_mapping=True)["offset_mapping"]

    labels_list = list()
    classes_list = list()
    for sample, input_ids, spans in zip(samples, input_ids_list, spans_list):
        labels, classes = encode_targets(sample, len(input_ids), spans)
        labels_list.append(torch.tensor(labels, dtype=torch.float))
        classes_list.append(torch.tensor(classes, dtype=torch.long))

    batch = dict()
    batch[BATCH_SAMPLE] = samples
    batch[BATCH_BERT_ENCODINGS] = encodings
    batch[BATCH_BERT_TARGET_BOOLEANS] = torch.stack(labels_list)
    batch[BATCH_BERT_TARGET_CLASSES] = torch.stack(classes_list)

    if device is not None:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch


def encode_targets(sample: Sample, number_of_tokens: int, spans: List[Tuple[int]]) -> Tuple[List, List]:
    """
    Encoding targets of sample into two vectors:
    - target_booleans for target/non-target
    - target_classes for sentiments

    If sentiments are not specified, returns (target_booleans, None)

    e.g.
        tokens = [the, screen, is, nice, and, the, images, comes, very,
        clear, the, keyboard, and, the, fit, just, feels, right]
        targets = screen, positive - keyboard, positive - fit, positive
        target_booleans = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        target_classes =  [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0]
    """
    target_booleans = [0] * number_of_tokens
    target_classes = [0] * number_of_tokens
    with_sentiment = True
    for target in sample.targets:
        if target.sentiment is None:
            with_sentiment = False

        indices = text_indices_to_tokens_indices(target.start_index, target.end_index, spans)
        for index in indices:
            target_booleans[index] = 1
            if with_sentiment:
                target_classes[index] = SENTIMENT_TO_CLASS[target.sentiment]
    if not with_sentiment:
        target_classes = None
    return target_booleans, target_classes


def text_indices_to_tokens_indices(text_start_index: int, text_end_index: int, spans: List[Tuple[int]]):
    """
    Convert indices of text into indices of tokens given spans of these tokens
    Args:
        text_start_index
        text_end_index
        spans

    Returns:
        (token_start_index, token_end_index)
    """
    text_span = (text_start_index, text_end_index)

    if text_span in spans:
        return [spans.index(text_span)]

    starts = [span[0] for span in spans]
    ends = [span[1] for span in spans]
    token_start_index = get_position(text_start_index, starts)
    token_end_index = get_position(text_end_index, ends)
    return range(token_start_index, token_end_index + 1)


def get_position(index: int, indices: List[int]) -> int:
    """
    Returns the position of the index in the indices or the first one after.
    """
    if index in indices:
        return indices.index(index)
    for i, el in enumerate(indices):
        if el > index:
            return i - 1


def encode_categories(categories: List[Category]) -> Tuple[List, List]:
    """
    Encoding categories into two vectors:
    - category_booleans for categories
    - category_classes for sentiments

    If sentiments are not specified, returns (category_booleans, None)
    """
    category_booleans = [0] * len(ALL_CATEGORIES)
    category_classes = [0] * len(ALL_CATEGORIES)
    with_sentiment = True
    for category in categories:
        if category.sentiment is None:
            with_sentiment = False

        category_index = ALL_CATEGORIES.index(category.category_name)
        category_booleans[category_index] = 1
        if with_sentiment:
            category_classes[category_index] = SENTIMENT_TO_CLASS[category.sentiment]

    if not with_sentiment:
        category_classes = None

    return category_booleans, category_classes
