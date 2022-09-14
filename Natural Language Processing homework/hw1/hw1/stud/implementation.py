import numpy as np
from typing import List, Tuple, Dict

import torch.nn
from model import Model

from stud.classifiers import SequenceEncodingClassifier, SimplerClassifier
from stud.embedding_utils import produce_word2index, produce_word2vector
from stud.data_utils import get_dataloader_of

# PATHS
SIMPLER_STATE_DICT_PATH = "model/SimplerClassifier.statedict"
SE_STATE_DICT_PATH = "model/SequenceEncodingClassifier.statedict"
GLOVE_PATH = "model/glove/glove.6B.50d.txt"
FALSE_LABEL = "False"
TRUE_LABEL = "True"


def build_model(device: str) -> Model:
    return StudentModel(device)


class RandomBaseline(Model):

    options = [
        (TRUE_LABEL, 40000),
        (FALSE_LABEL, 40000),
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]


class StudentModel(Model):

    def __init__(self, device: str):
        word2vector = produce_word2vector(GLOVE_PATH)
        num_of_features = int(GLOVE_PATH.split(".")[-2][:-1]) * 2
        self.word2vector = word2vector
        self.simpler_classifier = SimplerClassifier(num_of_features)
        self.simpler_classifier.load_state_dict(torch.load(SIMPLER_STATE_DICT_PATH))
        self.simpler_classifier.to(device)
        self.simpler_classifier.eval()

    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        dataloader = get_dataloader_of(sentence_pairs, self.word2vector, sequence_encoding=False)

        result = list()
        for x in dataloader:
            batch_out = self.simpler_classifier(x)
            preds = torch.round(batch_out['pred'])

            for pred in preds:
                if pred == 0:
                    result.append(FALSE_LABEL)
                else:
                    result.append(TRUE_LABEL)

        return result


# model to use the approach based on sequence encoding
class SequenceEncodingStudentModel(Model):

    def __init__(self, device: str):
        word2index, vectors_list = produce_word2index(GLOVE_PATH)
        self.word2index = word2index
        self.sequence_encoding_classifier = SequenceEncodingClassifier(vectors_list)
        self.sequence_encoding_classifier.load_state_dict(torch.load(SE_STATE_DICT_PATH))
        self.sequence_encoding_classifier.to(device)
        self.sequence_encoding_classifier.eval()

    @torch.no_grad()
    def predict(self, sentence_pairs: List[Dict]) -> List[Dict]:
        dataloader = get_dataloader_of(sentence_pairs, self.word2index, sequence_encoding=True)

        result = list()
        for x1, x1_lengths, x2, x2_lengths in dataloader:
            batch_out = self.sequence_encoding_classifier(x1, x1_lengths, x2, x2_lengths)
            preds = torch.round(batch_out['pred'])

            for pred in preds:
                if pred == 0:
                    result.append(FALSE_LABEL)
                else:
                    result.append(TRUE_LABEL)

        return result
