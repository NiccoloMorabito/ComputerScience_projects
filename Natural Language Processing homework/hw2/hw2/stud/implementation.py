import torch
import numpy as np
from typing import List, Dict
import random
import sys
sys.path.append("../")
from model import Model
from stud.embedding_utils import GloveSingleton
from stud.data_utils import Sample
from stud.task_a import AspectTermIdentifierModel
from stud.task_b import AspectTermClassifierModel

MODEL_A_PATH = "model/model_a.ckpt"
MODEL_B_PATH = "model/model_b.ckpt"


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    return ModelB(device)


def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    # return RandomBaseline(mode='ab')
    return ModelAB(device)


def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    # return RandomBaseline(mode='cd')
    raise NotImplementedError


class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else:
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0])
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0])
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, samples: List[Dict]) -> List[Dict]:
        '''
        --> !!! STUDENT: implement here your predict function !!! <--
        Args:
            - If you are doing model_b (ie. aspect sentiment analysis):
                sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza.",
                            "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            "targets": [[4, 9], "people", [[36, 40], "taste"]]
                        }
                    ]
            - If you are doing model_ab or model_cd:
                sentence: a dictionary that represents an input sentence, for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza."
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        }
                    ]
        Returns:
            A List of dictionaries with your predictions:
                - If you are doing target word identification + target polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                        }
                    ]
                - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                            "categories": [("food", "conflict")]
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                            "categories": [("service", "positive"), ("food", "positive")]
                        }
                    ]
        '''
        pass


class ModelB(Model):

    def __init__(self, device: str):
        self.model_b = AspectTermClassifierModel.load_from_checkpoint(MODEL_B_PATH,
                                                                      embeddings=GloveSingleton().vectors_list)
        self.model_b.to(device)
        self.model_b.eval()

    @torch.no_grad()
    def predict(self, samples: List[Dict]) -> List[Dict]:
        samples = [Sample(sample) for sample in samples]
        results = list()
        for sample in samples:
            x = torch.tensor(sample.input_emb_indices).unsqueeze(0)  # add a fake batch dimension to shape
            booleans = torch.tensor(sample.target_booleans, dtype=torch.float).unsqueeze(0)
            _, predictions = self.model_b.forward(x, booleans)
            prediction = predictions.squeeze(0)
            result = {"targets": self.model_b.prediction_to_terms_sentiments(prediction, sample.input_tokens)}
            results.append(result)

        return results


class ModelAB(Model):

    def __init__(self, device: str):
        vectors_list = GloveSingleton().vectors_list
        self.model_a = AspectTermIdentifierModel.load_from_checkpoint(MODEL_A_PATH, embeddings=vectors_list)
        self.model_a.to(device)
        self.model_a.eval()

        self.model_b = AspectTermClassifierModel.load_from_checkpoint(MODEL_B_PATH, embeddings=vectors_list)
        self.model_b.to(device)
        self.model_b.eval()

    @torch.no_grad()
    def predict(self, samples: List[Dict]) -> List[Dict]:
        samples = [Sample(sample) for sample in samples]
        results = list()
        for sample in samples:
            # add a fake batch dimension to shape
            x = torch.tensor(sample.input_emb_indices).unsqueeze(0)
            # results of model a
            _, predictions_a = self.model_a.forward(x)

            # results of model b and final results
            _, predictions_b = self.model_b.forward(x, predictions_a)
            prediction = predictions_b.squeeze(0)
            result = {"targets": self.model_b.prediction_to_terms_sentiments(prediction, sample.input_tokens)}
            results.append(result)

        return results

