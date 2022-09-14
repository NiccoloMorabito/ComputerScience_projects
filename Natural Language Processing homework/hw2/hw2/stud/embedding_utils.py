import torch
from collections import defaultdict
from typing import *

GLOVE_PATH = "../../model/glove/glove.6B.50d.txt"


class GloveSingleton:

    class __GloveSingleton:

        def __init__(self):
            self.word_to_index, self.vectors_list = self.__produce_word_to_index()

        def __produce_word_to_index(self) -> Tuple[Dict, List]:
            """
            Returns: dictionary word to index and the list of GloVe vectors
            """
            word_to_vector = self.__produce_word_to_vector()

            word_to_index = dict()
            vectors_list = list()  # list of GloVe vectors
            number_of_features = int(GLOVE_PATH.split(".")[-2][:-1])
            # pad token at index=0
            vectors_list.append(torch.rand(number_of_features))
            # unk token at index=1
            vectors_list.append(torch.rand(number_of_features))

            for word, vector in word_to_vector.items():
                word_to_index[word] = len(vectors_list)
                vectors_list.append(vector)

            word_to_index = defaultdict(lambda: 1, word_to_index)  # default index for unk
            vectors_list = torch.stack(vectors_list)
            return word_to_index, vectors_list

        def __produce_word_to_vector(self) -> Dict:
            """
            Returns: dictionary word to GloVe vector
            """
            word_to_vector = dict()
            with open(GLOVE_PATH, 'r', encoding="utf-8") as f:
                for line in f:
                    word, *vector = line.split()
                    vector = torch.tensor([float(c) for c in vector])
                    word_to_vector[word] = vector
            return word_to_vector

    instance = None

    def __init__(self):
        if not GloveSingleton.instance:
            GloveSingleton.instance = GloveSingleton.__GloveSingleton()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def indices_of_sentence(self, sentence: str) -> List:
        """
        Preprocess the sentence and convert it to a vector of indices for sequence encoding approach.

        Returns: the vector of indices corresponding to the sentence in input
        """
        return [self.word_to_index[word] for word in sentence]
