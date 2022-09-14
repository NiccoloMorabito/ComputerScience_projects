import torch
from collections import defaultdict
from typing import *
import sys
sys.path.append('../')
from stud.preprocessing import preprocess


def produce_word2vector(glove_path: str) -> Dict:
    """
    :param glove_path: path to the file containing the GloVe pre-trained word vectors
    :return: dictionary word to GloVe vector
    """
    word2vector = dict()
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            word, *vector = line.split()
            vector = torch.tensor([float(c) for c in vector])
            word2vector[word] = vector
    return word2vector


def produce_word2index(glove_path: str) -> Tuple[Dict, List]:
    """
    :param glove_path: path to the file containing the GloVe pre-trained word vectors
    :return: dictionary word to index and the list of GloVe vectors
    """
    word2vector = produce_word2vector(glove_path)

    word2index = dict()
    vectors_list = list()  # list of GloVe vectors
    number_of_features = int(glove_path.split(".")[-2][:-1])
    # pad token at index=0
    vectors_list.append(torch.rand(number_of_features))
    # unk token at index=1
    vectors_list.append(torch.rand(number_of_features))

    for word, vector in word2vector.items():
        word2index[word] = len(vectors_list)
        vectors_list.append(vector)

    word2index = defaultdict(lambda: 1, word2index)  # default index for unk
    vectors_list = torch.stack(vectors_list)
    return word2index, vectors_list


def indices_of_sentence(sentence: str, word2index: Dict) -> torch.Tensor:
    """
    Preprocess the sentence and convert it to a vector of indices for sequence encoding approach.
    :param sentence:
    :param word2index: dictionary to convert the sentence in a vector of indices
    :return: the vector of indices corresponding to the sentence in input
    """
    preprocessed_sentence = preprocess(sentence, remove_stopwords=False, remove_punctuation=False)
    return torch.tensor([word2index[word] for word in preprocessed_sentence], dtype=torch.long)


def vector_of_sentences(sentence1: str, sentence2: str, word2vector: Dict) -> torch.tensor:
    """
    Function to convert two sentences into a vector using the :param word2vector: passed as a parameter.
    :return the concatenation of the two vectors corresponding to the two sentences
    """
    vector1 = vector_of_sentence(sentence1, word2vector)
    vector2 = vector_of_sentence(sentence2, word2vector)
    return torch.cat((vector1, vector2))


def vector_of_sentence(sentence: str, word2vector) -> torch.Tensor:
    """
    Converts each word of :param sentence: using the :param word2vector: after preprocessing it.
    :return: the mean between the vectors
    """
    preprocessed_sentence = preprocess(sentence, remove_stopwords=True, remove_punctuation=True)
    sentence_words_vector = [word2vector[word] for word in preprocessed_sentence if word in word2vector]

    if len(sentence_words_vector) == 0:
        return None

    sentence_words_vector = torch.stack(sentence_words_vector)
    return torch.mean(sentence_words_vector, dim=0)
