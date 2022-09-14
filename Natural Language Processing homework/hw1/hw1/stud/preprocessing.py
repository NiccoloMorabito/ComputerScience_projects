from typing import *
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stopwords_set = {stopword for stopword in stopwords.words("english")}

LEMMATIZER = WordNetLemmatizer()
PUNCTUATION = ",;.:?!\"\'“”()[]"


def preprocess(sentence: str, remove_stopwords: bool = False, remove_punctuation: bool = False,
               lemmatization: bool = True) -> List[str]:
    """
    :param sentence:
    :param remove_stopwords: if true, stopwords from nltk (in English) are removed from sentence
    :param remove_punctuation: if true, punctuation in ,;.:?!\"\'“”()[] is stripped away from sentence
    :param lemmatization: if true, each word in sentence is lemmatized using nltk
    :return: tokenized sentence after set preprocessing
    """
    tokens = tokenize(sentence, remove_punctuation)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords_set]
    if lemmatization:
        tokens = lemmatize(tokens)
    return tokens


def tokenize(sentence, remove_punctuation=False) -> List[str]:
    """
    Preprocess the sentences passed as parameter removing punctuation or dividing it from words.
    :param sentence:
    :param remove_punctuation: if true, punctuation is stripped away
    :return: the tokenized sentence after preprocessing
    """
    sentence = sentence.lower()
    if remove_punctuation:
        sentence = sentence.replace("'s", "").replace("’s", "")
        sentence = sentence.replace("/", " ")
        sentence = sentence.replace("-", " ")
        split = sentence.split(' ')
        return [word.strip(PUNCTUATION) for word in split if word.strip(PUNCTUATION)]
    else:
        # omologate symbols
        sentence = sentence.replace("’", "'").replace("“", "\"").replace("”", "\"")
        sentence = sentence.replace("̓", "'")
        # splitting sentence on whitespace and all other non-word-characters
        return re.findall(r"[\w]+|'s|'m|'re|[^\w\s]", sentence)


def lemmatize(tokens: List[str]) -> List[str]:
    """
    For each token in the list passed, a POS is extracted through nltk and the resulting lemma is returned.
    :param tokens:
    :return: lemmatized tokens
    """
    words_and_pos = [(word, convert_tag_to_pos(tag)) for word, tag in nltk.pos_tag(tokens)]
    return [LEMMATIZER.lemmatize(word, pos) for word, pos in words_and_pos]


def convert_tag_to_pos(nltk_tag: str):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        # default POS used by lemmatizer
        return wordnet.NOUN
