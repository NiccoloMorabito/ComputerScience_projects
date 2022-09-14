from typing import *
import matplotlib.pyplot as plt
from data_utils import get_dataloader_from
from training import DEVSET_PATH, GLOVE_PATH
import torch
from embedding_utils import produce_word2vector, produce_word2index
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from classifiers import SimplerClassifier, SequenceEncodingClassifier
import pandas as pd
import seaborn as sn

SIMPLER_STATE_DICT = "../../model/SimplerClassifier.statedict"
SEQUENCE_ENCODING_STATE_DICT = "../../model/SequenceEncodingClassifier.statedict"

CLASSES = [True, False]


def plot_accuracy_from_logs(logs: Dict, title: str):
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(logs["train_history"]))), logs["train_history"], label="Train accuracy")
    plt.plot(list(range(len(logs["valid_history"]))), logs["valid_history"], label="Test accuracy")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")

    plt.show()


def print_metrics_simpler_classifier():
    word2vector = produce_word2vector(GLOVE_PATH)
    num_of_features = int(GLOVE_PATH.split(".")[-2][:-1]) * 2
    simpler_classifier = SimplerClassifier(num_of_features)
    simpler_classifier.load_state_dict(torch.load(SIMPLER_STATE_DICT))
    simpler_classifier.eval()
    dataloader = get_dataloader_from(DEVSET_PATH, word2vector, sequence_encoding=False)

    print("SIMPLER CLASSIFIER - METRICS")
    print_metrics(simpler_classifier, dataloader)


def print_metrics_sequence_encoding_classifier():
    word2index, vectors_list = produce_word2index(GLOVE_PATH)
    sequence_encoding_classifier = SequenceEncodingClassifier(vectors_list)
    sequence_encoding_classifier.load_state_dict(torch.load(SEQUENCE_ENCODING_STATE_DICT))
    sequence_encoding_classifier.eval()
    dataloader = get_dataloader_from(DEVSET_PATH, word2index, sequence_encoding=True)

    print("SEQUENCE ENCODING CLASSIFIER - METRICS")
    print_metrics(sequence_encoding_classifier, dataloader)


def print_metrics(classifier: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    y_true = list()
    y_pred = list()
    for tuple in dataloader:
        input = tuple[:-1]
        y = tuple[-1]
        y_true += y.tolist()

        batch_out = classifier(*input)
        preds = torch.round(batch_out['pred'])

        y_pred += preds.tolist()

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Precision: {precision:0.6f}")
    print(f"Recall: {recall:0.6f}")
    print(f"F1-score: {f1:0.6f}")
    print(f"Accuracy: {accuracy:0.6f}")
    print(f"Confusion matrix:\n {cm}")
    plot_confusion_matrix(cm)


def plot_confusion_matrix(cm: List[List]):
    df_cfm = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show(cfm_plot)


if __name__=='__main__':
    print_metrics_simpler_classifier()
    print_metrics_sequence_encoding_classifier()