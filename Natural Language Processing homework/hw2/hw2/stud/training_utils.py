import torch
import numpy as np
import itertools
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple

# HP names
DROPOUT = "dropout"
HIDDEN_LAYER_SIZES = "hidden_layer_sizes"
LOSS_WEIGHTS = "loss_weights"
LSTM_BIDIRECTIONAL = "bidirectional"
LSTM_NUM_LAYERS = "num_layers"
LSTM_HIDDEN_LAYER_DIM = "hidden_dim"

# METRICS
TP = "tp"
FP = "fp"
FN = "fn"
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"


def make_results_reproducible():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_hp_combinations(hp_to_fixedvalue: Dict[str, Any],
                        hp_to_possiblevalues: Dict[str, List[Any]]) -> List[Tuple[Dict[str, Any], List]]:
    """
    Creates a list of hyper-parameters configurations by merging all the possible configuration (obtained combining
     hp_to_possiblevalues) with hp_to_fixedvalue.
    Args:
        hp_to_fixedvalue: dictionary hyper-parameter name to its fixed value
        hp_to_possiblevalues: dictionary hyper-parameter name to a list of possible values
    Returns:
        list of couples where:
        - the first element is a dictionary hyper-parameter name to value
        - the second element is the name of that configuration, based on the values of hyper-parameters
    """
    keys, values = zip(*hp_to_possiblevalues.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    combinations_and_names = list()
    for permutations_dict in permutations_dicts:
        name = list(permutations_dict.values())                     # name of configuration is based on hp values
        permutations_dict.update(hp_to_fixedvalue)                  # adding fixed hp
        combinations_and_names.append((permutations_dict, name))
    return combinations_and_names


def get_callbacks(metric_to_monitor: str, early_stop_callback: bool = False, early_stop_patience: int = None,
                  checkpoint_callback: bool = False, models_dirpath: str = None) -> List:
    """
    Args:
        metric_to_monitor: name of the metric the two callbacks should be based on
        early_stop_callback: if true, the early-stop callback is returned
        early_stop_patience: the patience of the early-stop callback
        checkpoint_callback: if true, the checkpoint callback is returned
        models_dirpath: the path of the directory the checkpoint callback saves models into

    Returns:
        a list with the requested callbacks
    """
    callbacks = list()

    # early stop callback that stops training after the specified patience
    if early_stop_callback:
        assert early_stop_patience is not None and early_stop_patience > 1,\
            "For early-stop callback it is necessary to specify a patience greater than 0."
        early_stop_cb = pl.callbacks.EarlyStopping(
            monitor=metric_to_monitor,
            patience=early_stop_patience,
            verbose=True,
            mode='max',
        )
        callbacks.append(early_stop_cb)

    # checkpoint callback that saves the best model according to metric_to_monitor
    if checkpoint_callback:
        assert models_dirpath is not None, "For checkpoint callback it is necessary to specify a models_dirpath."
        filename = '{epoch}-{' + metric_to_monitor + ':.4f}'
        check_point_cb = pl.callbacks.ModelCheckpoint(
            monitor=metric_to_monitor,
            verbose=True,
            save_top_k=1,
            mode='max',
            dirpath=models_dirpath,
            filename=filename
        )
        callbacks.append(check_point_cb)

    return callbacks


def generate_linear_layers(input_size: int, hidden_layer_sizes: Tuple[int],
                           activation_function, num_classes: int) -> List:
    """
    Returns a list of linear layers following the specified sizes and alternating them with the specified activation
    function.
    Args:
        input_size:
        hidden_layer_sizes: tuple of n elements, where n is the number of desired hidden layers, each element is the
            number of units the corresponding hidden layer should have
        activation_function:
        num_classes: the number of output units
    """
    layers = list()
    prev = input_size
    if hidden_layer_sizes:
        for hidden_layer_size in hidden_layer_sizes:
            layers.append(torch.nn.Linear(prev, hidden_layer_size))
            layers.append(activation_function)
            prev = hidden_layer_size

    layers.append(torch.nn.Linear(prev, num_classes))
    return layers


def merge_predictions_a_with_logits_b(predictions_a, logits_b):
    """
    Merges predictions from A with the logits outputted by B
    Args:
        logits_b: output of model B
        predictions_a: predictions of model A

    Returns:
        updated logits
    """
    with torch.no_grad():
        for logits_b_, predictions_a_ in zip(logits_b, predictions_a):
            for el_b, el_a in zip(logits_b_, predictions_a_):
                # force that class 0 is predicted in case A predicted non-target for this el
                if int(el_a.item()) == 0:
                    el_b[0] = 9999999
                # avoid that class 0 is predicted in case A predicted target for this el
                else:
                    el_b[0] = -9999999
    return logits_b


class Metrics:
    def __init__(self, tp, fp, fn, precision, recall, f1):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.precision = precision
        self.recall = recall
        self.f1 = f1


class MetricsMultiClass:
    def __init__(self, tp, fp, fn, precision, recall, f1, precision_macro, recall_macro, f1_macro):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.precision_macro = precision_macro
        self.recall_macro = recall_macro
        self.f1_macro = f1_macro


def compute_metrics(predictions, ground_truths, verbose=False):
    tp = 0
    fp = 0
    fn = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        tp += len(prediction & ground_truth)
        fp += len(prediction - ground_truth)
        fn += len(ground_truth - prediction)

    try:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0

    if verbose:
        __print_all_metrics(tp, fp, fn, precision, recall, f1)

    return Metrics(tp, fp, fn, precision, recall, f1)


def __print_all_metrics(tp, fp, fn, precision, recall, f1):
    print(f"Metrics:")
    print("\t TP: {};\tFP: {};\tFN: {}".format(tp, fp, fn))
    print("\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f}".format(precision, recall, f1))


def compute_metrics_multiclass(predictions, ground_truths, verbose=False):
    classes = list(predictions.keys())
    scores = {c: {TP: 0, FP: 0, FN: 0} for c in classes + ["ALL"]}

    for class_ in classes:
        for prediction, ground_truth in zip(predictions[class_], ground_truths[class_]):
            scores[class_][TP] += len(prediction & ground_truth)
            scores[class_][FP] += len(prediction - ground_truth)
            scores[class_][FN] += len(ground_truth - prediction)

    for class_ in scores.keys():
        if scores[class_][TP]:
            scores[class_][PRECISION] = scores[class_][TP] / (scores[class_][FP] + scores[class_][TP]) * 100
            scores[class_][RECALL] = scores[class_][TP] / (scores[class_][FN] + scores[class_][TP]) * 100
        else:
            scores[class_][PRECISION], scores[class_][RECALL] = 0, 0

        if not scores[class_][PRECISION] + scores[class_][RECALL] == 0:
            scores[class_][F1] = 2 * scores[class_][PRECISION] * scores[class_][RECALL] \
                                 / (scores[class_][PRECISION] + scores[class_][RECALL])
        else:
            scores[class_][F1] = 0

    tp = sum([scores[class_][TP] for class_ in classes])
    fp = sum([scores[class_][FP] for class_ in classes])
    fn = sum([scores[class_][FN] for class_ in classes])

    precision, recall, f1 = __get_micro_metrics(tp, fp, fn)

    precisions = [scores[class_][PRECISION] for class_ in classes]
    recalls = [scores[class_][RECALL] for class_ in classes]
    f1s = [scores[class_][F1] for class_ in classes]

    precision_macro, recall_macro, f1_macro = __get_macro_metrics(precisions, recalls, f1s)

    if verbose:
        __print_all_metrics_multiclass(tp, fp, fn, precision, recall, f1, precision_macro, recall_macro, f1_macro,
                                       scores, classes)

    return MetricsMultiClass(tp, fp, fn, precision, recall, f1, precision_macro, recall_macro, f1_macro)


def __get_micro_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1


def __get_macro_metrics(precisions: List[float], recalls: List[float], f1s: List[float]) -> Tuple[float, float, float]:
    precision_macro = sum(precisions) / len(precisions)
    recall_macro = sum(recalls) / len(recalls)
    f1_macro = sum(f1s) / len(f1s)

    return precision_macro, recall_macro, f1_macro


def __print_all_metrics_multiclass(tp, fp, fn,
                                   precision, recall, f1,
                                   precision_macro, recall_macro, f1_macro,
                                   scores, classes):

    print("Overall metrics")
    print("\tTP: {};\tFP: {};\tFN: {}"
          .format(tp, fp, fn))
    print("\t(micro): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)"
          .format(precision, recall, f1))
    print("\t(macro): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n"
          .format(precision_macro, recall_macro, f1_macro))

    print("Per class metrics")
    for class_ in classes:
        print("\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            class_,
            scores[class_][TP],
            scores[class_][FP],
            scores[class_][FN],
            scores[class_][PRECISION],
            scores[class_][RECALL],
            scores[class_][F1],
            scores[class_][TP] + scores[class_][FP]))
