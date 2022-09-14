import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from typing import *

from classifiers import SimplerClassifier, SequenceEncodingClassifier
from data_utils import get_dataloader_from
from embedding_utils import produce_word2vector, produce_word2index

# paths
DATASET_FOLDER_PATH = "../../data/"
TRAINSET_PATH = os.path.join(DATASET_FOLDER_PATH, 'train.jsonl')
DEVSET_PATH = os.path.join(DATASET_FOLDER_PATH, 'dev.jsonl')
GLOVE_PATH = "../../model/glove/glove.6B.50d.txt"
SIMPLER_STATE_DICT_PATH = "../../model/SimplerClassifier.statedict"
SEQUENCE_ENCODING_STATE_DICT_PATH = "../../model/SequenceEncodingClassifier.statedict"

# making the results reproducible
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, dataloader: DataLoader):
    correct_predictions = 0
    num_predictions = 0

    for tuple in dataloader:
        input = tuple[:-1]
        y = tuple[-1]
        result = model(*input)
        prediction = torch.round(result['pred'])

        correct_predictions += (prediction == y).sum()
        num_predictions += prediction.shape[0]

    accuracy = correct_predictions / num_predictions
    return {
        'name': 'Accuracy',
        'value': accuracy,
    }


def train_and_validate(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        epochs: int = 100,
        valid_dataloader: DataLoader = None,
        valid_fn: Callable = None,
        save_best_model: bool = False,
        state_dict_path: str = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 0,
        verbose: bool = True,
):
    train_history = list()
    valid_history = list()
    patience_counter = 0
    best_valid_value = 0

    for epoch in range(epochs):
        losses = list()
        correct_predictions = 0
        num_predictions = 0

        # batches of the training set
        for tuple in train_dataloader:
            y = tuple[-1]
            optimizer.zero_grad()
            result = model(*tuple)
            loss = result['loss']
            losses.append(loss)
            prediction = torch.round(result['pred'])

            correct_predictions += (prediction == y).sum()
            num_predictions += prediction.shape[0]

            loss.backward()
            optimizer.step()

        model.global_epoch += 1
        mean_loss = sum(losses) / len(losses)
        accuracy = correct_predictions / num_predictions
        train_history.append(accuracy)

        if verbose or epoch == epochs - 1:
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f}')
            print(f'  Epoch {model.global_epoch:3d} => Accuracy: {accuracy:0.6f}')

        # VALIDATION
        if valid_dataloader:
            assert valid_fn is not None
            valid_output = valid_fn(model, valid_dataloader)
            valid_name = valid_output['name']
            valid_value = valid_output['value']
            valid_history.append(valid_value.item())

            if verbose:
                print(f'    Validation => {valid_name}: {valid_value:0.6f}')

            # this saves and maintains always the best model in the state_dict_path
            if save_best_model and valid_value > best_valid_value:
                assert state_dict_path is not None
                torch.save(model.state_dict(), state_dict_path)
                best_valid_value = valid_value

            # this stops the training process after the specified patience
            if early_stopping:
                if epoch > 0 and valid_history[-1] < valid_history[-2]:
                    if patience_counter >= early_stopping_patience:
                        print('Early stop.')
                        break
                    else:
                        print('Patience.')
                        patience_counter += 1
        print()

    if save_best_model:
        print(f"The best model has {valid_name}={best_valid_value} on validation set.\n" \
              f"State has been saved at the following path: {state_dict_path}")

    return {
        'train_history': train_history,
        'valid_history': valid_history,
    }


if __name__ == '__main__':
    # SIMPLER APPROACH
    '''
    word2vector = produce_word2vector(GLOVE_PATH)
    word_emb_dict = word2vector
    sequence_encoding = False
    num_of_features = int(GLOVE_PATH.split(".")[-2][:-1]) * 2
    model = SimplerClassifier(num_of_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    state_dict_path = SIMPLER_STATE_DICT_PATH
    '''

    # SEQUENCE ENCODER APPROACH
    #'''
    word2index, vectors_list = produce_word2index(GLOVE_PATH)
    word_emb_dict = word2index
    sequence_encoding = True
    model = SequenceEncodingClassifier(vectors_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.000001)
    state_dict_path = SEQUENCE_ENCODING_STATE_DICT_PATH
    #'''

    train_dataloader = get_dataloader_from(TRAINSET_PATH, word_emb_dict, sequence_encoding=sequence_encoding)
    valid_dataloader = get_dataloader_from(DEVSET_PATH, word_emb_dict, sequence_encoding=sequence_encoding)

    logs = train_and_validate(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        valid_fn=evaluate_accuracy,
        save_best_model=True,
        state_dict_path=state_dict_path,
        early_stopping=False,
        #early_stopping_mode='max',
        #early_stopping_patience=10,
        epochs=100,
        verbose=True
    )
