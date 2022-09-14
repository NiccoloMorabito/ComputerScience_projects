import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl

import sys
sys.path.append('../')
from stud.data_utils import AbsaDataset, AbsaDataModule
from stud.embedding_utils import GloveSingleton
from stud.training_utils import generate_linear_layers, compute_metrics, make_results_reproducible, get_callbacks,\
    get_hp_combinations
# HP constants
from stud.training_utils import HIDDEN_LAYER_SIZES, LSTM_HIDDEN_LAYER_DIM, LSTM_BIDIRECTIONAL, LSTM_NUM_LAYERS, DROPOUT
# other constants
from stud.constants import BATCH_INPUT_INDICES, BATCH_TARGET_BOOLEANS, BATCH_SAMPLE, VALID_F1


class AspectTermIdentifierModel(pl.LightningModule):
    def __init__(self, hparams, embeddings, *args, **kwargs):
        super(AspectTermIdentifierModel, self).__init__(*args, **kwargs)
        self.save_hyperparameters(hparams)
        hparams = self.hparams

        self.word_embedding = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        layers = generate_linear_layers(lstm_output_dim, hparams.hidden_layer_sizes, hparams.activation_function,
                                        hparams.num_classes)
        layers.append(torch.nn.Sigmoid())                   # to scale results in [0,1] and to make loss function work
        self.classifier = nn.Sequential(*layers)

        self.loss_function = nn.BCELoss()

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        logits = self.classifier(o)
        predictions = torch.round(logits)
        return logits, predictions

    def training_step(self, batch, batch_nb):
        inputs = batch[BATCH_INPUT_INDICES]
        labels = batch[BATCH_TARGET_BOOLEANS]
        logits, _ = self.forward(inputs)

        # Adapting logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1).unsqueeze(1)

        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        inputs = batch[BATCH_INPUT_INDICES]
        samples = batch[BATCH_SAMPLE]
        _, predictions = self.forward(inputs)

        # validation metrics
        precision, recall, f1 = self.__get_metrics(predictions, samples)
        self.log(VALID_F1, f1, prog_bar=True)

        return precision, recall, f1

    def __get_metrics(self, predictions, sample_list):
        predicted_terms = list()
        true_terms = list()

        for prediction, sample in zip(predictions, sample_list):
            terms = set(self.__prediction_to_target_terms(prediction, sample.input_tokens))
            predicted_terms.append(terms)
            ground_truth = set(sample.target_terms)
            true_terms.append(ground_truth)

        metrics = compute_metrics(predicted_terms, true_terms)
        return metrics.precision, metrics.recall, metrics.f1

    @staticmethod
    def __prediction_to_target_terms(prediction, words):
        targets = list()
        building_target = list()

        prediction = prediction[:len(words)]
        for i, el in enumerate(prediction):
            el = el.item()
            if el == 1.0:
                building_target.append(words[i])
            if el != 1.0 or i == len(prediction)-1:
                if building_target:
                    complete_target = " ".join(building_target)
                    targets.append(complete_target)

                    building_target = list()     # reset
        return targets

    def validation_epoch_end(self, metrics):
        precisions = [metric[0] for metric in metrics]
        recalls = [metric[1] for metric in metrics]
        f1s = [metric[2] for metric in metrics]

        print(f"\nMetrics for epoch: {self.current_epoch}\n"
              f"\taverage precision is {sum(precisions) / len(precisions)}\n"
              f"\taverage recall is {sum(recalls) / len(recalls)}\n"
              f"\taverage f1 is {sum(f1s) / len(f1s)}\n")

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


if __name__ == '__main__':
    make_results_reproducible()

    vectors_list = GloveSingleton().vectors_list

    hyperparameters = {
        'embedding_dim': len(vectors_list[0]),
        'activation_function': nn.ReLU(),       # in case hidden_layers_sizes != ()
        'num_classes': 1                        # target and non-target
    }

    hp_to_possiblevalues = {
        HIDDEN_LAYER_SIZES: [(), (500, 100), (200, 100, 10)],
        LSTM_HIDDEN_LAYER_DIM: [128, 256],
        LSTM_BIDIRECTIONAL: [False, True],
        LSTM_NUM_LAYERS: [1, 2, 3],
        DROPOUT: [0.0, 0.1, 0.5]
    }

    for hp_combination, combination_name in get_hp_combinations(hyperparameters, hp_to_possiblevalues):
        callbacks = get_callbacks(VALID_F1,
                                  early_stop_callback=True, early_stop_patience=5,
                                  checkpoint_callback=False, models_dirpath=f'tuning_task_a/{combination_name}')

        trainer = pl.Trainer(
            val_check_interval=1.0,
            max_epochs=100,
            callbacks=callbacks
        )

        data_module = AbsaDataModule()
        model = AspectTermIdentifierModel(hp_combination, embeddings=vectors_list)
        trainer.fit(model, datamodule=data_module)
