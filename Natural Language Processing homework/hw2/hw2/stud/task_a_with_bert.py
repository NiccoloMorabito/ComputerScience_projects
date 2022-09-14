import pytorch_lightning as pl
import torch
import torch.optim as optim
from transformers import BertTokenizerFast, BertModel
from torch import nn
import sys
sys.path.append('../')
from stud.data_utils import AbsaDataset, AbsaDataModule
from stud.training_utils import generate_linear_layers, compute_metrics, make_results_reproducible, get_callbacks,\
    get_hp_combinations
# HP constants
from stud.training_utils import HIDDEN_LAYER_SIZES, LSTM_HIDDEN_LAYER_DIM, LSTM_BIDIRECTIONAL, LSTM_NUM_LAYERS, DROPOUT
# other constants
from stud.constants import BATCH_SAMPLE, BATCH_BERT_ENCODINGS, BATCH_BERT_TARGET_BOOLEANS, BATCH_BERT_TARGET_CLASSES,\
    VALID_F1


class AspectTermIdentifierModel(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(AspectTermIdentifierModel, self).__init__(*args, **kwargs)
        self.save_hyperparameters(hparams)
        hparams = self.hparams

        # embeddings
        self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()

        self.lstm = nn.LSTM(self.bert.config.hidden_size, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0)
        self.dropout = nn.Dropout(hparams.dropout)

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        layers = generate_linear_layers(lstm_output_dim, hparams.hidden_layer_sizes, hparams.activation_function,
                                        hparams.num_classes)
        layers.append(torch.nn.Sigmoid())  # to scale results in [0,1] and to make loss function work
        self.classifier = nn.Sequential(*layers)

        self.loss_function = nn.BCELoss()

    def forward(self, encodings):
        embeddings = self.word_embedding(encodings)
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        logits = self.classifier(o)
        predictions = torch.round(logits)
        return logits, predictions

    def word_embedding(self, encoding):
        with torch.no_grad():
            outputs = self.bert(**encoding)[0]
        return outputs

    def training_step(self, batch, batch_nb):
        encodings = batch[BATCH_BERT_ENCODINGS]
        labels = batch[BATCH_BERT_TARGET_BOOLEANS]
        logits, _ = self.forward(encodings)

        # Adapting logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1).unsqueeze(1)

        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        encodings = batch[BATCH_BERT_ENCODINGS]
        samples = batch[BATCH_SAMPLE]
        _, predictions = self.forward(encodings)

        # validation metrics
        precision, recall, f1 = self.__get_metrics(predictions, encodings, samples)
        self.log(VALID_F1, f1, prog_bar=True)

        return precision, recall, f1

    def __get_metrics(self, predictions, encodings, samples):
        predicted_terms = list()
        true_terms = list()
        for logit, input_ids, sample in zip(predictions, encodings["input_ids"], samples):
            terms = set(self.__prediction_to_target_terms(logit, input_ids))
            predicted_terms.append(terms)
            ground_truth = set(sample.target_terms)
            true_terms.append(ground_truth)

        metrics = compute_metrics(predicted_terms, true_terms)
        return metrics.precision, metrics.recall, metrics.f1

    def __prediction_to_target_terms(self, prediction, input_ids):
        targets = list()
        building_target_indices = list()

        for el, input_id in zip(prediction, input_ids):
            el = el.item()
            if el == 1.0:
                building_target_indices.append(input_id)
            else:
                if building_target_indices:
                    target = self.bert_tokenizer.decode(building_target_indices)
                    targets.append(target)

                    building_target_indices = list()   # reset
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

    hyperparameters = {
        'activation_function': nn.ReLU(),   # in case hidden_layers_sizes != ()
        'num_classes': 1                    # target and non-target
    }

    hp_to_possiblevalues = {
        LSTM_HIDDEN_LAYER_DIM: [128, 256],
        LSTM_BIDIRECTIONAL: [False, True],
        LSTM_NUM_LAYERS: [1, 2, 3],
        DROPOUT: [0.0, 0.1, 0.5],
        HIDDEN_LAYER_SIZES: [(), (500, 100), (200, 100, 10)]
    }

    for hp_combination, combination_name in get_hp_combinations(hyperparameters, hp_to_possiblevalues):
        callbacks = get_callbacks(VALID_F1,
                                  early_stop_callback=True, early_stop_patience=5,
                                  checkpoint_callback=False, models_dirpath=f'tuning_task_a_with_bert/{combination_name}')

        trainer = pl.Trainer(
            val_check_interval=1.0,
            max_epochs=100,
            callbacks=callbacks
        )

        data_module = AbsaDataModule(for_bert=True)
        model = AspectTermIdentifierModel(hp_combination)
        trainer.fit(model, datamodule=data_module)
