import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim
import sys
sys.path.append('../')
from stud.data_utils import AbsaDataModule
from stud.embedding_utils import GloveSingleton
from stud.training_utils import generate_linear_layers, compute_metrics_multiclass, make_results_reproducible,\
    get_callbacks, get_hp_combinations, merge_predictions_a_with_logits_b
# HP constants
from stud.training_utils import HIDDEN_LAYER_SIZES, LSTM_HIDDEN_LAYER_DIM, LSTM_BIDIRECTIONAL, LSTM_NUM_LAYERS, \
    DROPOUT, LOSS_WEIGHTS
# other constants
from stud.constants import CLASS_TO_SENTIMENT, NON_TARGET_CLASS, BATCH_INPUT_INDICES, BATCH_TARGET_BOOLEANS,\
    BATCH_TARGET_CLASSES, BATCH_SAMPLE, VALID_F1_MACRO


class AspectTermClassifierModel(pl.LightningModule):

    SENTIMENTS = CLASS_TO_SENTIMENT.values()

    def __init__(self, hparams, embeddings, *args, **kwargs):
        super(AspectTermClassifierModel, self).__init__(*args, **kwargs)
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
        self.classifier = nn.Sequential(*layers)

        if hparams.loss_weights is not None:
            weights = torch.tensor(hparams.loss_weights)
        else:
            weights = None
        self.loss_function = nn.CrossEntropyLoss(weight=weights, ignore_index=NON_TARGET_CLASS)

    def forward(self, x, predictions_a):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.dropout(o)
        logits = self.classifier(o)

        updated_logits = merge_predictions_a_with_logits_b(predictions_a, logits)
        predictions = torch.argmax(updated_logits, dim=2)
        return updated_logits, predictions

    def training_step(self, batch, batch_nb):
        inputs = batch[BATCH_INPUT_INDICES]
        a_booleans = batch[BATCH_TARGET_BOOLEANS]
        labels = batch[BATCH_TARGET_CLASSES]

        logits, _ = self.forward(inputs, a_booleans)

        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        inputs = batch[BATCH_INPUT_INDICES]
        a_booleans = batch[BATCH_TARGET_BOOLEANS]
        labels = batch[BATCH_TARGET_CLASSES]
        samples = batch[BATCH_SAMPLE]

        logits, predictions = self.forward(inputs, a_booleans)

        # validation metrics
        f1_micro, f1_macro = self.__get_metrics(predictions, samples)

        # We adapt the logits and labels to fit the format required for the loss function
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        sample_loss = self.loss_function(logits, labels)
        self.log('valid_loss', sample_loss, prog_bar=True)
        self.log(VALID_F1_MACRO, f1_macro, prog_bar=True)

        return f1_micro, f1_macro

    def __get_metrics(self, predictions, samples):
        sentiment_to_predictions = {sentiment: [] for sentiment in self.SENTIMENTS}
        sentiment_to_true_predictions = {sentiment: [] for sentiment in self.SENTIMENTS}

        for prediction, sample in zip(predictions, samples):
            for sentiment in self.SENTIMENTS:
                terms_sentiments = {(t, s)
                                    for t, s in self.prediction_to_terms_sentiments(prediction, sample.input_tokens)
                                    if s == sentiment}
                sentiment_to_predictions[sentiment].append(terms_sentiments)

                ground_truth = {target.get_term_and_sentiment()
                                for target in sample.targets
                                if target.sentiment == sentiment}
                sentiment_to_true_predictions[sentiment].append(ground_truth)

        metrics = compute_metrics_multiclass(sentiment_to_predictions, sentiment_to_true_predictions)
        return metrics.f1, metrics.f1_macro

    @staticmethod
    def prediction_to_terms_sentiments(prediction, words):
        targets = list()
        building_target = list()
        intermediate_sentiments = list()

        prediction = prediction[:len(words)]
        for i, el in enumerate(prediction):
            el = el.item()
            if el >= 1:
                building_target.append(words[i])
                intermediate_sentiments.append(CLASS_TO_SENTIMENT[el])
            if el == 0.0 or i == len(prediction) - 1:
                if building_target:
                    complete_target = " ".join(building_target)
                    sentiment = max(set(intermediate_sentiments), key=intermediate_sentiments.count)  # mode
                    targets.append((complete_target, sentiment))

                    # reset
                    building_target = list()
                    intermediate_sentiments = list()
        return targets

    def validation_epoch_end(self, metrics):
        f1_micros = [metric[0] for metric in metrics]
        f1_macros = [metric[1] for metric in metrics]

        print(f"\nAverage f1 for epoch {self.current_epoch}:\n"
              f"\tmicro: {sum(f1_micros) / len(f1_micros)}\n"
              f"\tmacro: {sum(f1_macros) / len(f1_macros)}\n")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    make_results_reproducible()

    vectors_list = GloveSingleton().vectors_list

    hyperparameters = {
        'embedding_dim': len(vectors_list[0]),
        'activation_function': nn.ReLU(),           # in case hidden_layers_sizes != ()
        'num_classes': len(CLASS_TO_SENTIMENT)+1    # sentiments + non-target
    }

    hp_to_possiblevalues = {
        LSTM_HIDDEN_LAYER_DIM: [128, 256],
        LSTM_BIDIRECTIONAL: [True, False],
        LSTM_NUM_LAYERS: [1, 2, 3],
        DROPOUT: [0.0, 0.1, 0.3],
        HIDDEN_LAYER_SIZES: [(300, 500, 100), (50, 20, 10)],
        LOSS_WEIGHTS: [None, [0.0, 10.0, 5.0, 1.0, 3.0], [0.0, 5.0, 4.0, 2.0, 3.0], [0.0, 2.0, 2.0, 1.0, 2.0]]
    }

    for hp_combination, combination_name in get_hp_combinations(hyperparameters, hp_to_possiblevalues):
        callbacks = get_callbacks(VALID_F1_MACRO,
                                  early_stop_callback=True, early_stop_patience=10,
                                  checkpoint_callback=False, models_dirpath=f'tuning_task_b/{combination_name}')

        trainer = pl.Trainer(
            val_check_interval=1.0,
            max_epochs=100,
            callbacks=callbacks
        )

        data_module = AbsaDataModule()
        model = AspectTermClassifierModel(hp_combination, embeddings=vectors_list)
        trainer.fit(model, datamodule=data_module)
