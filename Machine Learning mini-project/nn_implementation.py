import torch
import sklearn.metrics

# making the results reproducible
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# constants
PRED_LABEL = "pred"
LOSS_LABEL = "loss"
F1_AVERAGE = "macro"
ACTIVATIONS = "activations"
HIDDEN_LAYER_SIZES = "hidden_layer_sizes"
EARLY_STOP_PATIENCES = "early_stop_patiences"
TRAIN_HISTORY = "train_history"
VALID_HISTORY = "valid_history"


class HappyNeuralNetwork(torch.nn.Module):

    def __init__(self, input_size, hidden_layer_sizes, activation_function):
        super().__init__()
        layers = self.__produce_layers__(input_size, hidden_layer_sizes, activation_function)
        self.projection_layer = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.MSELoss()
        self.global_epoch = 0

    @staticmethod
    def __produce_layers__(input_size, hidden_layer_sizes, activation_function):
        layers = [torch.nn.Linear(input_size, hidden_layer_sizes[0]), activation_function]
        prev = hidden_layer_sizes[0]
        for hidden_layer_size in hidden_layer_sizes[1:]:
            layers.append(torch.nn.Linear(prev, hidden_layer_size))
            layers.append(activation_function)
            prev = hidden_layer_size
        layers.append(torch.nn.Linear(prev, 1))
        return layers

    def forward(self, x, y=None):
        pred = self.projection_layer(x.float()).squeeze()
        result = {PRED_LABEL: pred}

        # compute loss
        if y is not None:
            loss = self.loss(pred, y.float())
            result[LOSS_LABEL] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)


@torch.no_grad()
def evaluate_f1_macro(model: torch.nn.Module, X_val, y_val):
    y_true = list()
    y_pred = list()

    for index, X in enumerate(X_val):
        y = convert(y_val[index])
        y_true.append(y)
        result = model(X)
        y_pred.append(round_and_convert(result[PRED_LABEL]))

    return sklearn.metrics.f1_score(y_val, y_pred, average=F1_AVERAGE)


def train_and_validate(
        model,
        optimizer,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        early_stopping=False,
        early_stop_patience=0,
        verbose=True,
):
    train_history = list()
    valid_history = list()
    patience_counter = 0

    for epoch in range(epochs):
        losses = list()
        y_pred = list()
        y_true = list()

        for index, X in enumerate(X_train):
            y = y_train[index]
            y_true.append(convert(y))

            optimizer.zero_grad()
            result = model(X, y)
            loss = result[LOSS_LABEL]
            losses.append(loss)
            y_pred.append(round_and_convert(result[PRED_LABEL]))

            loss.backward()
            optimizer.step()

        model.global_epoch += 1
        mean_loss = sum(losses) / len(losses)
        f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average=F1_AVERAGE)
        train_history.append(f1_macro)

        if verbose:
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f}')
            print(f'  Epoch {model.global_epoch:3d} => F1-macro: {f1_macro:0.6f}')

        # VALIDATION
        f1_macro = evaluate_f1_macro(model, X_val, y_val)
        valid_history.append(f1_macro.item())

        if verbose:
            print(f'    Validation => F1-macro: {f1_macro:0.6f}')

        # this stops the training process after the specified patience
        if early_stopping:
            if epoch > 0 and valid_history[-1] < valid_history[-2]:
                if patience_counter >= early_stop_patience:
                    if verbose: print('Early stop.')
                    break
                else:
                    if verbose: print('Patience.')
                    patience_counter += 1
        if verbose: print()

    return {
        TRAIN_HISTORY: train_history,
        VALID_HISTORY: valid_history,
    }


class NeuralNetworkClassifier:

    def __init__(self, n_features, hidden_layer_sizes, activation, early_stop_patience):
        self.model = HappyNeuralNetwork(n_features, hidden_layer_sizes, activation)
        self.early_stop_patience = early_stop_patience
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def fit(self, X_train, y_train, X_val, y_val):
        return train_and_validate(self.model, self.optimizer,
                                  X_train, y_train, X_val, y_val,
                                  epochs=100,
                                  early_stopping=True, early_stop_patience=self.early_stop_patience,
                                  verbose=False)

    @torch.no_grad()
    def predict(self, X_test):
        predictions = list()
        for X in X_test:
            result = self.model(X)
            predictions.append(round_and_convert(result[PRED_LABEL]))
        return predictions


def convert(tens):
    return int(tens.item())


def round_and_convert(tens):
    prediction = int(torch.round(tens).item())
    if prediction > 4:
        prediction = 4
    return prediction


class NeuralNetworkGridCV:

    def __init__(self, n_features, param_grid):
        self.n_features = n_features
        self.param_grid = param_grid
        self.act_str_to_torch = {
            'sigmoid': torch.nn.Sigmoid(),
            'hsigmoid': torch.nn.Hardsigmoid(),
            'relu': torch.nn.ReLU(),
            'prelu': torch.nn.PReLU(),
            'tanh': torch.nn.Tanh(),
        }
        self.best_score_ = 0
        self.best_params_ = dict()
        self.best_estimator_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        activations = [value for key, value in self.act_str_to_torch.items() if key in self.param_grid[ACTIVATIONS]]
        hidden_layer_sizes = self.param_grid[HIDDEN_LAYER_SIZES]
        early_stop_patiences = self.param_grid[EARLY_STOP_PATIENCES]

        for activation in activations:
            for hidden_layer_size in hidden_layer_sizes:
                for early_stop_patience in early_stop_patiences:
                    estimator = NeuralNetworkClassifier(self.n_features, hidden_layer_size,
                                                        activation, early_stop_patience)
                    logs = estimator.fit(X_train, y_train, X_val, y_val)
                    best_score = logs["valid_history"][-1]
                    if best_score > self.best_score_:
                        self.best_score_ = best_score
                        self.best_params_[ACTIVATIONS] = activation
                        self.best_params_[HIDDEN_LAYER_SIZES] = hidden_layer_size
                        self.best_params_[EARLY_STOP_PATIENCES] = early_stop_patience
                        self.best_estimator_ = estimator


def transform_train_val_for_pytorch(X_train, y_train, X_val, y_val):
    return transform_data_for_pytorch(X_train, y_train) +\
            transform_data_for_pytorch(X_val, y_val)


def transform_data_for_pytorch(X, y):
    return torch.from_numpy(X), torch.from_numpy(y.cat.codes.to_numpy(copy=True))
