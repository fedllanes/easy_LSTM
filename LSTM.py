import numpy as np
import pandas as pd


class EasyLSTM:
    def __init__(self, make_model):
        """"
        :param make_model: The function that returns the model
        """
        self.n_steps = None
        self.n_features = None
        self.make_model = make_model
        self.model = None
        self.X = None
        self.history = None
        self.columns = None

    def fit(self, dataset, train_test_split=0, test_elements=0, epochs=20, n_steps=20, *args, **kwargs):
        """
        :param dataset: Our input dataframe
        :param train_test_split: The percentage we want to reserve for testing
        :param test_elements: The number of elements we want to reserve for testing (train_test_split overrides it)
        :param epochs: The number of epochs
        :param n_steps: How far into the past we should look
        :param args: Extra arguments
        :param kwargs: Extra arguments for the fitting function, such as callbacks or verbose.
        """
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
        self.n_steps = n_steps
        self.columns = dataset.columns
        self.n_features = dataset.shape[1]
        self.model = self.make_model(self.n_steps, self.n_features)
        validation_data = None
        if train_test_split > 0:
            test_elements = int(len(dataset) * train_test_split)
        if train_test_split or test_elements:
            df_test = dataset[-test_elements:]
            validation_data = self._treat_dataframe(df_test)
            dataset = dataset[:-test_elements]
        X, y = self._treat_dataframe(dataset)
        self.X = self._format(X, y)
        self.history = self.model.fit(X, y, epochs=epochs, validation_data=validation_data, *args,
                                      **kwargs)

    def predict(self, n_predictions):
        X_entended = self._make_prediction(n_predictions)
        predictions = self._inverse_logic(X_entended)[-n_predictions:]
        predictions = pd.DataFrame(np.array(predictions), columns=self.columns)
        return predictions

    def _treat_dataframe(self, dataset):
        X_to_concat = []
        y_to_concat = []
        for element in dataset:
            series_x, series_y = self._split_sequence(dataset[element], self.n_steps)
            series_y = series_y.reshape((series_y.shape[0], 1))
            series_x = series_x.reshape((series_x.shape[0], series_x.shape[1], 1))
            X_to_concat.append(series_x)
            y_to_concat.append(series_y)
        X = np.concatenate(X_to_concat, axis=2)
        y = np.concatenate(y_to_concat, axis=1)
        return X, y

    @staticmethod
    def _split_sequence(sequence, n_steps):
        X, y = list(), list()
        sequence = sequence.reset_index(drop=True)
        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence) - 1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    @staticmethod
    def _inverse_logic(X):
        new = []
        for i in range(X.shape[0]):
            new.append(X[i][0])
        for j in range(X.shape[1]):
            if j == 0:
                continue
            new.append(X[i][j])
        return new

    def _make_prediction(self, n_predictions):
        X_new = self.X.copy()
        for i in range(n_predictions + 1):
            x_input = X_new[-1].reshape((1, self.n_steps, self.n_features))
            yhat = self.model.predict(x_input, verbose=0)
            new_element = np.concatenate([X_new[-1][1:, :], yhat])
            X_new = np.append(X_new, new_element.reshape(1, self.n_steps, self.n_features), axis=0)
        return X_new

    def _format(self, X,y):
        new_element = np.concatenate([X[-1][1:, :], y[-1].reshape(1, self.n_features)])
        X = np.concatenate([X,new_element.reshape(1,X.shape[1],X.shape[2])])
        return X
