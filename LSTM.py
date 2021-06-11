import numpy as np


class EasyLSTM:
    def __init__(self, make_model, n_steps=20, n_features=1):
        self.n_steps = n_steps
        self.n_features = n_features
        self.model = make_model(self.n_steps, self.n_features)
        self.X = []
        self.history = []

    def fit(self, dataset, train_test_split=0, epochs=20, verbose=0):
        X, y = self._treat_dataframe(dataset, self.n_steps)
        self.X = X
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose)

    def predict(self, n_predictions):
        X_entended = self._make_prediction(n_predictions)
        predictions = self._inverse_logic(X_entended)[-n_predictions:]
        return predictions

    def _treat_dataframe(self, dataset, n_steps):
        X_to_concat = []
        y_to_concat = []
        for element in dataset:
            series_x, series_y = self._split_sequence(dataset[element], n_steps)
            series_y = series_y.reshape((series_y.shape[0], 1))
            series_x = series_x.reshape((series_x.shape[0], series_x.shape[1], 1))
            X_to_concat.append(series_x)
            y_to_concat.append(series_y)
        X = np.concatenate(X_to_concat, axis=2)
        y = np.concatenate(y_to_concat, axis=1)
        return X,y


    @staticmethod
    def _split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # Encontrar el final de la secuencia
            end_ix = i + n_steps
            # Verificar que no nos hayamos pasado
            if end_ix > len(sequence) - 1:
                break
            # Agregar los valores a las salidas
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    @staticmethod
    def _inverse_logic(X):
        new = []  # Lista donde se guardará el array de los valores de entrenamiento
        # sumado a lo predicho
        for i in range(X.shape[0]):  # Iteramos sobre el primer eje, es decir la cantidad
            # de listas de listas que tenemos
            new.append(X[i][0])  # Guardamos el primer elemento de cada eje
        for j in range(X.shape[1]):  # Iteramos sobre el segundo eje, es decir
            # la cantidad de elementos que vemos hacía el pasado
            if j == 0:
                continue  # Este elemento ya fue agregado
            new.append(X[i][j])  # Agregamos cada elemento del último elemento de la
            # lista de lista a nuestra nueva lista
        return new

    def _make_prediction(self, n_predictions):
        X_new = self.X.copy()
        for i in range(n_predictions + 1):
            x_input = X_new[-1].reshape((1, self.n_steps, self.n_features))
            yhat = self.model.predict(x_input, verbose=0)
            new_element = np.concatenate([X_new[-1][1:, :].reshape(self.n_steps - 1, self.n_features), yhat])
            X_new = np.append(X_new, new_element.reshape(1, self.n_steps, self.n_features),
                              axis=0)
        return X_new
