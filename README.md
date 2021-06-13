# EASY LSTM

Dealing with simple LSTM models can be a real pain, due to the excessive ammount of preprocessing required.
This model takes very few inputs, among them, a function that returns model during the initialization, the hyperparameters to fit the model. It returns the output as a pandas DataFrame.

PD: This class only works as single-input to single-output and multiple-output to multiple-out. To work with multiple-input to single-output it should be tweaked.

# Example

```
def make_model(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    return model
```
Once the function is created, and we have our dataframe ready. Our function takea n_steps and n_features as inputs. Later on, they'll be automatically calculated.
Here's the magic part

```
model = EasyLSTM(make_model)
model.fit(dataset=df, n_steps=5, epochs=100, test_elements=20)
predictions = model.predict(n_predictions=20)
```

Model history is saved in model.history
Fit needs the parameters: dataset, epochs=20, n_steps=20
Predictions only needs the number of predictions and it'll start predicting from the last point of the training dataset. You can also pass the argument df, and it'll predict for whatever df you want it to.
Optionally it can take train_test_split=0, test_elements=0. train_test_split is the percentage of the dataset
reserved for validation. test_element is the number of elements. If both are given, train_test_split is given priority.

If we want to try several models, we just change the make_model function. If we want to try different hyperparameters, it'll be enough with fitting the same class as many times as neccesary. 
