import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense
import numpy as np

# Read data from CSV file / Input Your data
df = pd.read_csv('currency_data.csv')

split_limit = int(0.8 * len(df))
training = df[:split_limit]
testing = df[split_limit:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(training['Close'].values.reshape(-1, 1))

# Model parameters
prediction_days = 5
input_shape = (prediction_days, 1)
lstm_units = 50
dropout_rate = 0.2

# Define the model architecture
input_layer = Input(shape=input_shape)

# LSTM layers with Dropout
lstm1 = LSTM(units=lstm_units, return_sequences=True)(input_layer)
dropout1 = Dropout(dropout_rate)(lstm1)

lstm2 = LSTM(units=lstm_units, return_sequences=True)(dropout1)
dropout2 = Dropout(dropout_rate)(lstm2)

lstm3 = LSTM(units=lstm_units, return_sequences=True)(dropout2)
dropout3 = Dropout(dropout_rate)(lstm3)

lstm4 = LSTM(units=lstm_units)(dropout3)
dropout4 = Dropout(dropout_rate)(lstm4)

output_layer = Dense(units=1)(dropout4)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model.fit(x_train, y_train, epochs=100, batch_size=32)
total_dataset = pd.concat((training['Close'], testing['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(testing) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

print("Predicted Prices:", predicted_prices)
