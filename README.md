# price-prediction-with-LSTM

This Python script uses an LSTM (Long Short-Term Memory) neural network to predict cryptocurrency prices based on historical data. The model is trained on past closing prices and can be used to forecast future prices.

## Features

- Reads historical cryptocurrency data from a CSV file.
- Preprocesses data using MinMaxScaler.
- Defines and trains an LSTM neural network with dropout layers to prevent overfitting.
- Predicts future prices based on the trained model.

## Prerequisites

- Python 3.7 or later
- Pandas
- NumPy
- scikit-learn
- Keras with TensorFlow backend

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/crypto-price-prediction.git
    cd crypto-price-prediction
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install pandas numpy scikit-learn tensorflow keras
    ```

## Usage

1. Prepare your dataset:
   - Ensure you have a CSV file named `currency_data.csv` with a `Close` column containing the historical closing prices of the cryptocurrency you want to predict.

2. Run the script:

    ```sh
    python main.py
    ```

3. The script will print the predicted prices.

## Customization

- To use a different CSV file, update the `currency_data.csv` file path in the script:

    ```python
    df = pd.read_csv('your_data.csv')
    ```

- Adjust the `prediction_days` variable to change the number of days used for making predictions:

    ```python
    prediction_days = 5
    ```

- Modify the model parameters such as `lstm_units`, `dropout_rate`, `epochs`, and `batch_size` as needed:

    ```python
    lstm_units = 50
    dropout_rate = 0.2
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    ```
