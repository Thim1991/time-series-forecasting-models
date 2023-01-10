import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class TimeSeriesPredictor:
    def __init__(self, look_back=60, epochs=50, batch_size=32):
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _create_dataset(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), 0]
            X.append(a)
            Y.append(dataset[i + self.look_back, 0])
        return np.array(X), np.array(Y)

    def train(self, data_series):
        # Scale the data
        data = data_series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        # Create dataset for LSTM
        X_train, y_train = self._create_dataset(scaled_data)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, data_series):
        # Prepare input for prediction
        data = data_series.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Take the last 'look_back' points for prediction
        last_sequence = scaled_data[-self.look_back:].reshape(1, self.look_back, 1)
        
        # Predict next value
        predicted_scaled_value = self.model.predict(last_sequence)
        predicted_value = self.scaler.inverse_transform(predicted_scaled_value)
        return predicted_value[0][0]

    def plot_predictions(self, original_data, predictions, title="Time Series Prediction"):
        plt.figure(figsize=(12, 6))
        plt.plot(original_data.index[-len(predictions):], original_data.values[-len(predictions):], color="blue", label="Actual Price")
        plt.plot(original_data.index[-len(predictions):], predictions, color="red", label="Predicted Price")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Generate some dummy time series data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    data = np.sin(np.linspace(0, 100, 500)) * 10 + np.random.randn(500) * 2 + 50
    dummy_df = pd.DataFrame(data, index=dates, columns=["Value"])

    predictor = TimeSeriesPredictor(look_back=30, epochs=10, batch_size=16)
    predictor.train(dummy_df["Value"])

    # Make predictions for the next 5 days
    future_predictions = []
    current_series = dummy_df["Value"].copy()
    for _ in range(5):
        next_prediction = predictor.predict(current_series)
        future_predictions.append(next_prediction)
        # Append the prediction to the series to predict the next day
        current_series = pd.concat([current_series, pd.Series([next_prediction], index=[current_series.index[-1] + pd.Timedelta(days=1)])])

    print("Future predictions:", future_predictions)
    # predictor.plot_predictions(dummy_df["Value"], future_predictions)
