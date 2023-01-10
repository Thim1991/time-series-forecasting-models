# Time Series Forecasting Models

This repository contains various models and utilities for time series forecasting, including traditional statistical methods and advanced deep learning approaches like LSTM networks.

## Features

- **LSTM Models**: Implementations of Long Short-Term Memory networks for sequence prediction.
- **Data Preprocessing**: Tools for scaling, windowing, and preparing time series data.
- **Visualization**: Utilities for plotting actual vs. predicted values.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training and Prediction Example

```python
import pandas as pd
import numpy as np
from time_series_model import TimeSeriesPredictor

# Dummy data
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
data = np.sin(np.linspace(0, 20, 100)) * 5 + np.random.randn(100) * 0.5 + 100
dummy_df = pd.DataFrame(data, index=dates, columns=["Value"])

predictor = TimeSeriesPredictor(look_back=10, epochs=5, batch_size=1)
predictor.train(dummy_df["Value"])

# Predict next value
next_value = predictor.predict(dummy_df["Value"])
print(f"Predicted next value: {next_value}")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
