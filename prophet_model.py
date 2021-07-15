import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def train_prophet_model(df, periods=30):
    """
    Trains a Prophet model on the given DataFrame and makes future predictions.

    Args:
        df (pd.DataFrame): DataFrame with 'ds' (datestamp) and 'y' (value) columns.
        periods (int): Number of future periods to predict.

    Returns:
        pd.DataFrame: DataFrame with historical data and future predictions.
    """
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(df, forecast):
    """
    Plots the historical data and the Prophet forecast.

    Args:
        df (pd.DataFrame): Original DataFrame with 'ds' and 'y' columns.
        forecast (pd.DataFrame): Forecast DataFrame from Prophet.
    """
    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df['ds'], df['y'], 'k.', label='Observations')
    ax.plot(forecast['ds'], forecast['yhat'], ls='-', c='#0072B2', label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='#0072B2', alpha=0.2, label='Uncertainty Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Prophet Model Forecast')
    ax.legend()
    plt.tight_layout()
    plt.savefig('prophet_forecast.png')
    plt.close()

if __name__ == "__main__":
    # Example usage with dummy data
    data = {
        'ds': pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='D')),
        'y': [i + (i**0.5) * 5 + (i % 10) * 2 for i in range(100)]
    }
    df = pd.DataFrame(data)

    # Add some seasonality
    df['y'] = df['y'] + 10 * pd.np.sin(df['ds'].dt.dayofyear / 365 * 2 * pd.np.pi)

    forecast_df = train_prophet_model(df, periods=60)
    plot_forecast(df, forecast_df)
    print("Prophet model trained and forecast plotted to prophet_forecast.png")
