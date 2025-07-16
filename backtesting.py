import pandas as pd
import numpy as np
from modeling import create_lstm_dataset
from sklearn.preprocessing import MinMaxScaler

def run_backtest(test_data, lstm_model, scaler, garch_model, time_steps=60, vol_threshold=0.3):
    """
    Runs a backtest of the combined LSTM-GARCH strategy.
    """
    features = ['price', 'SMA_10', 'SMA_30', 'VOL_10']
    
    # Scale test data
    scaled_test_data = scaler.transform(test_data[features])
    
    X_test, y_test = create_lstm_dataset(pd.DataFrame(scaled_test_data), test_data['target'], time_steps)
    
    # Get LSTM predictions
    lstm_predictions = (lstm_model.predict(X_test) > 0.5).astype(int)
    
    # Get GARCH volatility forecasts
    garch_forecasts = garch_model.forecast(horizon=len(test_data) - time_steps)
    volatility_forecast = garch_forecasts.variance.iloc[-1].values / 100 # Convert back from %
    
    # Align data for strategy
    strategy_df = test_data.iloc[time_steps:].copy()
    strategy_df['lstm_pred'] = lstm_predictions
    strategy_df['garch_vol_forecast'] = volatility_forecast
    
    # Strategy Logic: Buy if LSTM predicts UP and GARCH volatility is below threshold
    strategy_df['signal'] = np.where(
        (strategy_df['lstm_pred'] == 1) & (strategy_df['garch_vol_forecast'] < vol_threshold), 
        1, 0
    )
    
    # Calculate strategy returns
    strategy_df['strategy_returns'] = strategy_df['returns'] * strategy_df['signal'].shift(1)
    
    # Calculate cumulative returns
    strategy_df['cumulative_returns'] = (1 + strategy_df['returns']).cumprod()
    strategy_df['cumulative_strategy_returns'] = (1 + strategy_df['strategy_returns']).cumprod()
    
    return strategy_df