import pandas as pd
import numpy as np

def simulate_financial_data(days=1000, initial_price=100):
    """
    Simulates realistic financial time-series data.
    Includes price movements with changing volatility.
    """
    returns = np.random.normal(loc=0.0005, scale=0.01, size=days)
    # Introduce some volatility shocks (market regimes)
    for i in range(0, days, 252): # Yearly shock
        shock_period = np.random.normal(loc=-0.001, scale=0.03, size=60)
        returns[i:i+60] = shock_period
    
    price = initial_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({'price': price})
    df['returns'] = df['price'].pct_change()
    df.dropna(inplace=True)
    
    return df