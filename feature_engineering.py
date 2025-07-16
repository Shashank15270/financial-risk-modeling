import pandas as pd
import numpy as np

def create_features(data):
    """
    Creates features for the models.
    """
    df = data.copy()
    
    # Momentum indicators
    df['SMA_10'] = df['price'].rolling(window=10).mean()
    df['SMA_30'] = df['price'].rolling(window=30).mean()
    
    # Volatility indicators
    df['VOL_10'] = df['returns'].rolling(window=10).std() * np.sqrt(252) # Annualized
    
    # Target variable for LSTM (1 for up, 0 for down)
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    df.dropna(inplace=True)
    return df