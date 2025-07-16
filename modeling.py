def train_lstm_model(data, time_steps=60):
    """
    Trains an LSTM model to predict market direction.
    """
    features = ['price', 'SMA_10', 'SMA_30', 'VOL_10']
    target = 'target'
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = create_lstm_dataset(pd.DataFrame(scaled_data), data[target], time_steps)
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    return model, scaler

def train_garch_model(data):
    """
    Trains a GARCH(1,1) model on the returns.
    """
    garch = arch_model(data['returns'] * 100, vol='Garch', p=1, q=1)
    model = garch.fit(disp='off')
    return model