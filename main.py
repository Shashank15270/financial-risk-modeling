import pandas as pd
import numpy as np
from data_handler import simulate_financial_data, save_to_db, load_from_db
from feature_engineering import create_features
from modeling import train_lstm_model, train_garch_model
from backtesting import run_backtest
from reporting import generate_report
import warnings

warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the financial risk modeling pipeline.
    """
    print("Starting the Advanced Financial Risk Modeling pipeline...")

    # --- 1. Data Simulation & Storage ---
    print("\nStep 1: Simulating financial data...")

    data = simulate_financial_data(days=252*5) # 5 years of trading data
    

    data.to_csv('simulated_financial_data.csv')
    print("Data simulation complete. Saved to 'simulated_financial_data.csv'")

    # --- 2. Feature Engineering ---
    print("\nStep 2: Engineering features...")
    featured_data = create_features(data)
    print("Feature engineering complete.")

    # --- 3. Modeling ---
    print("\nStep 3: Training predictive models...")
    
    # Split data into training and testing sets
    train_size = int(len(featured_data) * 0.8)
    train_data = featured_data[:train_size]
    test_data = featured_data[train_size:]

    # Train LSTM for directional prediction
    print("Training LSTM model for directional prediction...")
    lstm_model, scaler = train_lstm_model(train_data)
    
    # Train GARCH for volatility prediction
    print("Training GARCH model for volatility forecasting...")
    garch_model = train_garch_model(train_data)
    
    print("Model training complete.")

    # --- 4. Backtesting ---
    print("\nStep 4: Running backtest on the test data...")
    backtest_results = run_backtest(test_data, lstm_model, scaler, garch_model)
    print("Backtesting complete.")

    # --- 5. Reporting ---
    print("\nStep 5: Generating performance report...")
    generate_report(backtest_results)
    print("Performance report generated: 'performance_report.html'")
    
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
