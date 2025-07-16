import plotly.graph_objects as go
import numpy as np

def generate_report(results_df):
    """
    Generates an interactive HTML report of the backtest results.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['cumulative_returns'],
        mode='lines',
        name='Buy and Hold'
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['cumulative_strategy_returns'],
        mode='lines',
        name='LSTM-GARCH Strategy'
    ))

    # Calculate metrics
    sharpe_ratio = (results_df['strategy_returns'].mean() / results_df['strategy_returns'].std()) * np.sqrt(252)
    
    fig.update_layout(
        title=f'Backtest Performance: LSTM-GARCH Strategy vs. Buy & Hold<br>Annualized Sharpe Ratio: {sharpe_ratio:.2f}',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        legend_title='Strategy',
        template='plotly_white'
    )
    
    fig.write_html("performance_report.html")