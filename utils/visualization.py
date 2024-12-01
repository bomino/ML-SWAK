import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_feature_distributions(df):
    """Create distribution plots for numerical features."""
    figures = []
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig = px.histogram(df, x=column, title=f'Distribution of {column}')
        figures.append(fig)
    return figures

def plot_correlation_matrix(df):
    """Create correlation matrix heatmap."""
    corr = df.corr()
    fig = px.imshow(corr, 
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr.columns,
                    y=corr.columns)
    return fig

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """Create actual vs predicted scatter plot."""
    fig = go.Figure()
    fig.add_scatter(x=y_true, y=y_pred, mode='markers', name='Predictions')
    fig.add_scatter(x=y_true, y=y_true, mode='lines', name='Perfect Prediction')
    fig.update_layout(
        title=f'Actual vs Predicted ({model_name})',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values'
    )
    return fig

def plot_time_series_components(decomposition):
    """Create plots for time series decomposition."""
    figures = {
        'Trend': px.line(x=decomposition.trend.index, y=decomposition.trend.values, title='Trend'),
        'Seasonal': px.line(x=decomposition.seasonal.index, y=decomposition.seasonal.values, title='Seasonal'),
        'Residual': px.line(x=decomposition.resid.index, y=decomposition.resid.values, title='Residual')
    }
    return figures