import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def render_time_series_stats(ts_df):
    """Render basic time series statistics in a modern card layout"""
    st.markdown("### 📈 Time Series Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Date Range",
            f"{ts_df.index.min().strftime('%Y-%m-%d')}",
            f"to {ts_df.index.max().strftime('%Y-%m-%d')}"
        )
    with col2:
        time_span = (ts_df.index.max() - ts_df.index.min()).days
        st.metric(
            "Time Span",
            f"{time_span} days",
            f"{time_span/30:.1f} months"
        )
    with col3:
        st.metric(
            "Data Points",
            f"{len(ts_df):,}",
            f"Frequency: {infer_frequency(ts_df)}"
        )

def create_visualization(ts_df, target_col, viz_type, freq=None, window=None):
    """Create time series visualization based on selected type"""
    if viz_type == "Raw Data":
        fig = px.line(
            ts_df, 
            y=target_col, 
            title=f'{target_col} Over Time',
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=target_col,
            showlegend=True,
            height=500
        )
        return fig
        
    elif viz_type == "Resampled Data":
        resampled = ts_df[target_col].resample(freq).agg(['mean', 'min', 'max'])
        fig = go.Figure()
        
        # Add traces with custom styling
        fig.add_trace(go.Scatter(
            x=resampled.index, 
            y=resampled['mean'], 
            name='Mean',
            line=dict(color='#4F46E5', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=resampled.index, 
            y=resampled['min'], 
            name='Min',
            line=dict(color='#9333EA', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=resampled.index, 
            y=resampled['max'], 
            name='Max',
            line=dict(color='#3B82F6', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{target_col} - {get_frequency_label(freq)} View',
            template="plotly_white",
            height=500,
            xaxis_title="Date",
            yaxis_title=target_col,
            showlegend=True
        )
        return fig
        
    else:  # Rolling Statistics
        rolling = ts_df[target_col].rolling(window=window)
        fig = go.Figure()
        
        # Add traces with custom styling
        fig.add_trace(go.Scatter(
            x=ts_df.index, 
            y=ts_df[target_col], 
            name='Raw Data',
            line=dict(color='#6B7280', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=ts_df.index, 
            y=rolling.mean(), 
            name=f'{window}-period Moving Average',
            line=dict(color='#4F46E5', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=ts_df.index, 
            y=rolling.std(), 
            name=f'{window}-period Standard Deviation',
            line=dict(color='#EC4899', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{target_col} - Rolling Statistics (Window: {window})',
            template="plotly_white",
            height=500,
            xaxis_title="Date",
            yaxis_title=target_col,
            showlegend=True
        )
        return fig

def perform_decomposition(ts_df, target_col, period):
    """Perform seasonal decomposition of time series data"""
    return seasonal_decompose(
        ts_df[target_col].fillna(method='ffill'),
        period=period,
        extrapolate_trend='freq'
    )

def create_decomposition_plot(ts_df, target_col, decomposition):
    """Create seasonal decomposition plot with improved styling"""
    fig = make_subplots(
        rows=4, 
        cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.1
    )
    
    # Add components with custom styling
    components = [
        (ts_df[target_col], 'Original', '#4F46E5'),
        (decomposition.trend, 'Trend', '#3B82F6'),
        (decomposition.seasonal, 'Seasonal', '#EC4899'),
        (decomposition.resid, 'Residual', '#6B7280')
    ]
    
    for idx, (data, name, color) in enumerate(components, 1):
        fig.add_trace(
            go.Scatter(
                x=ts_df.index, 
                y=data, 
                name=name,
                line=dict(color=color, width=2)
            ),
            row=idx, 
            col=1
        )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Time Series Decomposition Analysis",
        template="plotly_white"
    )
    
    return fig

def infer_frequency(ts_df):
    """Infer the frequency of the time series data"""
    diff = pd.Series(ts_df.index).diff().mode().iloc[0]
    if diff.days == 1:
        return "Daily"
    elif diff.days == 7:
        return "Weekly"
    elif diff.days >= 28 and diff.days <= 31:
        return "Monthly"
    elif diff.days >= 90 and diff.days <= 92:
        return "Quarterly"
    return "Custom"

def get_frequency_label(freq):
    """Convert frequency code to readable label"""
    freq_map = {
        'D': 'Daily',
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly',
        'Y': 'Yearly'
    }
    return freq_map.get(freq, freq)

def render_time_series_page():
    st.markdown("## ⏳ Time Series Analysis")
    
    if 'data' not in st.session_state:
        st.warning("🚫 Please upload data first!")
        st.info("Go to the Data Upload & Analysis page to load your time series data.")
        return
    
    df = st.session_state['data']
    
    # Date column selection
    date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
    if len(date_columns) == 0:
        st.error("❌ No date columns found in the dataset!")
        return
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            date_col = st.selectbox(
                "🗓️ Select date column",
                date_columns,
                key='ts_date_select'
            )
        with col2:
            st.markdown("")  # Spacing
            st.markdown("")  # Spacing
            if st.checkbox("Show data preview"):
                st.write(df[date_col].head())
    
    # Convert to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"❌ Error converting {date_col} to datetime: {str(e)}")
        return
    
    # Target variable selection
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found for analysis!")
        return
    
    target_col = st.selectbox(
        "📊 Select target variable",
        numeric_cols,
        key='ts_target_select'
    )
    
    # Create time series DataFrame
    ts_df = df[[date_col, target_col]].copy()
    ts_df.set_index(date_col, inplace=True)
    ts_df.sort_index(inplace=True)
    
    # Render basic statistics
    render_time_series_stats(ts_df)
    
    # Visualization options
    st.markdown("### 🎨 Visualization Options")
    
    viz_type = st.radio(
        "Select visualization type",
        ["Raw Data", "Resampled Data", "Rolling Statistics"],
        horizontal=True,
        key='ts_viz_type'
    )
    
    # Show appropriate controls based on visualization type
    if viz_type == "Resampled Data":
        freq = st.select_slider(
            "Select resampling frequency",
            options=['D', 'W', 'M', 'Q', 'Y'],
            value='M',
            format_func=get_frequency_label,
            key='ts_resample_freq'
        )
        fig = create_visualization(ts_df, target_col, viz_type, freq=freq)
        
    elif viz_type == "Rolling Statistics":
        window = st.slider(
            "Select rolling window",
            min_value=2,
            max_value=min(100, len(ts_df)),
            value=7,
            key='ts_rolling_window'
        )
        fig = create_visualization(ts_df, target_col, viz_type, window=window)
        
    else:
        fig = create_visualization(ts_df, target_col, viz_type)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Analysis Section
    with st.expander("🔍 Advanced Analysis", expanded=False):
        if len(ts_df) < 2:
            st.warning("⚠️ Not enough data points for advanced analysis!")
            return
            
        try:
            # Decomposition Period Selection
            st.info("ℹ️ Select the decomposition period based on your data frequency (e.g., 12 for monthly data)")
            period = st.slider(
                "Decomposition Period",
                min_value=2,
                max_value=min(52, len(ts_df)//2),
                value=12,
                help="Period for seasonal decomposition",
                key='decomp_period_slider'
            )
            
            # Perform decomposition
            decomposition = perform_decomposition(ts_df, target_col, period)
            
            # Create and display decomposition plot
            decomp_fig = create_decomposition_plot(ts_df, target_col, decomposition)
            st.plotly_chart(decomp_fig, use_container_width=True)
            
            # Component Analysis in a modern grid
            st.markdown("### 📊 Component Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_data = decomposition.trend.dropna()
                trend_changes = trend_data.diff().dropna()
                trend_direction = "Upward 📈" if trend_changes.mean() > 0 else "Downward 📉"
                
                st.metric(
                    "Trend Direction",
                    trend_direction,
                    f"{abs(trend_changes.mean()):.2f} avg. change"
                )
            
            with col2:
                seasonal_data = decomposition.seasonal.dropna()
                st.metric(
                    "Seasonal Strength",
                    f"{(seasonal_data.std() / ts_df[target_col].std() * 100):.1f}%",
                    f"Range: {seasonal_data.min():.1f} to {seasonal_data.max():.1f}"
                )
            
            # Stationarity Analysis
            st.markdown("### 📊 Stationarity Analysis")
            result = adfuller(ts_df[target_col].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "ADF Statistic",
                    f"{result[0]:.4f}",
                    "Lower is better"
                )
            with col2:
                is_stationary = result[1] < 0.05
                st.metric(
                    "Stationarity",
                    "Stationary ✅" if is_stationary else "Non-stationary ⚠️",
                    f"p-value: {result[1]:.4f}"
                )
            
            if not is_stationary:
                st.info("💡 Consider differencing the series to achieve stationarity")
                
        except Exception as e:
            st.error(f"❌ Error during advanced analysis: {str(e)}")
            st.markdown("""
            **Troubleshooting tips:**
            1. ✓ Check if your data has a clear seasonal pattern
            2. ✓ Try adjusting the decomposition period
            3. ✓ Ensure your data is regularly spaced in time
            4. ✓ Make sure you have enough data points
            """)