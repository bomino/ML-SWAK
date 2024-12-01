import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

def create_download_buttons(csv_data, excel_data, base_filename, selected_cat_col):
    """Create styled download buttons"""
    st.markdown("""
        <div class="download-container">
            <h4>📥 Download Analysis Results</h4>
            <div class="button-grid">
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f'{base_filename}_{selected_cat_col}.csv',
            mime='text/csv',
            use_container_width=True,
        )
    
    with col2:
        st.download_button(
            label="📈 Download Excel",
            data=excel_data,
            file_name=f'{base_filename}_{selected_cat_col}.xlsx',
            mime='application/vnd.ms-excel',
            use_container_width=True,
        )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def create_analysis_card(title, content, icon="📊"):
    """Create a styled analysis section"""
    st.markdown(f"""
        <div class="analysis-card">
            <h3>
                {icon} {title}
            </h3>
        </div>
    """, unsafe_allow_html=True)
    content()

def display_metrics(df):
    """Display key metrics about the dataset"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Records",
            f"{df.shape[0]:,}",
            f"{df.shape[1]} columns"
        )
    
    with col2:
        missing_pct = (df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100)
        st.metric(
            "Data Completeness",
            f"{100-missing_pct:.1f}%",
            f"{missing_pct:.1f}% missing"
        )
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum()/1024/1024
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f} MB",
            "compressed"
        )

def display_distribution_analysis(df, numeric_cols, categorical_cols):
    """Display distribution analysis with improved styling"""
    if numeric_cols:
        st.markdown("#### 📊 Numerical Distributions")
        selected_num_col = st.selectbox(
            "Select numerical column to analyze", 
            numeric_cols,
            key="dist_num_select"
        )
        
        fig = px.histogram(
            df, 
            x=selected_num_col, 
            title=f'Distribution of {selected_num_col}',
            template="plotly_white",
            marginal="box"
        )
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            title_font_size=16
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if categorical_cols:
        st.markdown("#### 📊 Categorical Distributions")
        selected_cat_col = st.selectbox(
            "Select categorical column to analyze", 
            categorical_cols,
            key="dist_cat_select"
        )
        
        value_counts = df[selected_cat_col].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution of {selected_cat_col}',
            template="plotly_white"
        )
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            title_font_size=16,
            xaxis_title=selected_cat_col,
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_correlation_analysis(df, numeric_cols):
    """Display correlation analysis with improved styling"""
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="Correlation"),
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(
            title=dict(
                text="Correlation Matrix",
                x=0.5,
                font_size=16
            ),
            width=800,
            height=800,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        high_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr.append({
                        'features': f"{correlation_matrix.columns[i]} vs {correlation_matrix.columns[j]}",
                        'correlation': corr
                    })
        
        if high_corr:
            st.markdown("#### 🔍 High Correlations Detected")
            for corr in high_corr:
                st.markdown(
                    f"- **{corr['features']}**: {corr['correlation']:.2f}"
                )
def analyze_numerical_data(df, selected_num_col):
    """Analyze numerical column"""
    col1, col2 = st.columns(2)
    
    with col1:
        stats = df[selected_num_col].describe()
        st.write("Basic Statistics:")
        st.write(stats)
    
    with col2:
        fig = px.histogram(
            df, 
            x=selected_num_col, 
            title=f'Distribution of {selected_num_col}',
            marginal="box",
            template="plotly_white"
        )
        st.plotly_chart(fig)
    
    # Outlier analysis
    Q1 = df[selected_num_col].quantile(0.25)
    Q3 = df[selected_num_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[selected_num_col] < (Q1 - 1.5 * IQR)) | 
                (df[selected_num_col] > (Q3 + 1.5 * IQR))][selected_num_col]
    
    if len(outliers) > 0:
        st.info(f"📊 Outliers detected: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}% of data)")

def analyze_categorical_data(df, selected_cat_col):
    """Analyze categorical column"""
    # Basic Analysis section
    col1, col2 = st.columns(2)
    
    with col1:
        unique_vals = df[selected_cat_col].nunique()
        mode_val = df[selected_cat_col].mode()[0]
        null_count = df[selected_cat_col].isnull().sum()
        
        st.write("Category Statistics:")
        st.write(f"- Unique values: {unique_vals}")
        st.write(f"- Most common: {mode_val}")
        st.write(f"- Missing values: {null_count}")
    
    with col2:
        value_counts = df[selected_cat_col].value_counts().head(10)
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            title=f'Top 10 Categories in {selected_cat_col}',
            template="plotly_white"
        )
        st.plotly_chart(fig)
    
    st.write("Detailed Category Breakdown:")
    value_counts_df = pd.DataFrame({
        'Category': df[selected_cat_col].value_counts().index,
        'Count': df[selected_cat_col].value_counts().values,
        'Percentage': (df[selected_cat_col].value_counts(normalize=True) * 100).round(2)
    })
    st.dataframe(value_counts_df)

def perform_pareto_analysis(df, selected_cat_col, analysis_method):
    """Perform Pareto analysis on categorical data"""
    st.markdown("### 📈 Pareto Analysis")
    st.info("The Pareto principle states that roughly 80% of effects come from 20% of causes.")
    
    if analysis_method == "Count":
        # Calculate value counts
        value_counts = df[selected_cat_col].value_counts()
        total = value_counts.sum()
        
        pareto_df = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': (value_counts.values / total * 100).round(2)
        })
        
        y_axis_title = 'Count'
        
    else:  # Sum by Value
        # Let user select numerical column for sum
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        value_col = st.selectbox(
            "Select numerical column for sum",
            num_cols,
            key="pareto_value_column"
        )
        
        # Calculate sums by category
        value_sums = df.groupby(selected_cat_col)[value_col].sum().sort_values(ascending=False)
        total = value_sums.sum()
        
        pareto_df = pd.DataFrame({
            'Category': value_sums.index,
            'Sum': value_sums.values,
            'Percentage': (value_sums.values / total * 100).round(2)
        })
        
        y_axis_title = f'Sum of {value_col}'
    
    # Calculate cumulative percentage
    pareto_df['Cumulative_Percentage'] = pareto_df['Percentage'].cumsum()
    
    # Create Pareto chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=pareto_df['Category'],
        y=pareto_df['Count' if analysis_method == "Count" else 'Sum'],
        name=y_axis_title,
        marker_color='#4F46E5'  # Modern indigo color
    ))
    
    # Add cumulative line
    fig.add_trace(go.Scatter(
        x=pareto_df['Category'],
        y=pareto_df['Cumulative_Percentage'],
        name='Cumulative %',
        marker_color='#DC2626',  # Modern red color
        yaxis='y2'
    ))
    
    # Update layout with modern styling
    fig.update_layout(
        title=f'Pareto Chart - {selected_cat_col}',
        yaxis=dict(title=y_axis_title),
        yaxis2=dict(
            title='Cumulative %',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        showlegend=True,
        template="plotly_white",
        height=600
    )
    
    # Add 80% reference line with modern styling
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#059669",  # Modern green color
        annotation_text="80% Reference Line",
        yref='y2'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display key insights
    categories_80 = len(pareto_df[pareto_df['Cumulative_Percentage'] <= 80])
    total_categories = len(pareto_df)
    
    # Display insights in a modern card
    st.markdown("""
        <div style="
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h4 style="margin-bottom: 1rem;">🔍 Key Insights</h4>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Categories",
            f"{total_categories}",
            "unique categories"
        )
    with col2:
        st.metric(
            "80% Coverage",
            f"{categories_80}",
            f"{(categories_80/total_categories*100):.1f}% of categories"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display detailed Pareto table
    st.markdown("### 📊 Detailed Pareto Analysis")
    st.dataframe(pareto_df, use_container_width=True)
    
    return pareto_df

def analyze_temporal_data(df, selected_date_col, numeric_cols):
    """Analyze temporal data"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Date Range",
            f"{df[selected_date_col].min().strftime('%Y-%m-%d')}",
            f"to {df[selected_date_col].max().strftime('%Y-%m-%d')}"
        )
        span_days = (df[selected_date_col].max() - df[selected_date_col].min()).days
        st.metric(
            "Time Span",
            f"{span_days} days",
            f"{span_days/30:.1f} months"
        )
    
    with col2:
        if numeric_cols:
            selected_value_col = st.selectbox(
                "Select value column for time series",
                numeric_cols,
                key="temporal_value_col"
            )
            
            fig = px.line(
                df, 
                x=selected_date_col, 
                y=selected_value_col,
                title=f'{selected_value_col} over Time',
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title=selected_value_col,
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns available for time series analysis")                    