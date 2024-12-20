import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from utils.data_analysis_helpers import (
    create_download_buttons,
    create_analysis_card,
    display_metrics,
    display_distribution_analysis,
    display_correlation_analysis,
    analyze_numerical_data,
    analyze_categorical_data,
    perform_pareto_analysis,
    analyze_temporal_data
)

def render_data_analysis_page():
    """Render the data analysis page"""
    st.markdown("## 📊 Data Upload & Analysis")
    
    # Check if data already exists in session state
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.success("✅ Data already loaded!")
        
        # Option to upload new data using button and session state
        if st.button("Upload Different Data"):
            del st.session_state['data']
            st.rerun()  # Using st.rerun() instead of experimental_rerun
    else:
        st.markdown("Upload your data and get instant insights")
        # File uploader
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to upload",
            type="csv",
            help="Upload a CSV file to begin analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load and store data
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df
                st.success("✅ Data uploaded successfully!")
                st.rerun()  # Rerun to show analysis options
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.write("Please check your data format and try again.")
                return
        else:
            # Show sample data format
            st.info("👆 Please upload a CSV file to begin analysis.")
            st.markdown("### 📋 Sample Data Format")
            sample_df = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02'],
                'value': [100, 150],
                'category': ['A', 'B']
            })
            st.dataframe(
                sample_df,
                use_container_width=True,
                column_config={
                    "date": "Date",
                    "value": "Numeric Value",
                    "category": "Category"
                }
            )
            return
            
    # Only proceed with analysis if we have data
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Display key metrics for existing data
        display_metrics(df)
        
        # Data Preview Section
        with st.expander("👀 Data Preview", expanded=True):
            if st.checkbox("Show full data"):
                st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df.head(), use_container_width=True)
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Create tabs for different analyses
        tabs = st.tabs([
            "📊 Basic Analysis",
            "📈 Advanced Analysis",
            "🔍 Correlation Analysis"
        ])
        
        # Basic Analysis Tab
        with tabs[0]:
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Numerical Analysis", "Categorical Analysis", "Temporal Analysis"],
                horizontal=True,
                key="main_analysis_type"
            )
            
            if analysis_type == "Numerical Analysis" and numeric_cols:
                selected_num_col = st.selectbox(
                    "Select numerical column",
                    numeric_cols,
                    key="num_analysis_col"
                )
                analyze_numerical_data(df, selected_num_col)
                
            elif analysis_type == "Categorical Analysis" and categorical_cols:
                selected_cat_col = st.selectbox(
                    "Select categorical column",
                    categorical_cols,
                    key="cat_analysis_col"
                )
                
                cat_analysis_type = st.radio(
                    "Select Analysis Type",
                    ["Basic Analysis", "Pareto Analysis"],
                    horizontal=True,
                    key="cat_analysis_type"
                )
                
                if cat_analysis_type == "Basic Analysis":
                    analyze_categorical_data(df, selected_cat_col)
                else:  # Pareto Analysis
                    analysis_method = st.radio(
                        "Select Analysis Method",
                        ["Count", "Sum by Value"],
                        horizontal=True,
                        key="pareto_analysis_method"
                    )
                    
                    pareto_df = perform_pareto_analysis(df, selected_cat_col, analysis_method)
                    
                    # Create download options for Pareto analysis
                    if pareto_df is not None:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            pareto_df.to_excel(writer, sheet_name='Pareto Analysis', index=False)
                        
                        create_download_buttons(
                            csv_data=pareto_df.to_csv(index=False),
                            excel_data=excel_buffer.getvalue(),
                            base_filename='pareto_analysis',
                            selected_cat_col=selected_cat_col
                        )
                        
            elif analysis_type == "Temporal Analysis":
                # Try to identify potential date columns
                potential_date_cols = []
                
                # Check existing datetime columns
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                potential_date_cols.extend(datetime_cols)
                
                # Check object/string columns that might be dates
                for col in df.select_dtypes(include=['object']).columns:
                    # Sample the first non-null value
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample and isinstance(sample, str):
                        try:
                            pd.to_datetime(sample)
                            potential_date_cols.append(col)
                        except:
                            continue
                
                if potential_date_cols:
                    selected_date_col = st.selectbox(
                        "Select datetime column",
                        potential_date_cols,
                        key="temporal_analysis_col"
                    )
                    
                    try:
                        # Convert to datetime
                        df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Date Range",
                                f"{df[selected_date_col].min().strftime('%Y-%m-%d')} to {df[selected_date_col].max().strftime('%Y-%m-%d')}"
                            )
                            st.metric(
                                "Time Span",
                                f"{(df[selected_date_col].max() - df[selected_date_col].min()).days} days"
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
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No numeric columns available for time series plotting")
                                
                    except Exception as e:
                        st.error(f"Error processing date column: {str(e)}")
                        st.info("Try selecting a different column or check the date format")
                else:
                    st.warning("No date/time columns found. Please ensure your dataset includes a column with dates.")
                    st.info("Date columns should contain values like 'YYYY-MM-DD' or other standard date formats.")
        
        # Advanced Analysis Tab
        with tabs[1]:
            if numeric_cols:
                st.markdown("### 📊 Statistical Summary")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                st.markdown("### 📈 Distribution Analysis")
                display_distribution_analysis(df, numeric_cols, categorical_cols)
            else:
                st.warning("No numerical columns found for statistical analysis")
        
        # Correlation Analysis Tab
        with tabs[2]:
            if len(numeric_cols) > 1:
                display_correlation_analysis(df, numeric_cols)
            else:
                st.warning("Need at least 2 numerical columns for correlation analysis")
