import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def render_introduction():
    """Render the introduction section"""
    st.markdown("""
    This tutorial will guide you through the process of using the Model Training page effectively.
    Use the sidebar to navigate through different sections of the tutorial.
    
    ### What You'll Learn
    - How to understand and preprocess your data
    - How to select the right features for your model
    - How to choose and train appropriate models
    - Best practices and troubleshooting tips
    """)
    
    # Show sample dataset info if available
    if 'data' in st.session_state:
        st.subheader("Your Current Dataset Overview")
        df = st.session_state['data']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
        st.write("Columns:", list(df.columns))
    else:
        st.info("💡 Upload your data in the Data Upload page to see a personalized overview.")

def render_data_understanding():
    """Render the data understanding section"""
    st.header("Understanding Your Data")
    
    st.markdown("""
    ### Types of Data Columns
    
    📊 **Numeric Columns**
    - Contain numerical values (e.g., 'Spend')
    - Can be used directly in models
    - Example: Transaction amounts, counts, measurements
    
    📝 **Categorical Columns**
    - Contain text or categories (e.g., 'CategoryLevel1', 'TransactionGroup')
    - Need to be encoded before using in models
    - Example: Categories, status, groups
    """)
    
    # Display actual data types if data is available
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Numeric Columns")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                st.write(f"- {col}")
                    
        with col2:
            st.subheader("Categorical Columns")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                st.write(f"- {col}")
                    
        # Show sample of unique values
        st.subheader("Sample Categories")
        selected_col = st.selectbox("Select a categorical column to see unique values:", categorical_cols)
        st.write(f"Unique values in {selected_col}:", df[selected_col].nunique())
        st.write("Sample values:", list(df[selected_col].unique()[:5]))
            
    else:
        st.info("💡 Upload your data to see specific information about your columns.")


def render_preprocessing():
    """Render the data preprocessing section"""
    st.header("Data Preprocessing")
    
    st.markdown("""
    ### Categorical Column Encoding
    
    #### What to Encode
    
    ✅ **Good Candidates for Encoding:**
    - Columns with meaningful categories (e.g., 'CategoryLevel1')
    - Columns with reasonable number of unique values
    - Columns that could influence your target variable
    
    ❌ **Think Twice Before Encoding:**
    - Columns with too many unique values (e.g., IDs, names)
    - Columns with single value (no variation)
    - Free text columns
    
    #### Encoding Process
    1. Select categorical columns to encode
    2. System creates binary columns (one-hot encoding)
    3. Original categorical column is replaced with binary columns
    """)
    
    # Create a sample dataframe for demonstration
    sample_df = pd.DataFrame({
        'Category': ['Facilities', 'Marketing', 'HR', 'Facilities'],
        'Spend': [1000, 2000, 1500, 1200]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Encoding")
        st.write(sample_df)
        
    with col2:
        st.subheader("After Encoding")
        st.write(pd.get_dummies(sample_df, columns=['Category']))
        
    st.markdown("""
    ### 💡 Tips for Preprocessing
    1. Start with fewer categories to test the model
    2. Monitor the number of features created
    3. Consider combining rare categories
    """)

def render_feature_selection():
    """Render the feature selection section"""
    st.header("Feature Selection")
    
    st.markdown("""
    ### Selecting Target Variable
    
    Your target variable is what you want to predict. Choose based on your business goal:
    
    #### For Spend Analysis
    - Target: 'Spend'
    - Goal: Predict spending based on categories
    
    #### For Category Prediction
    - Target: 'CategoryLevel1' or 'Final Category'
    - Goal: Automatically categorize transactions
    
    ### Selecting Features
    
    #### Best Practices for Feature Selection:
    1. **Start Small**: Begin with 3-5 most relevant features
    2. **Use Domain Knowledge**: Choose features that logically influence your target
    3. **Check Correlations**: Use the correlation matrix to identify:
       - Strongly correlated features (potential redundancy)
       - Features with strong correlation to target
    """)
    
    # Interactive feature selection example
    st.subheader("Interactive Feature Selection Example")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Create correlation matrix
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            st.write("Correlation Matrix for Numeric Columns:")
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            #### How to Read the Correlation Matrix:
            - Values close to 1: Strong positive correlation
            - Values close to -1: Strong negative correlation
            - Values close to 0: Little to no correlation
            """)
            
            # Show high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'features': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                st.markdown("#### 🔍 High Correlations Detected")
                for corr in high_corr:
                    st.write(f"- **{corr['features']}**: {corr['correlation']:.2f}")
        else:
            st.info("Need more numeric columns to show correlations.")
    else:
        st.info("💡 Upload your data to see correlation analysis.")

    # Feature Selection Tips
    st.markdown("""
    ### Additional Feature Selection Tips
    
    #### 1. Consider Feature Importance
    - Use tree-based models to get feature importance
    - Focus on features with higher importance scores
    - Remove features with very low importance
    
    #### 2. Handle Multicollinearity
    - Remove highly correlated features
    - Keep features more relevant to your target
    - Use dimensionality reduction if needed
    
    #### 3. Validate Feature Selection
    - Test model performance with different feature sets
    - Use cross-validation to ensure stability
    - Monitor for overfitting
    """)
    
    # Common Pitfalls
    st.markdown("""
    ### 🚫 Common Pitfalls to Avoid
    
    1. **Using Too Many Features**
    - Can lead to overfitting
    - Increases model complexity
    - Slows down training
    
    2. **Ignoring Domain Knowledge**
    - Missing important predictive features
    - Including irrelevant features
    - Wrong feature combinations
    
    3. **Not Handling Missing Values**
    - Can affect model performance
    - Might introduce bias
    - Reduces data quality
    """)


def render_model_selection():
    """Render the model selection section"""
    st.header("Model Selection Guide")
    
    # Traditional Models Section
    st.markdown("""
    ### Types of Models
    
    #### 1. Traditional Machine Learning Models
    
    👉 **Ridge Regression**
    - Good baseline model
    - Works well with encoded categorical variables
    - Best for linear relationships
    
    👉 **Lasso Regression**
    - Similar to Ridge, but can eliminate irrelevant features
    - Good for feature selection
    
    👉 **Random Forest**
    - Handles non-linear relationships
    - Can capture complex patterns
    - Provides feature importance
    
    👉 **Gradient Boosting**
    - Often best performance
    - Can overfit if not tuned properly
    - Good for both regression and classification
    
    #### 2. Advanced Models
    
    👉 **XGBoost**
    - Very fast and efficient
    - Handles missing values well
    - Great for large datasets
    - Built-in regularization
    
    👉 **LightGBM**
    - Faster than XGBoost
    - Lower memory usage
    - Great for large datasets
    - Good with categorical features
    """)
    
    # Model Selection Helper
    st.markdown("### 🤖 Model Selection Helper")
    col1, col2 = st.columns(2)
    
    with col1:
        data_size = st.radio(
            "How much data do you have?", 
            ["Small (< 1000 rows)", 
             "Medium (1000-10000 rows)", 
             "Large (> 10000 rows)"]
        )
        
        feature_type = st.radio(
            "What type of features do you have?",
            ["Mostly numeric",
             "Mostly categorical",
             "Mixed"]
        )
    
    with col2:
        priority = st.radio(
            "What's your main priority?",
            ["Accuracy",
             "Training speed",
             "Model interpretability"]
        )
    
    # Show recommendation based on selections
    st.subheader("Recommended Model")
    if data_size == "Small (< 1000 rows)":
        if priority == "Accuracy":
            st.success("➡️ Try Random Forest or XGBoost")
            st.info("💡 Good for small datasets with complex patterns")
        elif priority == "Training speed":
            st.success("➡️ Try Ridge Regression")
            st.info("💡 Fast and efficient for small datasets")
        else:  # interpretability
            st.success("➡️ Try Lasso Regression")
            st.info("💡 Provides feature selection and interpretability")
    elif data_size == "Large (> 10000 rows)":
        if priority == "Accuracy":
            st.success("➡️ Try LightGBM or XGBoost")
            st.info("💡 Highly efficient for large datasets")
        elif priority == "Training speed":
            st.success("➡️ Try LightGBM")
            st.info("💡 Faster training on large datasets")
        else:  # interpretability
            st.success("➡️ Try Random Forest")
            st.info("💡 Good balance of performance and interpretability")

def render_forecasting_models():
    """Render the forecasting models section"""
    st.header("Forecasting Models")
    
    st.markdown("""
    ### Specialized Forecasting Models
    
    #### 1. Prophet (by Facebook)
    👉 **Best for:**
    - Daily/weekly data with seasonal patterns
    - Data with missing values
    - Data with outliers
    - Multiple seasonality patterns
    
    👉 **Features:**
    - Handles holidays automatically
    - Detects seasonality
    - Provides uncertainty intervals
    - Very robust to missing data
    
    #### 2. SARIMA
    👉 **Best for:**
    - Clear seasonal patterns
    - Regular time intervals
    - Stationary data (or easily made stationary)
    
    👉 **Features:**
    - Explicit modeling of trends
    - Handles seasonal components
    - Good for short-term forecasts
    
    ### Choosing the Right Forecasting Model
    """)
    
    # Interactive Model Selector
    col1, col2 = st.columns(2)
    with col1:
        data_frequency = st.selectbox(
            "What's your data frequency?",
            ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        )
        
        seasonality = st.multiselect(
            "What types of seasonality do you have?",
            ["Daily", "Weekly", "Monthly", "Yearly", "None"],
            default=["None"]
        )
    
    with col2:
        missing_values = st.radio(
            "Do you have missing values?",
            ["Yes", "No"]
        )
        
        external_features = st.radio(
            "Do you have external features (like weather, promotions)?",
            ["Yes", "No"]
        )
    
    # Show model recommendation
    st.subheader("Recommended Forecasting Model")
    
    if len(seasonality) > 1 or missing_values == "Yes":
        st.success("➡️ Prophet is recommended because:")
        if len(seasonality) > 1:
            st.info("- Can handle multiple seasonality patterns")
        if missing_values == "Yes":
            st.info("- Robust to missing values")
    
    elif external_features == "Yes":
        st.success("➡️ XGBoost/LightGBM is recommended because:")
        st.info("- Can incorporate external features")
        st.info("- Handles complex patterns well")
    
    elif "None" not in seasonality and data_frequency in ["Monthly", "Quarterly"]:
        st.success("➡️ SARIMA is recommended because:")
        st.info("- Good for regular seasonal patterns")
        st.info("- Works well with monthly/quarterly data")
    
    # Model Configuration Examples
    st.markdown("### 📝 Configuration Examples")
    
    model_example = st.selectbox(
        "Select model to see example configuration",
        ["Prophet", "SARIMA", "XGBoost"]
    )
    
    if model_example == "Prophet":
        st.code("""
# Prophet Configuration Example
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
        """)
    elif model_example == "SARIMA":
        st.code("""
# SARIMA Configuration Example
model = SARIMAX(
    order=(1, 1, 1),           # (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s)
)
        """)
    elif model_example == "XGBoost":
        st.code("""
# XGBoost Configuration Example
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
        """)



def render_best_practices():
    """Render the best practices section"""
    st.header("Best Practices")
    
    # Data Preparation
    st.markdown("""
    ### 1. Data Preparation
    - Clean your data before training
    - Handle missing values appropriately
    - Remove or fix outliers if necessary
    
    ### 2. Feature Engineering
    - Start with basic features
    - Add engineered features gradually
    - Document feature importance
    
    ### 3. Model Training Process
    - Start simple, add complexity as needed
    - Use cross-validation for robust results
    - Monitor for overfitting
    
    ### 4. Model Evaluation
    - Compare multiple models
    - Use appropriate metrics
    - Consider business context
    
    ### 5. Iteration Tips
    1. Make one change at a time
    2. Document changes and results
    3. Save best performing models
    """)
    
    # Add interactive examples if data is available
    if 'data' in st.session_state:
        st.subheader("Your Data Best Practices")
        df = st.session_state['data']
        
        # Show some basic statistics
        with st.expander("📊 Dataset Statistics", expanded=True):
            st.write(df.describe())
        
        # Show missing values if any
        missing = df.isnull().sum()
        if missing.sum() > 0:
            with st.expander("🔍 Missing Values Analysis", expanded=True):
                for column, count in missing[missing > 0].items():
                    st.metric(
                        column,
                        f"{count} missing",
                        f"{(count/len(df)*100):.1f}% of data"
                    )

def render_troubleshooting():
    """Render the troubleshooting section"""
    st.header("Troubleshooting Common Issues")
    
    # Common Problems and Solutions Section
    st.markdown("""
    ### Common Problems and Solutions
    """)
    
    # Using tabs for different issues
    tab1, tab2, tab3, tab4 = st.tabs([
        "Not Enough Features",
        "Poor Model Performance",
        "Overfitting",
        "Long Training Time"
    ])
    
    with tab1:
        st.markdown("""
        #### Not Enough Features Error
        **Problem**: Model training fails due to insufficient features
        
        **Solutions**:
        1. ✅ Encode more categorical columns
        2. ✅ Add relevant features
        3. ✅ Check for data loading issues
        """)
    
    with tab2:
        st.markdown("""
        #### Poor Model Performance
        **Problem**: Model predictions are not accurate enough
        
        **Solutions**:
        1. ✅ Add more relevant features
        2. ✅ Try different model types
        3. ✅ Check for data quality issues
        4. ✅ Consider feature engineering
        """)
    
    with tab3:
        st.markdown("""
        #### Overfitting
        **Problem**: Model performs well on training data but poorly on test data
        
        **Solutions**:
        1. ✅ Reduce number of features
        2. ✅ Use simpler model
        3. ✅ Increase training data
        4. ✅ Add regularization
        """)
    
    with tab4:
        st.markdown("""
        #### Long Training Time
        **Problem**: Model takes too long to train
        
        **Solutions**:
        1. ✅ Reduce number of features
        2. ✅ Use smaller data sample
        3. ✅ Choose simpler model
        4. ✅ Optimize feature selection
        """)
    
    # Interactive Troubleshooting Tool
    st.markdown("### 🔧 Interactive Troubleshooting")
    
    issue = st.selectbox(
        "What issue are you experiencing?",
        ["Select an issue...",
         "Not enough features",
         "Poor model performance",
         "Overfitting",
         "Long training time"]
    )
    
    if issue == "Not enough features":
        st.success("### Steps to Resolve:")
        st.info("1. Check your categorical column selection")
        st.info("2. Verify encoding process")
        st.info("3. Consider adding more features")
        
    elif issue == "Poor model performance":
        st.success("### Steps to Resolve:")
        st.info("1. Review feature selection")
        st.info("2. Try different models")
        st.info("3. Check data quality")
        
    elif issue == "Overfitting":
        st.success("### Steps to Resolve:")
        st.info("1. Reduce feature count")
        st.info("2. Increase regularization")
        st.info("3. Get more training data")
        
    elif issue == "Long training time":
        st.success("### Steps to Resolve:")
        st.info("1. Reduce feature count")
        st.info("2. Use data sample")
        st.info("3. Choose simpler model")
    
    # Additional Resources
    st.markdown("""
    ### 📚 Additional Resources
    
    #### Documentation
    - Model documentation
    - API references
    - Best practices guides
    
    #### Support Channels
    - Technical support
    - Community forums
    - Feature requests
    
    #### Learning Resources
    - Tutorials
    - Example notebooks
    - Video guides
    """)
    
    # Contact Support
    with st.expander("📞 Contact Support", expanded=False):
        st.markdown("""
        Need additional help? Contact our support team:
        
        - Email: support@example.com
        - Discord: [Join our community]
        - GitHub: [Report issues]
        """)


def render_metrics_guide():
    """Render the metrics understanding section"""
    st.markdown("""
    ### 📊 Understanding Model Metrics
    
    #### Regression Metrics
    
    **RMSE (Root Mean Square Error)**
    - Average prediction error
    - Same units as target variable
    - Lower is better
    - Penalizes large errors more
    
    **MAE (Mean Absolute Error)**
    - Average absolute prediction error
    - Same units as target variable
    - More robust to outliers
    - Lower is better
    
    **R² Score**
    - Proportion of variance explained
    - Range: 0 to 1
    - Higher is better
    - Independent of scale
    
    #### Classification Metrics
    
    **Accuracy**
    - Proportion of correct predictions
    - Range: 0 to 1
    - Higher is better
    - May be misleading for imbalanced data
    
    **Precision & Recall**
    - Precision: Accuracy of positive predictions
    - Recall: Proportion of actual positives identified
    - Consider trade-off between them
    """)

def render_helper_tools():
    """Render helper tools section"""
    st.markdown("""
    ### 🛠️ Helper Tools
    
    #### Data Preparation
    - Data cleaning utilities
    - Missing value handlers
    - Outlier detection
    
    #### Feature Engineering
    - Automated feature creation
    - Encoding utilities
    - Scaling functions
    
    #### Model Training
    - Cross-validation tools
    - Hyperparameter tuning
    - Model comparison
    """)

def format_metric(value, format_type="number"):
    """Format metric values for display"""
    if format_type == "number":
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"
    elif format_type == "percentage":
        return f"{value:.1f}%"
    return str(value)

def render_tutorial_page():
    """Main function to render the tutorial page"""
    st.title("📚 Model Training Tutorial")
    
    # Section definitions with icons and descriptions
    sections = {
        "Introduction": {
            "icon": "📝",
            "render": render_introduction,
            "description": "Get started with the basics"
        },
        "Understanding Your Data": {
            "icon": "📊",
            "render": render_data_understanding,
            "description": "Learn about different data types"
        },
        "Data Preprocessing": {
            "icon": "🔄",
            "render": render_preprocessing,
            "description": "Prepare your data for modeling"
        },
        "Feature Selection": {
            "icon": "🎯",
            "render": render_feature_selection,
            "description": "Choose the right features"
        },
        "Model Selection": {
            "icon": "🤖",
            "render": render_model_selection,
            "description": "Pick the best model for your task"
        },
        "Forecasting Models": {
            "icon": "📈",
            "render": render_forecasting_models,
            "description": "Time series specific models"
        },
        "Best Practices": {
            "icon": "✨",
            "render": render_best_practices,
            "description": "Tips and tricks for success"
        },
        "Troubleshooting": {
            "icon": "🔧",
            "render": render_troubleshooting,
            "description": "Solve common issues"
        },
        "Metrics Guide": {
            "icon": "📊",
            "render": render_metrics_guide,
            "description": "Understanding model evaluation"
        },
        "Helper Tools": {
            "icon": "🛠️",
            "render": render_helper_tools,
            "description": "Useful utilities and functions"
        }
    }
    
    # Section selection
    col1, col2 = st.columns([2, 1])
    with col1:
        current_section = st.selectbox(
            "Select Tutorial Section",
            options=list(sections.keys()),
            format_func=lambda x: f"{sections[x]['icon']} {x}"
        )
    
    with col2:
        st.info(sections[current_section]['description'])
    
    # Visual separator
    st.markdown("---")
    
    # Render selected section
    sections[current_section]['render']()
    
    # Navigation arrows
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    section_list = list(sections.keys())
    current_index = section_list.index(current_section)
    
    with col1:
        if current_index > 0:
            prev_section = section_list[current_index - 1]
            if st.button(f"← {sections[prev_section]['icon']} Previous"):
                st.session_state['current_tutorial_section'] = prev_section
                st.rerun()
    
    with col3:
        if current_index < len(section_list) - 1:
            next_section = section_list[current_index + 1]
            if st.button(f"Next {sections[next_section]['icon']} →"):
                st.session_state['current_tutorial_section'] = next_section
                st.rerun()
    
    # Help section in sidebar
    with st.sidebar:
        with st.expander("📞 Need Help?", expanded=False):
            st.info("""
            If you need assistance:
            
            - Check our documentation
            - Join our community
            - Contact support
            
            Email: Bomino@mlawali.com
            """)

# Initialize session state if needed
if 'current_tutorial_section' not in st.session_state:
    st.session_state['current_tutorial_section'] = "Introduction"
