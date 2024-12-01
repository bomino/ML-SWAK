import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

def create_correlation_plot(corr_matrix):
    """Create a correlation matrix heatmap with improved styling"""
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_white",
        height=600,
        width=800
    )
    
    return fig

def create_actual_vs_predicted_plot(y_test, y_pred, model_name):
    """Create actual vs predicted scatter plot with improved styling"""
    fig = go.Figure()
    
    # Add scatter plot for predictions
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            color='#4F46E5',
            size=8,
            opacity=0.6
        )
    ))
    
    # Add perfect prediction line
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_test,
        mode='lines',
        name='Perfect Prediction',
        line=dict(
            color='#DC2626',
            dash='dash'
        )
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted Values ({model_name})',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_feature_importance_plot(feature_names, importance_values, model_name):
    """Create feature importance bar plot with improved styling"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=importance_df['feature'],
        x=importance_df['importance'],
        orientation='h',
        marker_color='#4F46E5'
    ))
    
    fig.update_layout(
        title=f'Feature Importance ({model_name})',
        xaxis_title='Importance',
        yaxis_title='Features',
        template='plotly_white',
        height=max(400, len(feature_names) * 25),
        showlegend=False
    )
    
    return fig

def get_available_models():
    """Return dictionary of available models with default parameters"""
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

def render_model_training_page():
    st.markdown("## 🤖 Model Training")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data first!")
        st.info("Go to the Data Upload & Analysis page to load your dataset.")
        return
    
    df = st.session_state['data']
    
    # Data preprocessing section
    st.markdown("### 🔄 Data Preprocessing")
    
    col1, col2 = st.columns(2)
    with col1:
        # Identify column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        st.write("📊 Available Features:")
        st.write(f"- Numeric columns: {len(numeric_cols)}")
        st.write(f"- Categorical columns: {len(categorical_cols)}")
    
    with col2:
        # Encoding preview
        if len(categorical_cols) > 0:
            st.write("🔍 Categorical Columns Preview:")
            for col in categorical_cols[:3]:  # Show first 3 columns
                unique_vals = df[col].nunique()
                st.write(f"- {col}: {unique_vals} unique values")
            if len(categorical_cols) > 3:
                st.write(f"... and {len(categorical_cols) - 3} more")
    
    # Categorical encoding section
    if len(categorical_cols) > 0:
        st.markdown("#### 📝 Categorical Encoding")
        selected_categorical = st.multiselect(
            "Select categorical columns to encode",
            options=list(categorical_cols),
            default=list(categorical_cols)[:2],
            help="Choose categorical columns to convert to numeric features"
        )
        
        # Create dummy variables
        df_encoded = df.copy()
        if selected_categorical:
            df_encoded = pd.get_dummies(df_encoded, columns=selected_categorical, drop_first=True)
            st.success(f"✅ Encoded {len(selected_categorical)} categorical columns")
    else:
        df_encoded = df.copy()
    
    # Feature Selection
    st.markdown("### 🎯 Feature Selection")
    
    available_features = df_encoded.columns
    if len(available_features) < 2:
        st.error("❌ Not enough features available for modeling (minimum 2 required)!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        target = st.selectbox(
            "Select target variable",
            available_features,
            help="Choose the variable you want to predict"
        )
    
    # Create list of features excluding target
    potential_features = [col for col in available_features if col != target]
    default_n_features = min(5, len(potential_features))
    
    with col2:
        features = st.multiselect(
            "Select features",
            options=potential_features,
            default=potential_features[:default_n_features],
            help="Choose the features to use for prediction"
        )
    
    if not features:
        st.warning("⚠️ Please select at least one feature!")
        return
    
    # Prepare data
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Show correlation matrix
    st.markdown("### 📊 Feature Correlations")
    corr_matrix = X.corr(numeric_only=True)
    fig = create_correlation_plot(corr_matrix)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature selection based on correlation
    st.markdown("#### 🔄 Feature Selection Refinement")
    correlation_threshold = st.slider(
        "Remove highly correlated features above threshold",
        0.0, 1.0, 0.9,
        help="Features with correlation above this threshold will be removed"
    )
    
    # Remove highly correlated features
    features_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                colname = corr_matrix.columns[i]
                features_to_drop.add(colname)
    
    X = X.drop(columns=list(features_to_drop))
    if len(features_to_drop) > 0:
        st.info(f"ℹ️ Removed {len(features_to_drop)} highly correlated features: {', '.join(features_to_drop)}")
    
    # Model Training Section
    st.markdown("### 🚀 Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test set size",
            0.1, 0.4, 0.2,
            help="Proportion of data to use for testing"
        )
    
    with col2:
        # Model selection
        available_models = get_available_models()
        selected_models = st.multiselect(
            "Select models to train",
            options=list(available_models.keys()),
            default=["Ridge Regression"],
            help="Choose one or more models to train"
        )
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model!")
        return
    
    # Training button
    if st.button("🎯 Train Models", use_container_width=True):
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models and collect results
            results = {}
            with st.spinner('🔄 Training models...'):
                for model_name in selected_models:
                    # Create progress message
                    st.text(f"Training {model_name}...")
                    
                    # Train model
                    model = available_models[model_name]
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[model_name] = {
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2,
                        "model": model,
                        "predictions": y_pred
                    }
                    
                    # Save model and scaler
                    st.session_state[f'model_{model_name}'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = features
            
            # Display results
            st.markdown("### 📊 Model Comparison")
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                model_name: {
                    "RMSE": results[model_name]["RMSE"],
                    "MAE": results[model_name]["MAE"],
                    "R²": results[model_name]["R2"]
                }
                for model_name in results
            }).round(4)
            
            # Display metrics
            st.dataframe(metrics_df, use_container_width=True)
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda x: results[x]["R2"])
            best_model = results[best_model_name]["model"]
            best_predictions = results[best_model_name]["predictions"]
            
            # Show actual vs predicted plot
            st.markdown("### 📈 Model Performance")
            fig = create_actual_vs_predicted_plot(y_test, best_predictions, best_model_name)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance for tree-based models
            if best_model_name in ["Random Forest", "Gradient Boosting"]:
                st.markdown("### 🎯 Feature Importance")
                fig = create_feature_importance_plot(
                    X.columns,
                    best_model.feature_importances_,
                    best_model_name
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.success('✅ Models trained successfully!')
            
        except Exception as e:
            st.error(f"❌ An error occurred during model training: {str(e)}")
            st.info("💡 Try adjusting the feature selection or preprocessing parameters.")