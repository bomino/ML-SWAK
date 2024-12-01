import streamlit as st
import pandas as pd
import base64
import io

def render_single_prediction_form(features):
    """Render form for single prediction input"""
    st.markdown("### 📝 Enter Feature Values")
    
    # Create input fields for each feature
    input_data = {}
    
    # Create columns for input fields (2 columns)
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        col_idx = idx % 2
        with cols[col_idx]:
            input_data[feature] = st.number_input(
                f"{feature}",
                help=f"Enter value for {feature}",
                format="%.4f",
                step=0.1
            )
    
    return input_data

def make_prediction(input_data, scaler):
    """Make predictions using trained models"""
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    # Collect predictions from all available models
    predictions = {}
    model_names = [key.replace('model_', '') for key in st.session_state.keys() 
                  if key.startswith('model_')]
    
    for model_name in model_names:
        model = st.session_state[f'model_{model_name}']
        prediction = model.predict(input_scaled)[0]
        predictions[model_name] = prediction
    
    return predictions

def display_predictions(predictions):
    """Display predictions in a modern layout"""
    st.markdown("### 🎯 Model Predictions")
    
    # Create columns for predictions
    cols = st.columns(len(predictions))
    for col, (model_name, prediction) in zip(cols, predictions.items()):
        with col:
            st.metric(
                label=f"{model_name}",
                value=f"{prediction:.2f}",
                help=f"Prediction from {model_name} model"
            )

def create_download_link(df, filename, file_type="csv"):
    """Create download link for predictions"""
    if file_type == "csv":
        data = df.to_csv(index=False)
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">Download CSV</a>'
    else:  # Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Predictions', index=False)
        
        b64 = base64.b64encode(buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx" class="download-button">Download Excel</a>'
    
    return href

def render_download_buttons(results_df, base_filename):
    """Render download buttons for predictions"""
    st.markdown("### 💾 Download Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        st.markdown(
            create_download_link(results_df, base_filename, "csv"),
            unsafe_allow_html=True
        )
    
    with col2:
        # Excel download
        st.markdown(
            create_download_link(results_df, base_filename, "excel"),
            unsafe_allow_html=True
        )

def handle_batch_prediction(uploaded_file, features, scaler):
    """Handle batch prediction process"""
    try:
        input_df = pd.read_csv(uploaded_file)
        
        # Verify features
        missing_features = set(features) - set(input_df.columns)
        if missing_features:
            st.error(f"❌ Missing features in input data: {', '.join(missing_features)}")
            st.info("💡 Make sure your CSV contains all required features.")
            return None
        
        # Prepare input data
        input_scaled = scaler.transform(input_df[features])
        
        # Make predictions with all available models
        predictions = {}
        for model_name in [key.replace('model_', '') for key in st.session_state.keys() 
                          if key.startswith('model_')]:
            model = st.session_state[f'model_{model_name}']
            predictions[f'{model_name}_prediction'] = model.predict(input_scaled)
        
        # Create results DataFrame
        results_df = pd.concat(
            [input_df, pd.DataFrame(predictions)],
            axis=1
        )
        
        return results_df
        
    except Exception as e:
        st.error(f"❌ Error processing batch predictions: {str(e)}")
        st.info("💡 Check your input file format and try again.")
        return None

def render_predictions_page():
    """Render the predictions page"""
    st.markdown("## 🎯 Make Predictions")
    
    # Check if models are trained
    if 'features' not in st.session_state:
        st.warning("⚠️ Please train models first!")
        st.info("Go to the Model Training page to train your models.")
        return
    
    # Get features and scaler from session state
    features = st.session_state.get('features')
    scaler = st.session_state.get('scaler')
    
    # Prediction type selection
    prediction_type = st.radio(
        "Select prediction type",
        ["Single Prediction", "Batch Prediction"],
        horizontal=True,
        help="Choose whether to predict for a single instance or multiple instances"
    )
    
    if prediction_type == "Single Prediction":
        st.markdown("### 🔍 Single Prediction")
        
        # Render input form
        input_data = render_single_prediction_form(features)
        
        # Make prediction
        if st.button("🎯 Predict", use_container_width=True):
            with st.spinner("Making predictions..."):
                predictions = make_prediction(input_data, scaler)
                display_predictions(predictions)
                
    else:  # Batch Prediction
        st.markdown("### 📊 Batch Prediction")
        
        # Show feature requirements
        with st.expander("ℹ️ CSV Format Requirements"):
            st.markdown("Your CSV file should contain the following features:")
            for feature in features:
                st.markdown(f"- `{feature}`")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for predictions",
            type="csv",
            help="Upload a CSV file containing the required features"
        )
        
        if uploaded_file is not None:
            st.markdown("#### 📄 Preview of uploaded file")
            preview_df = pd.read_csv(uploaded_file, nrows=5)
            st.dataframe(preview_df.head(), use_container_width=True)
            
            if st.button("🎯 Make Predictions", use_container_width=True):
                with st.spinner("Processing batch predictions..."):
                    results_df = handle_batch_prediction(uploaded_file, features, scaler)
                    
                    if results_df is not None:
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Add download buttons
                        render_download_buttons(results_df, "batch_predictions")