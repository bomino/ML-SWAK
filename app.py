import streamlit as st
from styles import apply_custom_style, nav_button
from modules.data_analysis import render_data_analysis_page
from modules.time_series import render_time_series_page
from modules.model_training import render_model_training_page
from modules.predictions import render_predictions_page
from modules.tutorial import render_tutorial_page

def main():
    st.set_page_config(
        page_title="ML Swiss Army Knife",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_style()
    
    # Initialize page state if not exist
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'tutorial'
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### ML Swiss Army Knife")
        st.markdown("---")
        
        # Navigation items
        nav_items = [
            ("tutorial", "Tutorial", "📚"),
            ("data", "Data Upload & Analysis", "📊"),
            ("timeseries", "Time Series Analysis", "⏳"),
            ("training", "Model Training", "🤖"),
            ("predictions", "Predictions", "🎯")
        ]
        
        # Create navigation buttons
        for nav_id, label, icon in nav_items:
            nav_button(label, icon, nav_id)
        
        # Add version info at bottom
        st.markdown("---")
        st.markdown("##### Version 1.0.0")
    
    # Page routing
    try:
        current_page = st.session_state['current_page']
        
        if current_page == 'data':
            render_data_analysis_page()
        elif current_page == 'timeseries':
            render_time_series_page()
        elif current_page == 'training':
            render_model_training_page()
        elif current_page == 'predictions':
            render_predictions_page()
        else:  # tutorial is default
            render_tutorial_page()
            
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.write("Please check that all required modules are properly installed.")

if __name__ == "__main__":
    main()