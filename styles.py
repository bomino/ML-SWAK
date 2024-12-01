import streamlit as st

def apply_custom_style():
    """Apply custom styling to the Streamlit app"""
    st.markdown("""
        <style>
        /* Main app container */
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            width: 300px;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        
        /* Hide default menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Navigation buttons */
        div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
            padding: 0;
            margin: 0;
            gap: 0;
        }
        
        div[data-testid="stHorizontalBlock"] button {
            width: 100% !important;
            border: none;
            background: transparent;
            font-weight: 500;
            text-align: left;
            padding: 0.75rem 1rem;
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            color: #4b5563;
            border-radius: 0.375rem;
            gap: 0.75rem;
            transition: all 0.2s;
        }
        
        div[data-testid="stHorizontalBlock"] button:hover {
            background-color: #f1f5f9;
            color: #3b82f6;
        }
        
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background-color: transparent;
            color: #4b5563;
        }
        
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background-color: #3b82f6;
            color: white;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #1e293b;
            font-weight: 600;
        }
        
        /* Sidebar title */
        section[data-testid="stSidebar"] h3 {
            padding-left: 1rem;
            padding-right: 1rem;
            margin-bottom: 2rem;
        }
        
        /* Metrics */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        /* Dataframes */
        div[data-testid="stDataFrame"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        /* File uploader */
        div.uploadedFile {
            background-color: #ffffff;
            border: 2px dashed #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        /* Plots */
        div.stPlot > div {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        
        /* Info boxes */
        div.stAlert {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def nav_button(label, icon, key):
    """Create a custom navigation button with state management"""
    # Check if this button is active
    is_active = st.session_state.get('current_page') == key
    
    # Create full-width button container
    col1, _ = st.columns([6, 1])  # Use ratio for better button width
    with col1:
        # Create the button with proper styling
        if st.button(
            f"{icon} {label}",
            key=f"nav_{key}",
            help=f"Navigate to {label}",
            use_container_width=True,
            # Set button type based on active state
            type="primary" if is_active else "secondary"
        ):
            # Update navigation state
            st.session_state['current_page'] = key

    
    return is_active