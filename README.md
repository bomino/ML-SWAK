# ML Swiss Army Knife 🛠️

An intuitive, user-friendly machine learning application that simplifies the entire ML workflow from data analysis to model deployment. Built with Streamlit, this tool provides a comprehensive suite of ML capabilities accessible through a modern web interface.

![ML Swiss Army Knife Interface](path_to_screenshot.png)

## 🌟 Key Features

### 📊 Data Analysis & Preprocessing
- Interactive data upload and preview
- Automated data type detection and quality analysis
- Missing value visualization and handling
- Feature correlation analysis
- Automated data preprocessing pipeline

### 🤖 Model Training & Evaluation
- **Traditional Models**
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting
  - SVR (Support Vector Regression)

- **Advanced Models**
  - XGBoost
  - LightGBM
  - Prophet (Time Series)
  - SARIMA (Time Series)

- **Features**
  - Automated feature preprocessing
  - Model performance metrics
  - Cross-validation support
  - Feature importance analysis
  - Interactive parameter tuning

### 📈 Visualization
- Feature distribution plots
- Correlation heatmaps
- Model performance comparisons
- Prediction vs Actual plots
- Time series forecasting plots

### 📚 Interactive Tutorial
- Step-by-step guidance
- Best practices
- Troubleshooting tips
- Real-world examples
- Advanced topics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-swiss-army-knife.git
cd ml-swiss-army-knife
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run app.py
```

## 📖 Usage Guide

### Basic Workflow

1. **Data Upload**
```python
# Example CSV structure
date,feature1,feature2,target
2024-01-01,23.5,high,100
2024-01-02,24.1,low,95
```

2. **Data Preprocessing**
- Select features for encoding
- Handle missing values
- Scale numerical features

3. **Model Training**
```python
# Example model configuration
model_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5
}
```

4. **Evaluation & Prediction**
- View performance metrics
- Analyze feature importance
- Make predictions on new data

### Advanced Features

#### Time Series Forecasting
```python
# Example Prophet configuration
prophet_params = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False
}
```

#### Custom Model Training
- Parameter tuning
- Cross-validation
- Feature selection

## 🔧 Configuration

### System Requirements
- RAM: 8GB minimum (16GB recommended)
- Storage: 1GB free space
- Processor: Multi-core processor recommended

### Environment Variables
```bash
# Optional configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

## 📁 Project Structure

```
ml-swiss-army-knife/
├── .streamlit/               # Streamlit configuration
│   └── config.toml          # Theme and settings
├── app.py                   # Main application
├── requirements.txt         # Dependencies
├── README.md               # Documentation
├── styles.py               # Custom styling
└── modules/               # Application modules
    ├── data_analysis.py    # Data analysis functionality
    ├── time_series.py      # Time series analysis
    ├── model_training.py   # Model training functionality
    ├── predictions.py      # Prediction functionality
    └── tutorial.py         # Tutorial content
```


## 💻 Usage Guide

1. Start the application
2. Upload your data (CSV format)
3. Explore data insights automatically
4. Train models with guided selection
5. Make predictions with trained models

## 🔧 Configuration

### Theme Configuration
Located in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#7C3AED"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3F4F6"
textColor = "#111827"
font = "sans serif"
```

## 🔍 Example Use Cases

### 1. Sales Forecasting
```python
# Sample data structure
sales_data = {
    'date': ['2024-01-01', '2024-01-02'],
    'sales': [1000, 1200],
    'promotion': ['yes', 'no']
}
```

### 2. Category Prediction
```python
# Sample categorical features
category_data = {
    'feature1': ['A', 'B', 'C'],
    'feature2': [1, 2, 3],
    'target': ['cat1', 'cat2', 'cat1']
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/AmazingFeature
```
3. Commit changes:
```bash
git commit -m 'Add AmazingFeature'
```
4. Push to branch:
```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **Installation Problems**
```bash
# If you encounter SSL errors
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

2. **Memory Issues**
```python
# Reduce memory usage
import pandas as pd
pd.read_csv('large_file.csv', nrows=1000)  # Load subset for testing
```

## 📊 Performance Tips

1. **Large Datasets**
- Use chunked processing
- Implement memory optimization
- Consider data sampling

2. **Model Training**
- Start with simple models
- Use cross-validation
- Monitor resource usage

## 📝 Version History

- v1.0.0 (2024-01-01)
  - Initial release
  - Basic model support
  - Data preprocessing

- v1.1.0 (2024-02-01)
  - Added time series support
  - Improved visualization
  - Bug fixes

## 📫 Support

- Create an issue on GitHub
- Email: Bomino@mlawali.com
- Documentation: [Wiki Link]

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team
- Scikit-learn community
- All contributors

---
Made with ❤️ by [Mlawali]