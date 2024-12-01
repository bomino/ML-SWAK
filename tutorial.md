# Tutorial: Making the Most of Model Training Page

## 1. Understanding Your Data

First, let's analyze your specific dataset (UCHCategoryData-InScope.csv):
- **Numeric Column**: 'Spend' (the only numeric column)
- **Categorical Columns**: 
  - Normalized Supplier
  - CategoryLevel1
  - Final Category
  - In Scope
  - TransactionGroup

## 2. Data Preprocessing

### Categorical Column Encoding
When selecting categorical columns to encode, consider:

✅ **DO Encode**:
- `TransactionGroup`: Important because it represents spending ranges
- `CategoryLevel1`: Represents broad business categories
- `Final Category`: Provides specific categorization

❌ **Consider NOT Encoding**:
- `Normalized Supplier`: Too many unique values could lead to excessive features
- `In Scope`: If all values are the same (all "In Scope"), it won't provide useful information

**Best Practice**: Start with `CategoryLevel1` and `Final Category` for encoding. These provide meaningful business context without creating too many features.

### Why This Matters
- Each categorical column will be converted into multiple binary columns (one-hot encoding)
- Example: If `CategoryLevel1` has values ["Facilities", "Marketing", "Human Resources"], it will create:
  - CategoryLevel1_Facilities (0 or 1)
  - CategoryLevel1_Marketing (0 or 1)
  - CategoryLevel1_Human_Resources (0 or 1)

## 3. Feature Selection

### Selecting Target Variable
Choose based on your business goal:

1. **For Spend Analysis**:
   - Target: `Spend`
   - Use Case: Predict spending based on category and transaction group

2. **For Category Prediction**:
   - Target: `CategoryLevel1` or `Final Category`
   - Use Case: Automatically categorize new vendors/transactions

### Selecting Features

**For Spend Prediction**:
Best Features:
1. `TransactionGroup_*` (encoded columns)
2. `CategoryLevel1_*` (encoded columns)
3. `Final Category_*` (encoded columns)

**For Category Prediction**:
Best Features:
1. `Spend`
2. `TransactionGroup_*` (encoded columns)

### Feature Selection Tips:
1. **Start Small**: Begin with 3-5 most relevant features
2. **Monitor Correlation**: Use the correlation matrix to:
   - Identify strongly correlated features (potentially redundant)
   - Find features with strong correlation to your target
3. **Iterative Process**: 
   - Start with a basic model
   - Add/remove features based on model performance
   - Watch for overfitting when adding too many features

## 4. Model Training Best Practices

### Test Size Selection
- Standard: 20% (0.2)
- Larger dataset (>10,000 rows): Consider 10-15%
- Smaller dataset (<1,000 rows): Consider 25-30%

### Model Selection
Based on your data:

1. **For Spend Prediction (Numeric Target)**:
   - Start with: Ridge Regression (good baseline)
   - Then try: Random Forest (handles non-linear relationships)
   - Advanced: Gradient Boosting (often best performance)

2. **For Category Prediction (Categorical Target)**:
   - Start with: Random Forest
   - Then try: Gradient Boosting

### Model Comparison Metrics
For Spend Prediction:
- **RMSE**: Lower is better, shows average prediction error in same units as spend
- **R2**: Higher is better, shows percentage of variance explained
- **MAE**: Lower is better, shows average absolute error

## 5. Step-by-Step Process

1. **Initial Setup**:
   ```
   a. Select CategoryLevel1 and Final Category for encoding
   b. Target Variable: Spend
   c. Features: Start with encoded CategoryLevel1 and TransactionGroup
   d. Test Size: 0.2 (20%)
   e. Model: Ridge Regression
   ```

2. **Evaluate Initial Results**:
   - Check RMSE and R2 scores
   - Review feature importance (for tree-based models)

3. **Iterate and Improve**:
   ```
   a. Add more relevant features
   b. Try different models
   c. Adjust test size if needed
   d. Remove highly correlated features
   ```

## 6. Common Issues and Solutions

1. **Too Many Features**:
   - Problem: Model becomes slow or overfits
   - Solution: Use correlation threshold to remove highly correlated features

2. **Poor Model Performance**:
   - First try: Add more relevant features
   - Then try: Different model types
   - Finally: Feature engineering (e.g., combining categories)

3. **Overfitting**:
   - Signs: Great training performance, poor test performance
   - Solution: Reduce features, increase test size, use simpler models

## 7. Best Practices for Your Specific Data

1. **Spend Analysis Focus**:
   ```python
   # Best Feature CombinationError loading data: module 'streamlit' has no attribute 'experimental_rerun'
   features = [
       'CategoryLevel1_*',  # Business category impact
       'TransactionGroup_*',  # Size of transaction impact
       'Final Category_*'   # Specific category nuances
   ]
   ```

2. **Category Analysis Focus**:
   ```python
   # Best Feature Combination
   features = [
       'Spend',            # Main numeric indicator
       'TransactionGroup_*' # Transaction size context
   ]
   ```

Remember: The goal is to balance model complexity with performance. Start simple and add complexity only when it provides clear benefits.