import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Simulated dataset for regression analysis based on provided context
# IVs: 'dataset_size', 'preprocessing_score', 'feature_complexity', 'algorithm_score', 'validation_score'
# DV: 'performance' (average of Accuracy, Precision, Recall, F1-Score)
data = pd.DataFrame({
    'dataset_size': [500, 1000, 1500, 2000, 2500, 3000],
    'preprocessing_score': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    'feature_complexity': [1, 2, 3, 4, 5, 6],
    'algorithm_score': [0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'validation_score': [0.8, 0.85, 0.9, 0.95, 0.97, 1.0],
    'performance': [0.72, 0.77, 0.82, 0.88, 0.91, 0.96]
})

# Separate independent variables (IVs) and dependent variable (DV)
X = data[['dataset_size', 'preprocessing_score', 'feature_complexity', 'algorithm_score', 'validation_score']]
y = data['performance']

# Add a constant term for the intercept in the regression model
X = sm.add_constant(X)

# Perform the regression analysis
model = sm.OLS(y, X).fit()

# Summarize the regression results
regression_summary = model.summary()

# Display the summary
print(regression_summary)
