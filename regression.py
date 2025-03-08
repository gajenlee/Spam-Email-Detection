import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = './spam_ham_dataset.csv'
data = pd.read_csv(file_path)

# Preprocess the dataset
data = data.dropna()  # Drop rows with missing values
if 'label_num' in data.columns:
    data['label_num'] = data['label_num'].astype(int)  # Ensure 'label_num' is integer
else:
    raise ValueError("Column 'label_num' not found in dataset")

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))  # Improved text representation
X = vectorizer.fit_transform(data['message']).toarray()  # Convert to dense matrix
y = data['label_num']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform regression analysis
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Regression Analysis Results:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Coefficients of the features
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_
top_features = sorted(zip(coefficients, feature_names), reverse=True, key=lambda x: abs(x[0]))[:10]

print("\nTop 10 Features Influencing the Model:")
for coef, feature in top_features:
    print(f"{feature}: {coef:.6f}")

# Visualization of top features
plt.figure(figsize=(10, 5))
features, coef_values = zip(*top_features)
plt.barh(features, coef_values, color='skyblue')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Top 10 Most Influential Features")
plt.gca().invert_yaxis()
plt.show()
