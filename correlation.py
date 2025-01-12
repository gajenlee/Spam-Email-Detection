# Correlation Analysis for Spam Detection

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (example dataset for spam detection)
data = pd.read_csv('./spam_ham_dataset.csv')

# Preprocessing the dataset
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_transformed = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Define a function to calculate metrics and correlations
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

# Store results for each model
results = {}
for name, model in models.items():
    results[name] = calculate_metrics(model, X_train, X_test, y_train, y_test)

# Create a DataFrame of the results
results_df = pd.DataFrame(results).T

# Display correlation matrix
correlation_matrix = results_df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Matrix of Model Performance Metrics")
plt.show()
