# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = './spam_ham_dataset.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

print(data)


# # Feature extraction using TF-IDF
# vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
# X_transformed = vectorizer.fit_transform(X)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# # Model training and evaluation function
# def evaluate_model(model, name):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     return {
#         'Model': name,
#         'Accuracy': round(accuracy_score(y_test, y_pred), 4),
#         'Precision': round(precision_score(y_test, y_pred), 4),
#         'Recall': round(recall_score(y_test, y_pred), 4),
#         'F1-score': round(f1_score(y_test, y_pred), 4)
#     }

# # Train and evaluate models
# models = [
#     (MultinomialNB(), "Naive Bayes"),
#     (SVC(kernel='linear'), "SVM"),
#     (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
# ]

# results = [evaluate_model(model, name) for model, name in models]

# # Create DataFrame for results
# results_df = pd.DataFrame(results)
# print(results_df)

# # Calculate statistical metrics for filling the table
# summary = {
#     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
#     'Mean': results_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].mean().round(4).values,
#     'STD. Deviation': results_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].std().round(4).values,
#     'Median': results_df[['Accuracy', 'Precision', 'Recall', 'F1-score']].median().round(4).values
# }

# summary_df = pd.DataFrame(summary)
# print(summary_df)
