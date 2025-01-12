# Spam Email Detection Model
# Using Naive Bayes, SVM, and Random Forest

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam_ham_dataset.csv")

# Data preprocessing
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X = data['message']
y = data['label']

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_transformed = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)

# Define evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

# Naive Bayes Model
print("Naive Bayes Model Results:")
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
evaluate_model(nb_model, X_test, y_test)

# SVM Model
print("SVM Model Results:")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_test, y_test)

# Random Forest Model
print("Random Forest Model Results:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)

# Compare models
results = {
    "Model": ["Naive Bayes", "SVM", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, nb_model.predict(X_test)),
        accuracy_score(y_test, svm_model.predict(X_test)),
        accuracy_score(y_test, rf_model.predict(X_test))
    ],
    "Precision": [
        precision_score(y_test, nb_model.predict(X_test)),
        precision_score(y_test, svm_model.predict(X_test)),
        precision_score(y_test, rf_model.predict(X_test))
    ],
    "Recall": [
        recall_score(y_test, nb_model.predict(X_test)),
        recall_score(y_test, svm_model.predict(X_test)),
        recall_score(y_test, rf_model.predict(X_test))
    ],
    "F1-score": [
        f1_score(y_test, nb_model.predict(X_test)),
        f1_score(y_test, svm_model.predict(X_test)),
        f1_score(y_test, rf_model.predict(X_test))
    ]
}

# Create DataFrame for Results
results_df = pd.DataFrame(results)
print(results_df)

# Visualization
results_df.set_index("Model").plot(kind="bar", figsize=(10, 6))
plt.title("Model Comparison")
plt.ylabel("Performance Metrics")
plt.xticks(rotation=0)
plt.show()
