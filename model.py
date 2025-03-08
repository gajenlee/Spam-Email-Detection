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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, clasclassification_report
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
    print(f"\nModel: {type(model).__name__}")
    print("Classification Report:\n", clasclassification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": accuracy_score(y_test, y_pred),
        "Recall": accuracy_score(y_test, y_pred),
        "F1-score": accuracy_score(y_test, y_pred)
    }


# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=43, class_weight='balanced')
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate_model(model, X_train, y_train)

# Convert results to DataFrame:
results_df = pd.DataFrame(results).T
print(results_df)

# Visualization
plt.figure(figsize=(10, 6))
results_df.plot(kind="bar", figsize=(10, 6), colormap='viridis')
plt.title("Spam Detection Model Comparison")
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()