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
