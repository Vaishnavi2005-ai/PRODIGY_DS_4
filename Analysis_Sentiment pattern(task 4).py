Step 1: Install required libraries (if running in Colab)
!pip install --quiet gspread pandas scikit-learn nltk seaborn matplotlib

# Step 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

# Step 3: Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 4: Load data from Google Sheets
"sheet_url = ""https://docs.google.com/spreadsheets/d/1g3VND33Q9qXJACd0aBPazUbv_Ug2yw1_MWQ6xn-2UAc/export?format=csv"""
data = pd.read_csv(sheet_url)

"print(""Columns in dataset:"",data.columns.tolist())"

# Step 5: Data Cleaning & Preprocessing
data=data[['text','airline_sentiment']]
data.dropna(inplace=True)

X = data['text'].astype(str) # Replace with your actual text column name
y = data['airline_sentiment'].astype(str) # Replace with your actual sentiment column name

# Step 6: Vectorization
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X_vec = vectorizer.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Step 8: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9: Evaluate model
y_pred = model.predict(X_test)
"print(""Accuracy:"", accuracy_score(y_test, y_pred))"
"print(""\nClassification Report:\n"", classification_report(y_test, y_pred))"

# Step 10: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
"sns.heatmap(cm, annot=True, fmt=""d"", cmap=""Blues"", xticklabels=model.classes_, yticklabels=model.classes_)"
"plt.title(""Confusion Matrix"")"
"plt.xlabel(""Predicted"")"
"plt.ylabel(""Actual"")"
plt.show()

# Step 11: Plot sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
"plt.title(""Sentiment Distribution"")"
"plt.xlabel(""Sentiment"")"
"plt.ylabel(""Tweet Count"")"
plt.tight_layout()
plt.show()
