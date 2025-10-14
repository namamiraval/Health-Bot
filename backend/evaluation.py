# backend/evaluation.py
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import retrieve
import re
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Datasets/Dipiro-Handbook.pdf')

y_true = df["correct_answer"].tolist()
y_pred = []

# Evaluate each user query
for question in df["user_input"]:
    answer, score, source = retrieve(question, top_k=1)
    y_pred.append(answer if answer else "no_answer")

# Basic text cleaning for better matching
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()

y_true_clean = [normalize(a) for a in y_true]
y_pred_clean = [normalize(a) for a in y_pred]

# Measure overlap-based accuracy (exact match or similarity)
matches = [1 if yt in yp or yp in yt else 0 for yt, yp in zip(y_true_clean, y_pred_clean)]
accuracy = sum(matches) / len(matches)

print("Evaluation Results:")
print(f"Accuracy: {accuracy:.2f}")

# Optional — precision/recall/F1 (binary 1 if correct else 0)
precision = precision_score(matches, matches, zero_division=0)
recall = recall_score(matches, matches, zero_division=0)
f1 = f1_score(matches, matches, zero_division=0)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
