import math
import pickle
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # ✅ NEW

DATA_DIR = "./data"
ANIME_CLEANED = f"{DATA_DIR}/anime_cleaned.csv"
USER_CLEANED = f"{DATA_DIR}/users_cleaned_1000.csv"
PREPROCESSED_INPUT = f"{DATA_DIR}/preprocessed_input_1000.csv"

def list_chunk(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def batch_predict(model, X_batches, num_batches):
    y_pred = None
    total_time = 0
    avg_time = 0

    print("========== Predicting in Batches ==========")
    for i, batch in enumerate(X_batches):
        start = time.time()
        batch = batch.astype(np.float32)
        pred = model.predict(batch)
        y_pred = pred if y_pred is None else np.concatenate((y_pred, pred))
        elapsed = time.time() - start
        total_time += elapsed
        avg_time = (avg_time + elapsed) / 2 if avg_time > 0 else elapsed
        remaining = avg_time * (num_batches - (i + 1))

        print(f"Batch {i + 1}/{num_batches}")
        print(f"Elapsed: {timedelta(seconds=total_time)} | Remaining: {timedelta(seconds=remaining)}")

    return y_pred

# Load users
user = pd.read_csv(USER_CLEANED, usecols=['username', 'user_id'])
print(user.shape, '\n', user.head(), '\n')

# Load anime data and extract unique genres
anime = pd.read_csv(ANIME_CLEANED, usecols=[
    'anime_id', 'title', 'title_english', 'score', 'scored_by', 'rank',
    'popularity', 'members', 'favorites', 'genre'
])
print(anime.shape, '\n', anime.head(), '\n')

# Extract genres
genre_list = anime['genre'].dropna().str.split(',').explode().str.strip().unique().tolist()

# Load preprocessed input
dtypes = {"my_score": "int8", "label": "category"}
dtypes.update({genre: "int8" for genre in genre_list})
dtypes.update({str(uid): "int8" for uid in user['user_id']})
dtypes.update({f"a_{aid}": "int8" for aid in anime['anime_id']})

preprocessed_input = pd.read_csv(PREPROCESSED_INPUT, dtype=dtypes).reset_index(drop=True)

print(preprocessed_input.shape)
print(preprocessed_input.head())
print(preprocessed_input.info(verbose=False, memory_usage="deep"))

# Prepare model input and output
X = preprocessed_input.drop(columns=['label', 'my_score'])
y = preprocessed_input['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
print(np.unique(y_train, return_counts=True))

# ✅ SCALE the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Training
n_estimators = 5

clf = BaggingClassifier(
    estimator=svm.LinearSVC(dual=False, C=3.5),
    n_jobs=1,
    verbose=3,
    max_samples=0.05,
    n_estimators=n_estimators
)

start = time.time()
clf.fit(X_train, y_train)
print("Training Time:", timedelta(seconds=(time.time() - start)))

# ✅ Save model
with open('./export/classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# ✅ Save scaler
with open('./export/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Test prediction (batch)
X_test_batches = list(list_chunk(X_test, 1000))
y_pred = batch_predict(clf, X_test_batches, len(X_test_batches))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, pos_label="Suka"))
print("Recall:", metrics.recall_score(y_test, y_pred, pos_label="Suka"))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka', 'Suka']))
print("(tn, fp, fn, tp):", metrics.confusion_matrix(y_test, y_pred, labels=['Tidak Suka', 'Suka']).ravel())

# Trimmed test for additional evaluation
trimmed = preprocessed_input.sample(10000)
X_trimmed = trimmed.drop(columns=['label', 'my_score'])
X_trimmed = scaler.transform(X_trimmed).astype(np.float32)  # ✅ scaled
y_trimmed = trimmed['label']
y_pred_trim = clf.predict(X_trimmed)

print("Trimmed Accuracy:", metrics.accuracy_score(y_trimmed, y_pred_trim))
print("Trimmed Precision:", metrics.precision_score(y_trimmed, y_pred_trim, pos_label="Suka"))
print("Trimmed Recall:", metrics.recall_score(y_trimmed, y_pred_trim, pos_label="Suka"))
print("Trimmed Confusion Matrix:\n", metrics.confusion_matrix(y_trimmed, y_pred_trim, labels=['Tidak Suka', 'Suka']))
print("(tn, fp, fn, tp):", metrics.confusion_matrix(y_trimmed, y_pred_trim, labels=['Tidak Suka', 'Suka']).ravel())