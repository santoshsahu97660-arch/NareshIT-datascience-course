
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dataset = pd.read_csv(
    r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\2nd feb\01_Restaurant_Reviews.tsv",
    delimiter='\t',
    quoting=3
)

corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = ' '.join(review)
    corpus.append(review)

y = dataset.iloc[:, 1].values


print("\n========== BAG OF WORDS ==========")

cv = CountVectorizer()
X_bow = cv.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_bow, y, test_size=0.20, random_state=0
)

# ---- Decision Tree ----
dt_bow = DecisionTreeClassifier(random_state=0)
dt_bow.fit(X_train, y_train)
y_pred_dt = dt_bow.predict(X_test)

print("\nDecision Tree Accuracy (BoW):",
      accuracy_score(y_test, y_pred_dt))

# ---- KNN ----
knn_bow = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_bow.fit(X_train, y_train)
y_pred_knn = knn_bow.predict(X_test)

print("KNN Accuracy (BoW):",
      accuracy_score(y_test, y_pred_knn))

# ---- SVM ----
svm_bow = SVC(kernel='linear', random_state=0)
svm_bow.fit(X_train, y_train)
y_pred_svm = svm_bow.predict(X_test)

print("SVM Accuracy (BoW):",
      accuracy_score(y_test, y_pred_svm))


print("\n========== TF-IDF ==========")

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.20, random_state=0
)

# ---- Decision Tree ----
dt_tf = DecisionTreeClassifier(random_state=0)
dt_tf.fit(X_train, y_train)
y_pred_dt = dt_tf.predict(X_test)

print("\nDecision Tree Accuracy (TF-IDF):",
      accuracy_score(y_test, y_pred_dt))

# ---- KNN ----
knn_tf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_tf.fit(X_train, y_train)
y_pred_knn = knn_tf.predict(X_test)

print("KNN Accuracy (TF-IDF):",
      accuracy_score(y_test, y_pred_knn))

# ---- SVM ----
svm_tf = SVC(kernel='linear', random_state=0)
svm_tf.fit(X_train, y_train)
y_pred_svm = svm_tf.predict(X_test)

print("SVM Accuracy (TF-IDF):",
      accuracy_score(y_test, y_pred_svm))



