'''Using Classification'''
import os
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

dataset = []
labels = []
for subdir in os.listdir("./training_sets"):
    for filename in os.listdir("./training_sets/"+subdir):
        with open("./training_sets/"+subdir+"/"+filename, 'r') as data_file:
            dataset.append(''.join([line.replace('\n', '') for line in data_file.readlines()]))
            labels.append(subdir)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataset)
joblib.dump(count_vect.vocabulary_, 'dict.pkl')
print('Export dict successful.')

tf_transformer = TfidfTransformer()
X_train_tf = tf_transformer.fit_transform(X_train_counts)

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
clf.fit(X_train_tf, labels)
predicted = cross_val_predict(clf, X_train_tf, labels, cv=10)

print(metrics.accuracy_score(labels, predicted))
joblib.dump(clf, 'classification.pkl')
print('Export model successful.')
