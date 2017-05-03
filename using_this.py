'''Document prediction using trained model.'''
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

clf = joblib.load('classification.pkl')
vocabulary_to_load = joblib.load('dict.pkl')
predict_data = []
with open("./training_sets/sci.space/60214.txt", 'r') as data_file:
    predict_data.append(''.join([line.replace('\n', '') for line in data_file.readlines()]))

count_vect = CountVectorizer(vocabulary=vocabulary_to_load)
X_new_counts = count_vect.transform(predict_data)
tf_transformer = TfidfTransformer()
X_predict_tf = tf_transformer.fit_transform(X_new_counts)
print(clf.predict(X_predict_tf))
