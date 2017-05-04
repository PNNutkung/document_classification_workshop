'''Document prediction using trained model.'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
import pickle

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

km = joblib.load('model.pkl')
test_data = []
with open("./training_sets/sci.space/60214.txt", 'r') as data_file:
    test_data.append(''.join([line.replace('\n', '') for line in data_file.readlines()]))
print(km.predict(vectorizer.transform(test_data)))
