from sklearn.datasets import fetch_20newsgroups
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
#print(dataset.data)
#print(dataset.target_names)
print(type(dataset.data))
