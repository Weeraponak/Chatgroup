import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump

data = pd.read_csv("bbc-text.csv")
print(data['category'].value_counts())
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_text, train_cat)
dump(model, 'chatgroup.model')
labels = model.predict(test_text)
testcat = [i for i in test_cat]
n = (len(test_cat))
corrects = [ 1 for i in range(n) if testcat[i] == labels[i] ]
print('Total testdata : ', n)
print('Corrects : ', sum(corrects))
print('Accuracy :', round(sum(corrects)*100/n, 2), '%')

