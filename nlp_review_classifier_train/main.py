import warnings
import numpy as np
import pandas as pd

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# text preprocessing modules
from string import punctuation

from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# save model
import joblib

import os

for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset", "omw-1.4"):
    nltk.download(dependency)

warnings.filterwarnings("ignore")

np.random.seed(123)

data = pd.read_csv(os.path.abspath(__file__) +
                   "\..\data\labeledTrainData.tsv", sep='\t')

stop_words = stopwords.words('english')


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # clean the text, with the option to remove stop words and lemmatize word

    # clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers

    # remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text


# clean the review
data["cleaned_review"] = data["review"].apply(text_cleaning)

# split features and target from data
x = data["cleaned_review"]
y = data.sentiment.values

# split data into train and validate
x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)

# create a classifier in pipeline
sentiment_classifier = Pipeline(steps=[
    ('pre_proccessing', TfidfVectorizer(lowercase=False)
     ), ('naive_bayes', MultinomialNB())
])

# train the sentiment classifier
sentiment_classifier.fit(x_train, y_train)

# test model performance on valid data
y_preds = sentiment_classifier.predict(x_valid)

joblib.dump(sentiment_classifier, os.path.abspath(
    __file__) + '\..\models\sentiment_model_pipeline.pkl')
