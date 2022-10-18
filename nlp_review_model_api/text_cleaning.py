# text preprocessing modules
from string import punctuation

from nltk.corpus import stopwords

import nltk
from nltk.stem import WordNetLemmatizer
import re

for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset", "omw-1.4"):
    nltk.download(dependency)


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
        stop_words = stopwords.words('english')
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    return text
