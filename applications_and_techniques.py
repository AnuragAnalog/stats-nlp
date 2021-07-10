#!/usr/bin/python3

import nltk
import numpy
from nltk import decorators
from nltk.cluster import GAAClusterer

# Download packages
nltk.download('stopwords')

stemmer_func = nltk.stem.PorterStemmer().stem
stopwords = set(nltk.corpus.stopwords.words('english'))

@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())

def get_words(titles):
    words = set()

    for title in sentences:
        for word in title.split():
            words.add(normalize_word(word))

    return list(words)

@decorators.memoize
def vectorspaced(title):
    title_components = [normalize_word(word) for word in title.split()]

    return numpy.array([
        word in title_components and not word in stopwords
        for word in words], numpy.short)

if __name__ == '__main__':
    filename = 'raw_data.txt'

    with open(filename) as fh:
        sentences = [line.strip() for line in fh.readlines()]

        words = get_words(sentences)

        cluster = GAAClusterer(5)
        cluster.cluster([vectorspaced(title) for title in sentences if title])

        classified_examples = [
                cluster.classify(vectorspaced(title)) for title in sentences
            ]

        for cluster_id, title in sorted(zip(classified_examples, sentences)):
            print(cluster_id, title)