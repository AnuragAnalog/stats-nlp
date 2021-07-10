#!/usr/bin/python3

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

stop_words = set(stopwords.words('english'))
  
with open('raw_data.txt') as fh:
    for idx, line in enumerate(fh):
        if idx > 2000:
            break
        wordsList = nltk.word_tokenize(line)
        wordsList = [w for w in wordsList if not w in stop_words] 
        tagged = nltk.pos_tag(wordsList)

        print(tagged)