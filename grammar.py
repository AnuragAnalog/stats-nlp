#!/usr/bin/python3

# Importing libraries
import time
import nltk
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english'))

nltk_data = list()
# tokenized = sent_tokenize(open('raw_data.txt').readlines())
with open('raw_data.txt', 'r') as fh:
    for i, line in enumerate(fh):
        wordsList = nltk.word_tokenize(line)
        wordsList = [w for w in wordsList if not w in stop_words] 
        tagged = nltk.pos_tag(wordsList)
        nltk_data.append(tagged)

# Create tarin and test sets
train_set,test_set = train_test_split(nltk_data, train_size=0.20, random_state = 88)

# Create list of train and test tagged words
train_tagged_words = [ tup for sent in train_set for tup in sent ]
test_tagged_words = [ tup for sent in test_set for tup in sent ]

# check some of the tagged words.
train_tagged_words[:5]

#use set datatype to check how many unique tags are present in training data
tags = {tag for _, tag in train_tagged_words}

# check total words in vocabulary
vocab = {word for word, _ in train_tagged_words}

# Compute Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]

    #  Now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)

    return (count_w_given_tag, count_tag)

# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0

    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1

    return (count_t2_t1, count_t1)

tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, t1 in enumerate(list(tags)):
    for j, t2 in enumerate(list(tags)):
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

#the table is same as the transition table shown in section 3 of article
tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))

def Viterbi(words, train_bag = train_tagged_words):
    state = list()
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = list()
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)

        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)

    return list(zip(words, state))

# Let's test our Viterbi algorithm on a few sample sentences of test dataset
random.seed(1234)

# choose random 10 numbers
random = [random.randint(1, len(test_set)) for _ in range(100)]

# list of 10 sents on which we test the model
test_run = [test_set[i] for i in random]

# list of tagged words
test_run_base = [tup for sent in test_run for tup in sent]

# list of untagged words
test_tagged_words = [tup[0] for sent in test_run for tup in sent]

#as testing the whole training set takes huge amount of time
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()

check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
accuracy = len(check)/len(tagged_seq)

print("Time taken in seconds: ", end-start)
print('Non-Rule based Viterbi Algorithm Accuracy: ',accuracy*100)

# #To improve the performance,we specify a rule base tagger for unknown words 
# # specify patterns for tagging
patterns = [
    (r'.*ing$', 'VERB'),              # gerund
    (r'.*ed$', 'VERB'),               # past tense 
    (r'.*es$', 'VERB'),               # verb    
    (r'.*\'s$', 'NOUN'),              # possessive nouns
    (r'.*s$', 'NOUN'),                # plural nouns
    (r'\*T?\*?-[0-9]+$', 'X'),        # X
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'), # cardinal numbers
    (r'.*', 'NOUN')                   # nouns
]
 
# rule based tagger
rule_based_tagger = nltk.RegexpTagger(patterns)

# #modified Viterbi to include rule based tagger in it
def Viterbi_rule_based(words, train_bag = train_tagged_words):
    state = list()
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = list()
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        state_max = rule_based_tagger.tag([word])[0][1]


        if pmax == 0:
            state_max = rule_based_tagger.tag([word])[0][1] # assign based on rule based tagger
        else:
            if state_max != 'X':
                # Getting state for which probability is maximum
                state_max = T[p.index(pmax)]

        state.append(state_max)
    return list(zip(words, state))

# Test accuracy on subset of test data
start = time.time()
tagged_seq = Viterbi_rule_based(test_tagged_words)
end = time.time()

check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
accuracy = len(check)/len(tagged_seq)
print("Time taken in seconds: ", end-start)
print('Rule Based Viterbi Algorithm Accuracy: ',accuracy*100)

#Check how a sentence is tagged by the two POS taggers and compare them
test_sent = "Will can see Marry"
pred_tags_rule = Viterbi_rule_based(test_sent.split())
pred_tags_withoutRules = Viterbi(test_sent.split())
print("\nRule based POS Tagging", pred_tags_rule)
print("\nNon-Rule based POS Tagging", pred_tags_withoutRules)
#Will and Marry are tagged as NUM as they are unknown words for Viterbi Algorithm


