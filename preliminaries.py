#!/usr/bin/python3

import warnings
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from collections import defaultdict, Counter, deque

warnings.filterwarnings("ignore")

# File paths
data_path = './raw_data.txt'

def default_count():
    return 0

def word_count(data_path):
    word_freq = defaultdict(default_count)
    text = ""

    with open(data_path) as fh:
        for line in fh:
            text += line

    for word in text.split():
        word_freq[word] += 1
    return dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

def get_word_count_report(word_freq, show=True):
    total_unique_words = len(word_freq.keys())
    total_words = sum(word_freq.values())
    count_100words = sum(list(word_freq.values())[:100])
    print(f"Most Common 100 words account to {(count_100words/total_words)*100}% of the corpus")

    only_1 = sum([v for v in word_freq.values() if v == 1])
    lessthan_10 = sum([1 for v in word_freq.values() if v <= 10])

    print(f"Count of hapax legomena {(only_1/total_unique_words)*100}% of the corpus")
    print(f"About {(lessthan_10/total_unique_words)*100}% of the corpus occur less than 10 times")

def freq_count(word_freq):
    freq_freq = defaultdict(default_count)

    for count in word_freq.values():
        if count in range(1, 11):
            freq_freq[str(count)] += 1
        elif count in range(11, 51):
            freq_freq['11-50'] += 1
        elif count in range(51, 101):
            freq_freq['51-100'] += 1
        else:
            freq_freq['>100'] += 1

    return dict(freq_freq)

# mandelbrot's P = 10^5.4, B = 1.15, rho = 100
def zipf_law(word_freq):
    zipf = list()

    for i, (k, v) in enumerate(word_freq.items()):
        zipf.append([k, v, i+1, v * (i+1), np.round(np.sqrt(v))])

    zipf = pd.DataFrame(columns=['Word', 'Freq.', 'Rank', 'f * r', 'Meaning'], data=zipf)

    return zipf

def calculate_relative_frequency(word_freq):
    total_count = sum(word_freq.values())
    relative_freq = defaultdict(default_count)

    for k, v in word_freq.items():
        relative_freq[k] = v/total_count

    return relative_freq

class MarkovChains():
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)
        self.freqs = Counter()
        self.buffer = deque(maxlen=self.n)
    
    def fit(self, stream):
        for token in stream:
            prefix = tuple(self.buffer)
            self.buffer.append(token)

            if len(prefix) == self.n:
                self.freqs[prefix] += 1
                self.model[prefix][token] += 1
        
        return
    
    def entropy(self, prefix):
        prefix_freqs = self.model[prefix].values()
        normalization_factor = self.freqs[prefix]

        return -np.sum(f/normalization_factor * np.log2(f/normalization_factor) 
                       for f in prefix_freqs)
                
    def entropy_rate(self):
        normalization_factor = sum(self.freqs.values())
        unnormalized_rate = np.sum(self.freqs[prefix] * self.entropy(prefix) for prefix in self.freqs)

        try:
            return unnormalized_rate/normalization_factor
        except ZeroDivisionError:
            return 0

def make_zipf_plot(zipf):
    plt.title(r"$ log_e(Frequency)\ vs\ log_e(Rank) $")
    plt.ylabel(r"$ log_e(Frequency) $")
    plt.xlabel(r"$ log_e(Rank) $")
    plt.plot(np.log(zipf['Rank']), np.log(zipf['Freq.']))
    plt.show()

    return

# Basic
data_list = []
with open(data_path, 'r') as fh:
    for line in fh:
        data_list.extend(line.split())

if __name__ == '__main__':
    word_freq = word_count(data_path)
    avg_freq = sum(word_freq.values()) / len(word_freq.values())
    get_word_count_report(word_freq)
    print("done")

    zipf = zipf_law(word_freq)
    make_zipf_plot(zipf)
    print("done")

    for i in range(3):
        mm = MarkovChains(n=i)
        mm.fit(stream=data_list)
        rate = mm.entropy_rate()
        print(f'Entropy Rate of {i}th Order is {rate}')

    # Suppressed the output as it is Huge
    # print(freq_count(word_freq))
    # print(calculate_relative_frequency(word_freq))