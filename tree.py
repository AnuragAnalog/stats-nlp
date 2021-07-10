#!/usr/bin/python3

import time
import nltk
import numpy as np
from functools import reduce    
from nltk.corpus import treebank
from nltk.grammar import toy_pcfg1
from nltk.parse import pchart, ViterbiParser
from nltk import Tree, CFG, PCFG, induce_pcfg, nonterminals, Nonterminal, Production

idx = np.random.randint(2)

nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[idx].draw()

# Parse a tree from a string with parentheses.
s = '(S (NP (DT A) (NN Aeroplane)) (VP (VBD landed) (NP (DT in) (NN Hyderabad))))'
t = Tree.fromstring(s)
print("Convert bracketed string into tree:")
print(t)
print(t.__repr__())

# Demonstrate tree modification.
the_cat = t[0]
the_cat.insert(1, Tree.fromstring('(JJ big)'))
print("Tree modification:")
print(t)
t[1,1,1] = Tree.fromstring('(NN cake)')
print(t)


# Tree transforms
print("Collapse unary:")
t.collapse_unary()
print(t)
print("Chomsky normal form:")
t.chomsky_normal_form()
print(t)


# Demonstrate probabilistic trees.
pt = nltk.tree.ProbabilisticTree('x', ['y', 'z'], prob=0.5)
print("Probabilistic Tree:")
print(pt)

# Demonstrate tree nodes containing objects other than strings
t.set_label(('test', 3))
print(t)

# Create some nonterminals
S, NP, VP, PP = nonterminals('S, NP, VP, PP')
N, V, P, Det = nonterminals('N, V, P, Det')
VP_slash_NP = Nonterminal('VP/NP')

print('Some nonterminals:', [S, NP, VP, PP, N, V, P, Det, VP_slash_NP])
print('S.symbol() =>', S.symbol())


print(Production(S, [NP]))

# Create some Grammar Productions

grammar = CFG.fromstring("""
  S -> NP VP
  PP -> P NP
  NP -> Det N | NP PP
  VP -> V NP | VP PP
  Det -> 'a' | 'the'
  N -> 'dog' | 'cat'
  V -> 'chased' | 'sat'
  P -> 'on' | 'in'
""")

print('A Grammar:', grammar)
print('grammar.start()   =>', grammar.start())
print('grammar.productions() =>')
# Use string.replace(...) is to line-wrap the output.
print(grammar.productions())

print('Coverage of input words by a grammar:')
try:
    grammar.check_coverage(['a','dog'])
    print("All words covered")
except:
    print("Strange")
try:
    print(grammar.check_coverage(['a','toy']))
except:
    print("Some words not covered")

toy_pcfg1 = PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.5] | NP PP [0.25] | 'Anurag' [0.1] | 'I' [0.15]
    Det -> 'the' [0.8] | 'my' [0.2]
    N -> 'man' [0.5] | 'binoculars' [0.5]
    VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
    V -> 'ate' [0.35] | 'saw' [0.65]
    PP -> P NP [1.0]
    P -> 'with' [0.61] | 'under' [0.39]
""")

pcfg_prods = toy_pcfg1.productions()

pcfg_prod = pcfg_prods[2]
print('A PCFG production:', pcfg_prod)
print('pcfg_prod.lhs()  =>', pcfg_prod.lhs())
print('pcfg_prod.rhs()  =>', pcfg_prod.rhs())
print('pcfg_prod.prob() =>', pcfg_prod.prob())

# extract productions from three trees and induce the PCFG
print("Induce PCFG grammar from treebank data:")
productions = []
for item in treebank.fileids()[:2]:
  for tree in treebank.parsed_sents(item):
    # perform optional tree transformations, e.g.:
    tree.collapse_unary(collapsePOS = False)
    tree.chomsky_normal_form(horzMarkov = 2)
    productions += tree.productions()

S = Nonterminal('S')
grammar = induce_pcfg(S, productions)

print("Parse sentence using induced grammar:")

parser = pchart.InsideChartParser(grammar)
parser.trace(3)

sent = treebank.parsed_sents('wsj_0001.mrg')[0].leaves()

demos = [('I saw Anurag with my binoculars', toy_pcfg1)]
sent, grammar = demos[0]

# Tokenize the sentence.
tokens = sent.split()

# Define a list of parsers.  We'll use all parsers.
parsers = [
ViterbiParser(grammar),
pchart.InsideChartParser(grammar),
pchart.RandomChartParser(grammar),
pchart.UnsortedChartParser(grammar),
pchart.LongestChartParser(grammar),
pchart.InsideChartParser(grammar, beam_size = len(tokens)+1)
]

# Run the parsers on the tokenized sentence.
times = list()
average_p = list()
num_parses = list()
all_parses = {}
for parser in parsers:
    print('\ns: %s\nparser: %s\ngrammar: %s' % (sent,parser,grammar))
    parser.trace(3)
    t = time.time()
    parses = parser.parse_all(tokens)
    times.append(time.time()-t)
    if parses:
        lp = len(parses)
        p = reduce(lambda a,b:a+b.prob(), parses, 0.0)
    else:
        p = 0
    average_p.append(p)
    num_parses.append(len(parses))
    for p in parses:
        all_parses[p.freeze()] = 1

# Print summary statistics
print()
print('-------------------------+------------------------------------------')
print('   Parser           Beam | Time (secs)   # Parses   Average P(parse)')
print('-------------------------+------------------------------------------')
for i in range(len(parsers)):
    print('%19s %4d |%11.4f%11d%19.14f' % (parsers[i].__class__.__name__,
      getattr(parsers[0], "beam_size", 0),
      times[i],
      num_parses[i],
      average_p[i]))
parses = all_parses.keys()

if parses:
    p = reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
else:
    p = 0

print('-------------------------+------------------------------------------')
print('%19s      |%11s%11d%19.14f' % ('(All Parses)', 'n/a', len(parses), p))
print()

for parse in parses:
    print(parse)