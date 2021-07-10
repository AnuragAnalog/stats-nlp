#!/usr/bin/python3

# Statistical N-grams
import numpy as np

# MLE: Maximum Likelihood estimate
def MLE(frq, n):
    return frq/n

# Laplace's Law
def Laplace(frq, n,bins):
    return (frq+1)/(n+bins)

# Held out estimator
def heldout(frq, Nr, N):
    return np.sum(frq)/(Nr*N)

# Deleted estimator
def del_est(frq1, frq2, Nr1, Nr2, N):
    return np.sum(frq1 + frq2)/((Nr1+Nr2)*N)

# Good Turing Estimator
def gotur(exp, r, N):
    return ((r+1)/N)*(exp[r+1]/exp[r])

# Katz Backoff
def katz_backoff(frq, i, k, d):
    if frq[i]>k:
        return 1 - d[i]