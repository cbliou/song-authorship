import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from collections import Counter
from sklearn.metrics import confusion_matrix


def grid_search_nonparametric(x, y, num, clf, train, test, classes, compare, useprior):
    
    grid = {}
    grid2 = {}

    for n in np.linspace(x, y, num)[1:]:
        tclf = clf(alpha = float(n), compare = compare, useprior = useprior)
        tclf.fit(train, classes)
        grid.update({n: confusion_matrix(tclf.predict(test, classes), tclf.test_Y)})
        grid2.update({n: np.diag(grid[n]).sum() / grid[n].sum()})
    
    best = sorted(grid2.items(), key = lambda x: x[1], reverse = True)[0]
    print("Best accuracy:", best[1])
    print("Parameter", best[0])

    plt.plot([i for i in grid2], [grid2[i] for i in grid2])
    
    return grid

def grid_search_cdf(x, y, num, clf, train, test, classes, compare):
    
    grid = {}
    grid2 = {}

    for n in np.linspace(x, y, num):
        tclf = clf(alpha = n, compare = compare)
        tclf.fit(train, classes)
        grid.update({n: confusion_matrix(tclf.predict(test, classes), tclf.test_Y)})
        grid2.update({n: np.diag(grid[n]).sum() / grid[n].sum()})
        
    best = sorted(grid2.items(), key = lambda x: x[1], reverse = True)[0]
    print("Best accuracy:", best[1])
    print("Parameter", best[0])

    plt.plot([i for i in grid2], [grid2[i] for i in grid2])
    
    return grid

def create_genre(row, genre):
    if re.search(genre, row["Genre"], flags = re.I) != None:
        return 1
    else:
        return 0
    
def convert_genre(y):
    """
    Convert test vector into numbers.
    """
    y = np.array(y)
    counts = Counter(y)
    itor = 0
    for genre in counts:
        y[y == genre] = itor
        itor += 1
    return y.astype(int)

def calculate_gini_index(vocabulary):
    index = {}
    
    for word in vocabulary.index:
        index.update({word: sum((vocabulary.loc[word] / vocabulary.loc[word].sum()) ** 2)})
        
    return index

def build_vocabulary(df, vocab = {}):
    """
    df: dataframe of tidy data to build vocab
    """
    index = len(vocab)

    for document in df.ID.unique():
        words = df[df.ID == document].word
        for word in words:
            if word not in vocab:
                vocab.update({word: index})
                index += 1
    return vocab


def build_word_vector(vocab, document):
    """
    vocab: dictionary with key = word, value = index of list to populate
    document: document to build a word vector from
    returns: a word vector with 0 = word not present, 1+ = number of times word shows up
    """
    vec = np.zeros(len(vocab))
    words = document.word
    for word in words:
        if word in vocab:
            vec[vocab[word]] += 1
    return vec


def prepare_binary_data(train, test):

    train_words = train[["ID", "word"]]
    test_words = test[["ID", "word"]]

    vocab = build_vocabulary(train_words)
    vocab = build_vocabulary(test_words, vocab)
    X_train, y_train, X_test, y_test = [], [], [], []

    poptrain = train[train["Pop"] == 1]
    raptrain = train[train["Rap"] == 1]

    poptest = test[test["Pop"] == 1]
    raptest = test[test["Rap"] == 1]

    # build training set

    for doc in poptrain.ID.unique():
        tmp = poptrain[poptrain.ID == doc]
        X_train.append(build_word_vector(vocab, tmp))
        y_train.append("pop")

    for doc in raptrain.ID.unique():
        tmp = raptrain[raptrain.ID == doc]
        X_train.append(build_word_vector(vocab, tmp))
        y_train.append("rap")  

    # build testing set

    for doc in poptest.ID.unique():
        tmp = poptest[poptest.ID == doc]
        X_test.append(build_word_vector(vocab, tmp))
        y_test.append("pop")

    for doc in raptest.ID.unique():
        tmp = raptest[raptest.ID == doc]
        X_test.append(build_word_vector(vocab, tmp))
        y_test.append("rap")

    return X_train, y_train, X_test, y_test