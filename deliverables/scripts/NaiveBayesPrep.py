import numpy as np
import pandas as pd

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


def prepare_data(train, test):

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