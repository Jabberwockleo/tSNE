#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2019 Wan Li. All Rights Reserved
#
########################################################################

"""
File: tsne_sklearn.py
Author: Wan Li
Date: 2019/01/05 21:30:35
"""

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

def tsne_project(X, y, out_file="./tsne.png"):
    """
        Run tSNE projection
    """
    y_set = list(set(y))
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    target_ids = range(len(y_set))
    plt.figure(figsize=(12, 9))
    for i, label in zip(target_ids, y_set):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=label, cmap=plt.cm.cool)
    plt.legend()
    plt.savefig('tsne.png')


if __name__ == "__main__":
    from sklearn import datasets
    digits = datasets.load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    tsne_project(X, y)
