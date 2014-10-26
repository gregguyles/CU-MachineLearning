# Greg Guyles
# Final Project: Text Classification
# Machine Learning
# CU Boulder
# 04/28/2014

from multiprocessing import Process
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from kaggle_test import metrics_est, train_full
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from os.path import isfile
import sys

if __name__ == "__main__":

    # import data
    subset = sys.argv[1]
    if(subset == '-full'):
        movie_reviews_data_folder = "/home/gregor/ipyServer/data/movie_review/train"
        movie_reviews_test_data_folder = "/home/gregor/ipyServer/data/movie_review/test"
    else:
        movie_reviews_data_folder = "/home/gregor/ipyServer/data/movie_review/train_sub"
        movie_reviews_test_data_folder = "/home/gregor/ipyServer/data/movie_review/test_sub"

    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    test_data = load_files(movie_reviews_test_data_folder, shuffle=False)
    print(len(test_data))
    print("n_samples: %d\n" % len(dataset.data))

    # Build vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.1,
                            ngram_range=(1,2))

    text_clf = ExtraTreesClassifier(max_depth=1024, min_samples_leaf=8,
                                    min_samples_split=16)

    reduceParams = 80 * 1000

    kaggle_test_out = '/home/gregor/ipyServer/movie_review/output/kaggle_test_04_ExTrees.csv'


###############################################################################

    if isfile(kaggle_test_out):
        kaggle_test_out = kaggle_test_out + '.out'

    outfile_kaggle = open(kaggle_test_out, 'w+')
    outfile_kaggle.write('Id,Prediction\n')

    # train in entire set for kaggle test
    X_train = vectorizer.fit_transform(dataset.data)
    X_test = vectorizer.transform(test_data.data)
    print("Org. shape kaggle test: " + str(X_train.get_shape) + '\n')

    if(subset == '-full'):
        kbest = reduceParams
    else:
        kbest = 'all'
    ch2 = SelectKBest(chi2, k=kbest)
    X_train = ch2.fit_transform(X_train, dataset.target)
    X_test = ch2.transform(X_test)
    print("Reduced Shape kaggle test: " + str(X_train.get_shape) + '\n')

    text_clf.fit(X_train.toarray(), dataset.target)

    # predict the outcome on the traing test and store it in target_perdicted
    target_perdicted = text_clf.predict(X_test.toarray())

    for i in range(len(target_perdicted)):
        outfile_kaggle.write(str(i + 1) + ',' + str(target_perdicted[i]) + '\n')

    outfile_kaggle.close()
    print('Test Compleated')