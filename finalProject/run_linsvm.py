# Greg Guyles
# Final Project: Text Classification
# Machine Learning
# CU Boulder
# 04/28/2014

from multiprocessing import Process
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from kaggle_test import metrics_est, train_full
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
    print("n_samples: %d\n" % len(dataset.data))

    # Build vectorizer / classifier pipeline
    text_clf = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), max_df=0.75)),
        ('clf', LinearSVC(C=1))
        ])

    metrics_out = '/home/gregor/ipyServer/movie_review/output/linsvm_metrics.out'
    kaggle_test_out = '/home/gregor/ipyServer/movie_review/output/linsvm_kaggle_test_01.csv'

    p1 = Process(target=metrics_est, args=(text_clf, dataset, metrics_out))
    p1.start()

    test_data = load_files(movie_reviews_test_data_folder, shuffle=False)
    p2 = Process(target=train_full, args=(text_clf, dataset, test_data, kaggle_test_out))
    p2.start()

    p1.join()
    p2.join()

    print('Test Compleated')