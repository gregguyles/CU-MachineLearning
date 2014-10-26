# Greg Guyles
# Final Project: Text Classification
# Machine Learning
# CU Boulder
# 04/28/2014

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile

def metrics_est(text_clf, dataset, metrics_out):

    outfile = open(metrics_out, 'a+')
    outfile.write(('*' * 70)+'\n')
    outfile.write(('*' * 5) + (' ' * 25) + 'Begin  Run' + (' ' * 25) + ('*' * 5) + '\n')
    outfile.write(('*' * 70)+'\n\n')

       # split data into train and test sets
    mov_train, mov_test, target_train, target_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    text_clf.fit(mov_train, target_train)

    # predict the outcome on the traing test and store it in target_perdicted
    target_perdicted = text_clf.predict(mov_test)
    score = text_clf.score(mov_test, target_test)
    # outfile.write classification report
    outfile.write('Accuracy Score: ' + str(score) + '\n')

    # outfile.write and plot
    cm = metrics.confusion_matrix(target_test, target_perdicted)
    outfile.write("Confusion Matrix:\n")
    outfile.write("      neg     pos\n")
    cm_str = format("neg%6i  %6i\npos%6i  %6i\n" % 
                (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))
    outfile.write(cm_str)

    outfile.write(('#' * 70)+'\n')
    outfile.write(('#' * 5) + (' ' * 25) + 'End    Run' + (' ' * 25) + ('#' * 5) + '\n')
    outfile.write(('#' * 70) + '\n\n')

    outfile.close()

def train_full(text_clf, dataset, test_data, train_full_out):

    if isfile(train_full_out):
        train_full_out = train_full_out + '.out'

    outfile = open(train_full_out, 'w+')
    outfile.write('Id,Prediction\n')

    # train in entire set for kaggle test
    text_clf.fit(dataset.data, dataset.target)

    # predict the outcome on the traing test and store it in target_perdicted
    target_perdicted = text_clf.predict(test_data.data)

    for i in range(len(target_perdicted)):
        outfile.write(str(i + 1) + ',' + str(target_perdicted[i]) + '\n')

    outfile.close()