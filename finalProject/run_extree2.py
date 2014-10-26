# Greg Guyles
# Final Project: Text Classification
# Machine Learning
# CU Boulder
# 04/28/2014

from multiprocessing import Process
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import ExtraTreesClassifier
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
    if(subset == '-sub'):
        movie_reviews_data_folder = "/home/gregor/ipyServer/data/movie_review/train_sub"
        movie_reviews_test_data_folder = "/home/gregor/ipyServer/data/movie_review/test_sub"
    else:
        movie_reviews_data_folder = "/home/gregor/ipyServer/data/movie_review/train"
        movie_reviews_test_data_folder = "/home/gregor/ipyServer/data/movie_review/test"

    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    n_samples = len(dataset.data)

    if(subset != '-sub' and subset != '-full'):
        n_samples = int(float(subset) * n_samples)
        dataset.data = dataset.data[:n_samples] + dataset.data[len(dataset.data)-n_samples:]
        dataset.target = np.append(dataset.target[:n_samples],
                            dataset.target[len(dataset.target)-n_samples:])

    print("Number of Samples: %d\n" % n_samples)   
    # Build vectorizer / classifier pipeline

    mov_train, mov_test, target_train, target_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_selection import SelectKBest, chi2

    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.1,
                                 ngram_range=(1,2))
    #
    X_train = vectorizer.fit_transform(mov_train)
    X_test = vectorizer.transform(mov_test)

    print(X_train.get_shape)

    ch2 = SelectKBest(chi2, k=(100 * 1000))
    X_train = ch2.fit_transform(X_train, target_train)
    X_test = ch2.transform(X_test)

    print(X_train.get_shape)

    #clf = ExtraTreesClassifier(max_depth=512, min_samples_split=8, min_samples_leaf=16)
    # 'max_depth': (None, 256, 512, 1024),
    # 'min_samples_split': (2, 4, 8, 16),
    # 'min_samples_leaf': (8, 16, 32, 64)

    clf = ExtraTreesClassifier(max_depth=None, min_samples_split=2, 
                                min_samples_leaf=8)

    clf.fit(X_train.toarray(), target_train)
    X_test_arr = X_test.toarray()
    target_perdicted = clf.predict(X_test_arr)
    print('Test Score: ' + str(clf.score(X_test_arr, target_test)))
    cm = metrics.confusion_matrix(target_test, target_perdicted)
    print("Confusion Matrix:")
    print(cm)

    plt.matshow(cm)
    plt.show()

    # metrics_out = '/home/gregor/ipyServer/movie_review/output/linsvm_metrics.out'
    # kaggle_test_out = '/home/gregor/ipyServer/movie_review/output/linsvm_kaggle_test_01.csv'

    # p1 = Process(target=metrics_est, args=(text_clf, dataset, metrics_out))
    # p1.start()

    # test_data = load_files(movie_reviews_test_data_folder, shuffle=False)
    # p2 = Process(target=train_full, args=(text_clf, dataset, test_data, kaggle_test_out))
    # p2.start()

    # p1.join()
    # p2.join()

    print('Test Compleated')