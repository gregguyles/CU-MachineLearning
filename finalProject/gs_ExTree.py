# Greg Guyles
# Final Project: Text Classification
# Machine Learning
# CU Boulder
# 04/28/2014

import sys
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
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2

if __name__ == "__main__":

    print('*' * 70)
    print(('*' * 5) + (' ' * 25) + 'Begin  Run' + (' ' * 25) + ('*' * 5))
    print('*' * 70)

    # import data
    subset = sys.argv[1]
    print_plot = sys.argv[2]
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

    # split data into train and test sets
    mov_train, mov_test, target_train, target_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.1,
                                 ngram_range=(1,2))
    #'vect__ngram_range': [(1, 1), (1, 2)],
    #'vect__max_df': (0.5, 0.7, 1.0)
    X_train = vectorizer.fit_transform(mov_train)
    X_test = vectorizer.transform(mov_test)

    print(X_train.get_shape)
    from sklearn.feature_selection import SelectKBest, chi2
    #ch2 = SelectKBest(chi2, k='all')
    #ch2 = SelectKBest(chi2, k=(20 * 1000))
    ch2 = SelectKBest(chi2, k=(80 * 1000))
    X_train = ch2.fit_transform(X_train, target_train)
    X_test = ch2.transform(X_test)

    print(X_train.get_shape)

    # Build Grid Search
    parameters = {
                    'max_depth': (1024, 2048, 4096),
                    'min_samples_split': (16, 32, 64),
                    'min_samples_leaf': (2, 4, 8)
                }

    # predict the outcome on the traing test and store it in target_perdicted
    grid_search = GridSearchCV(ExtraTreesClassifier(max_depth=1024, min_samples_split=16
                        min_samples_leaf=8), parameters, n_jobs=1)
    grid_search.fit(X_train.toarray(), target_train)

    # print cross validation values
    print("Grid Search Parameters:")
    print(parameters)
    
    print("\nBest Parameters:")
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print('score: %f\n' % score)

    # X_test_counts = count_vect.transform(mov_test)
    # X_test_tf = tf_transformer.transform(X_test_counts)
    #target_perdicted = clf.predict(X_test_tf.toarray())
    predict_score = grid_search.score(X_test.toarray(), target_test)
    target_perdicted = grid_search.predict(X_test.toarray())
    # print classification report

    print('Test Score: ' + str(predict_score))
    print('Test Metrics Classification Report')
    print(metrics.classification_report(target_test, target_perdicted,
                target_names=dataset.target_names))

    # print and plot
    cm = metrics.confusion_matrix(target_test, target_perdicted)
    print("Confusion Matrix:")
    print(cm)

    if(print_plot == '-print_plot'):
        plt.matshow(cm)
        plt.show()
    else:
        print("\nAll Grid Scores:\n")
        print(grid_search.grid_scores_)

    print('#' * 70)
    print(('#' * 5) + (' ' * 25) + 'End    Run' + (' ' * 25) + ('#' * 5))
    print('#' * 70 + '\n\n')