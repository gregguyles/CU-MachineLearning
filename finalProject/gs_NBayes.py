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
from sklearn.naive_bayes import MultinomialNB

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

    

    # Build vectorizer / classifier pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB())
        ])

    # Build Grid Search
    parameters = {
                    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'vect__max_df': (0.1, 0.30, 0.50, 0.70)
                }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(mov_train, target_train)

    # print cross validation values
    print("Grid Search Parameters:")
    print(parameters)
    
    print("\nBest Parameters:")
    best_parameters, score, _ = max(grid_search.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print('score: %f\n' % score)

    # predict the outcome on the traing test and store it in target_perdicted
    target_perdicted = grid_search.predict(mov_test)

    # print classification report
    print('Test Metrics Classification Report')
    print(metrics.classification_report(target_test, target_perdicted,
                target_names=dataset.target_names))

    # print and plot
    cm = metrics.confusion_matrix(target_test, target_perdicted)
    print("Confusion Matrix:")
    print(cm)

    if(print_plot == 'print_plot'):
        plt.matshow(cm)
        plt.show()
    else:
        print("\nAll Grid Scores:\n")
        print(grid_search.grid_scores_)

    print('#' * 70)
    print(('#' * 5) + (' ' * 25) + 'End    Run' + (' ' * 25) + ('#' * 5))
    print('#' * 70 + '\n\n')