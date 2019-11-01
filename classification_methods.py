import numpy as np
import pickle
import pandas as pd
from scipy.stats import variation
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

pd.set_option('display.max_columns', None)  # Show all rows and columns
pd.set_option('display.max_rows', None)


def cov_and_abundance_filter(df, cov_limit=0):
    """Remove columns with mean zero and desired CoV, and species that are present only once

    :param df: Pandas DataFrame of domain count with species per row and domains as columns
    :param cov_limit: Coefficient of Variation limit, integer. CoV = sd / mean.
        High CoV means high sd and/or low mean. High CoV aims to filter out household genes
    :return:
    """
    print(f"Data has {df.shape[0]} species (rows) and {df.shape[1]} domains (columns) \n")
    df_dropped = df.copy(deep=True)  # So we can change that one to our needs

    print(f'\n__________Filtering data__________')
    print('Dropping the species that are present only once...\n')

    t0 = time.time()
    only_once = df_dropped.index.value_counts() == 1
    df_dropped.drop(df_dropped[only_once].index, inplace=True)
    t1 = time.time()
    print(f'Done, {len(only_once)} domains removed ({(t1 - t0):.1f} sec)\n')

    print(f"Removing columns with only zeroes...")
    t0 = time.time()
    col_with_mean_zero = df_dropped.columns[df_dropped.sum(axis=0) == 0]
    df_dropped.drop(col_with_mean_zero, axis=1, inplace=True)
    t1 = time.time()
    print(f'Done, {len(col_with_mean_zero)} domains removed ({(t1 - t0):.1f} sec)\n')

    print(f"Removing domains with CoV <= {cov_limit}...")
    t0 = time.time()
    col_with_cov_x = df_dropped.columns[variation(df_dropped, axis=0) <= cov_limit]
    df_dropped.drop(col_with_cov_x, axis=1, inplace=True)
    t1 = time.time()
    print(f'Done, {len(col_with_cov_x)} domains removed ({(t1 - t0):.1f} sec)\n')
    print(f"Filtered data has {df_dropped.shape[0]} species (rows) and {df_dropped.shape[1]} "
          f"domains (columns) left with CoV > {cov_limit}\n")
    return df_dropped


def perform_kNN(df_dropped, iterations=1, prepro=False):
    print(f'__________Performing kNN regression__________')
    # Train & Test data
    # Define features X (domain count)
    X = df_dropped.values  # axis for columns
    if prepro:
        # Preprocess x in order to scale it so it improves the model or something
        print("Preprocessing data...")
        t0 = time.time()
        X = preprocessing.scale(X)
        t1 = time.time()
        print(f'Done ({(t1 - t0):.1f} sec)\n')
    # Define classifier (species name = index)
    y = list(df_dropped.index.values)

    # kNN
    t0 = time.time()
    scores = []
    for i in range(iterations):
        if i in range(0, iterations, 10):
            print(f"Doing iteration {i}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # NO random seed because else iterations will be the same

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X_train, y_train)  # = the training, making a fit line
        scores.append(clf.score(X_test, y_test))  # see how good did the model do, on the test data = R2
    t1 = time.time()
    running_time = t1 - t0
    print(f'Done ({running_time:.1f} sec)\n')

    print("Performance:")
    print(f"Min = {min(scores):.2f}")
    print(f"Mean = {np.mean(scores):.2f}")
    print(f"Max = {max(scores):.2f}\n")
    return scores, running_time


def perform_logreg(df_dropped, iterations=1, prepro=False):
    print(f'__________Performing logistic regression__________')
    # Train & Test data
    # Define features X (domain count)
    X = df_dropped.values  # axis for columns
    if prepro:
        # Preprocess x in order to scale it so it improves the model or something
        print("Preprocessing data...")
        t0 = time.time()
        X = preprocessing.scale(X)
        t1 = time.time()
        print(f'Done ({(t1 - t0):.1f} sec)\n')
    # Define classifier (species name = index)
    y = list(df_dropped.index.values)

    # Logistic regression
    t0 = time.time()
    scores = []

    for i in range(iterations):
        if i in range(0, iterations, 10):
            print(f"Doing iteration {i}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # NO random seed because else iterations will be the same

        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10e5)  # solver='saga'
        clf.fit(X_train, y_train)  # = the training, making a fit line
        scores.append(clf.score(X_test, y_test))  # see how good did the model do, on the test data = R2
    t1 = time.time()
    running_time = t1 - t0
    print(f'Done ({running_time:.1f} sec)\n')

    print("Performance:")
    print(f"Min = {min(scores):.2f}")
    print(f"Mean = {np.mean(scores):.2f}")
    print(f"Max = {max(scores):.2f}\n")
    return scores, running_time


def perform_lasso_logreg(df_dropped, iterations=1, prepro=False):
    print(f'__________Performing lasso logistic regression__________')
    # Train & Test data
    # Define features X (domain count)
    X = df_dropped.values  # axis for columns
    if prepro:
        # Preprocess x in order to scale it so it improves the model or something
        print("Preprocessing data...")
        t0 = time.time()
        X = preprocessing.scale(X)
        t1 = time.time()
        print(f'Done ({(t1 - t0):.1f} sec)\n')
    # Define classifier (species name = index)
    y = list(df_dropped.index.values)

    # Lasso logistic regression
    t0 = time.time()
    scores = []
    for i in range(iterations):
        if i in range(0, iterations, 10):
            print(f"Doing iteration {i}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # NO random seed because else iterations will be the same
        clf = LogisticRegression(penalty='l1', solver='saga', multi_class='auto', max_iter=1000) #saga
        clf.fit(X_train, y_train)  # = the training, making a fit line
        scores.append(clf.score(X_test, y_test))  # see how good did the model do, on the test data = R2
    t1 = time.time()
    running_time = t1 - t0
    print(f'Done ({running_time:.1f} sec)\n')

    print("Performance:")
    print(f"Min = {min(scores):.2f}")
    print(f"Mean = {np.mean(scores):.2f}")
    print(f"Max = {max(scores):.2f}\n")
    return scores, running_time


def perform_SVM(df_dropped, iterations=1, prepro=False):
    print(f'__________Performing SVM__________')
    # Train & Test data
    # Define features X (domain count)
    X = df_dropped.values  # axis for columns
    if prepro:
        # Preprocess x in order to scale it so it improves the model or something
        print("Preprocessing data...")
        t0 = time.time()
        X = preprocessing.scale(X)
        t1 = time.time()
        print(f'Done ({(t1 - t0):.1f} sec)\n')
    # Define classifier (species name = index)
    y = list(df_dropped.index.values)

    # SVM Linear kernel
    t0 = time.time()
    scores = []
    clf = svm.SVC(kernel="linear", C=100, gamma=0.0001)

    for i in range(iterations):
        if i in range(0, iterations, 10):
            print(f"Doing iteration {i}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) #NO random seed because else iterations will be the same
        clf.fit(X_train, y_train)  # = the training, making a fit line
        scores.append(clf.score(X_test, y_test))  # see how good did the model do, on the test data = R2
    t1 = time.time()
    running_time = t1 - t0
    print(f'Done ({running_time:.1f} sec)\n')

    print("Performance:")
    print(f"Min = {min(scores):.2f}")
    print(f"Mean = {np.mean(scores):.2f}")
    print(f"Max = {max(scores):.2f}\n")
    return scores, running_time


def perform_rdf(df_dropped, iterations=1, prepro=False):
    print(f'__________Performing Random Forest__________')
    # Train & Test data
    # Define features X (domain count)
    X = df_dropped.values  # axis for columns
    if prepro:
        # Preprocess x in order to scale it so it improves the model or something
        print("Preprocessing data...")
        t0 = time.time()
        X = preprocessing.scale(X)
        t1 = time.time()
        print(f'Done ({(t1 - t0):.1f} sec)\n')
    # Define classifier (species name = index)
    y = list(df_dropped.index.values)

    # Random Forest
    t0 = time.time()
    scores = []
    for i in range(iterations):
        if i in range(0, iterations, 10):
            print(f"Doing iteration {i}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25) #NO random seed because else iterations will be the same
        clf = RandomForestClassifier(n_estimators=100)  # Else FutureWarning default value changes in new version of function
        clf.fit(X_train, y_train)  # = the training, making a fit line
        scores.append(clf.score(X_test, y_test))  # see how good did the model do, on the test data = R2
    t1 = time.time()
    running_time = t1 - t0
    print(f'Done ({running_time:.1f} sec)\n')

    print("Performance:")
    print(f"Min = {min(scores):.2f}")
    print(f"Mean = {np.mean(scores):.2f}")
    print(f"Max = {max(scores):.2f}\n")
    return scores, running_time


def save_performance(data, genus=False, iterations=1):
    if genus:
        # using only genus, not species.
        data.index = data.index.str.split().str[0]
    prepro = [True, False]
    cov_limit_list = np.arange(20, -2.5, -2.5)
    # cov_limit_list = np.array([20, 17.5])
    results_df = pd.DataFrame()
    # model_list = ['svm', 'rdf', 'kNN']
    model_list = ['logreg']

    for model in model_list:
        print(f'\n=============== Model: {model} ===============\n')
        for state in prepro:
            for cov_limit in cov_limit_list:
                print(f"__________{model} CoV limit {cov_limit}, prepro={state}__________")
                df_dropped = cov_and_abundance_filter(data, cov_limit=cov_limit)
                if model == 'svm':
                    scores, running_time = perform_SVM(df_dropped, iterations=iterations, prepro=state)
                elif model == 'rdf':
                    scores, running_time = perform_rdf(df_dropped, iterations=iterations, prepro=state)
                elif model == 'logreg':
                    scores, running_time = perform_logreg(df_dropped, iterations=iterations, prepro=state)
                elif model == 'kNN':
                    scores, running_time = perform_kNN(df_dropped, iterations=iterations, prepro=state)
                else:  # model == 'lasso':
                    scores, running_time = perform_lasso_logreg(df_dropped, iterations=iterations, prepro=state)
                results_df[f'cov_limit_{cov_limit}_{state}'] = scores + [running_time]

        print(f'Making output file CoVscores_{model}_genus{str(genus)}.csv...')
        results_df.to_csv(f'CoVscores_{model}_genus{str(genus)}.csv', index=True, sep=',')

        print(f'Model {model} done\n')
    print('All done')


# # Raw data column count != 0 (= how many domains)
# df.gt(0).sum(axis=1)
# # Mean domain count
# domain_count = df[df.gt(0)].mean(axis=1)
#
# #covs per species
# covs = variation(df_dropped)
# with open('your_file.txt3', 'w') as f:
#     for item in covs:
#         f.write("%s\n" % item)



if __name__ == "__main__":
    limit = 0
    pickle_in = open(f"/home/kewin/PycharmProjects/Capita_Selecta/all_samples_info_df{limit}.pickle", 'rb')
    df = pickle.load(pickle_in)
    df_dropped = cov_and_abundance_filter(df, cov_limit=0)
    save_performance(data=df, genus=True, iterations=100)
    save_performance(data=df, genus=False, iterations=100)
