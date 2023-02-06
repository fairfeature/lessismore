from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset,MEPSDataset19
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import aif360
from skfeature.function.similarity_based import fisher_score
import numpy as np
import heapq
import copy
from sklearn import tree


def FRFE(X, y, y1, alpha, estimator1, estimator2, step_score=None):
    #from sklearn.feature_selection._base import _get_feature_importances
    from sklearn.base import clone
    # Parameter step_score controls the calculation of self.scores_
    # step_score is not exposed to users
    # and is used when implementing RFECV
    # self.scores_ will not be calculated when calling _fit through fit

    # Initialization
    n_features = X.shape[1]

    n_features_to_select = 1

    step = 1

    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)

    if step_score:
        scores_ = []

    # Elimination
    while np.sum(support_) > n_features_to_select:
        # Remaining features
        features = np.arange(n_features)[support_]

        # Rank the remaining features
        estimator1 = clone(estimator1)

        estimator2 = clone(estimator2)

        estimator1.fit(X[:, features], y)

        estimator2.fit(X[:, features], y1)

        # Get importance and rank them
        importances = estimator1.feature_importances_

        importances1 = estimator2.feature_importances_

        for i in range(len(importances)):
            importances[i] = importances[i] - alpha * importances1[i]

        ranks = np.argsort(importances)

        # for sparse case ranks is matrix
        ranks = np.ravel(ranks)

        # Eliminate the worse features
        threshold = min(step, np.sum(support_) - n_features_to_select)

        # Compute step score on the previous selection iteration
        # because 'estimator' must use features
        # that have not been eliminated yet
        if step_score:
            scores_.append(step_score(estimator, features))
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1

    # Set final attributes

    n_features_ = support_.sum()
    support_ = support_
    ranking_ = ranking_

    return ranking_


def HEFRRanking(datasetname, dataset_orig_train, protected_feature, alpha):
    from sklearn.feature_selection import RFE

    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], \
                           dataset_orig_train[
                               'income-per-year']
    if datasetname == 'compas':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'two_year_recid'], \
                           dataset_orig_train['two_year_recid']
    if datasetname == 'bank':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'y'], \
                           dataset_orig_train['y']
    if datasetname == 'german':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'credit'], \
                           dataset_orig_train['credit']
    if datasetname == 'meps':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'UTILIZATION'], \
                           dataset_orig_train['UTILIZATION']

    if datasetname == 'default':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']

    if datasetname == 'home':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'TARGET'], \
                           dataset_orig_train['TARGET']

    y_pro = X_train[protected_feature]
    X_train = X_train.drop(protected_feature, axis=1)
    column_train = [column for column in X_train]
    X_train = np.array(X_train)
    y_pro = np.array(y_pro)
    y_train = np.array(y_train)
    estimator = tree.DecisionTreeClassifier(max_depth=10, random_state=0)
    ranking_list = FRFE(X_train, y_train, y_pro, alpha, estimator1=estimator, estimator2=estimator)
    choose_list = [''] * len(column_train)
    for i in range(len(ranking_list)):
        choose_list[ranking_list[i]-1] = column_train[i]

    return choose_list

def ChiRanking(datasetname, dataset_orig_train, protected_feature):
    from skfeature.function.statistical_based import chi_square
    from sklearn.preprocessing import MinMaxScaler

    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], \
                           dataset_orig_train[
                               'income-per-year']
    if datasetname == 'compas':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'two_year_recid'], \
                           dataset_orig_train['two_year_recid']
    if datasetname == 'bank':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'y'], \
                           dataset_orig_train['y']
    if datasetname == 'german':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'credit'], \
                           dataset_orig_train['credit']
    if datasetname == 'meps':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'UTILIZATION'], \
                           dataset_orig_train['UTILIZATION']

    if datasetname == 'default':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']

    if datasetname == 'home':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'TARGET'], \
                           dataset_orig_train['TARGET']

    X_train = X_train.drop(protected_feature, axis=1)
    column_train = [column for column in X_train]
    X_train = np.array(X_train)
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    y_train = np.array(y_train)
    ranking_list = chi_square.chi_square(X_train, y_train)
    ranking_list = chi_square.feature_ranking(ranking_list)
    choose_list = []
    for i in ranking_list:
        choose_list.append(column_train[i])
    return choose_list

def ReliefRanking(datasetname, dataset_orig_train, protected_feature):
    from skfeature.function.similarity_based import reliefF
    from sklearn.preprocessing import MinMaxScaler

    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], \
                           dataset_orig_train[
                               'income-per-year']
    if datasetname == 'compas':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'two_year_recid'], \
                           dataset_orig_train['two_year_recid']
    if datasetname == 'bank':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'y'], \
                           dataset_orig_train['y']
    if datasetname == 'german':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'credit'], \
                           dataset_orig_train['credit']
    if datasetname == 'meps':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'UTILIZATION'], \
                           dataset_orig_train['UTILIZATION']

    if datasetname == 'default':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']

    if datasetname == 'home':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'TARGET'], \
                           dataset_orig_train['TARGET']

    X_train = X_train.drop(protected_feature, axis=1)
    column_train = [column for column in X_train]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    ranking_list = reliefF.reliefF(X_train, y_train)
    ranking_list = reliefF.feature_ranking(ranking_list)
    choose_list = []
    for i in ranking_list:
        choose_list.append(column_train[i])
    return choose_list

def SFMRanking(datasetname, dataset_orig_train, protected_feature, clf):
    from sklearn.feature_selection import SelectFromModel

    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], \
                           dataset_orig_train[
                               'income-per-year']
    if datasetname == 'compas':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'two_year_recid'], \
                           dataset_orig_train['two_year_recid']
    if datasetname == 'bank':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'y'], \
                           dataset_orig_train['y']
    if datasetname == 'german':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'credit'], \
                           dataset_orig_train['credit']
    if datasetname == 'meps':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'UTILIZATION'], \
                           dataset_orig_train['UTILIZATION']

    if datasetname == 'default':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], \
                           dataset_orig_train['Probability']

    if datasetname == 'home':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'TARGET'], \
                           dataset_orig_train['TARGET']

    X_train = X_train.drop(protected_feature, axis=1)
    column_train = [column for column in X_train]
    X_train = np.array(X_train)
    # mm = MinMaxScaler()
    # X_train = mm.fit_transform(X_train)
    y_train = np.array(y_train)

    selector = SelectFromModel(estimator=clf).fit(X_train, y_train)
    try:
        ranking_list = np.argsort(selector.estimator_.coef_)[::-1]
    except:
        ranking_list = np.argsort(selector.estimator_.feature_importances_)[::-1]
    choose_list = []
    for i in ranking_list:
        choose_list.append(column_train[i])
    print(choose_list)
    return choose_list

