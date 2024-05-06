# Load all necessary packages
import sys
import numpy as np
import pandas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
sys.path.append("../")
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset,MEPSDataset19
from aif360.metrics import BinaryLabelDatasetMetric, SampleDistortionMetric
from sklearn.feature_selection import chi2
from aif360.metrics import ClassificationMetric
from sklearn import tree
import statistics
import json
from sklearn.neighbors import KNeighborsClassifier
import os
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer

import lib

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
import random

## import dataset

from collections import OrderedDict
from aif360.metrics import ClassificationMetric



def collectdata(datasetname,protectedattribute):
    from sklearn.decomposition import PCA
    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from filter import HEFRRanking

    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(datasetname, protectedattribute)
    trainsizeratio = 0.1
    label_name = dataset_orig.label_names[0]
    originalfeatureset = dataset_orig.feature_names

    for turn in np.arange(0, 50, 1):

        seedr = random.randint(0, 1000)
        print('================================================Turn:' + str(turn))

        dataset_orig_train_total, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seedr)
        dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seedr)
        dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
        dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_valid_pred_p = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_valid_pred_f = dataset_orig_valid.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split([trainsizeratio], shuffle=True,
                                                                              seed=seedr)

        featureset_index_p, featureset_p = HEFRRanking(datasetname, dataset_orig_train, protectedattribute, 0)

        featuresubset_p = []
        for feature in featureset_p[:3]:
            featuresubset_p.append(originalfeatureset.index(feature))
        scale_orig = StandardScaler()
        featuresubset_p = list(set(featuresubset_p))
        X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        X_train_p = X_train_fullfeature[:, featuresubset_p]

        tmod1 = tree.DecisionTreeClassifier(max_depth=10)
        tmod1.fit(X_train_p, y_train)
        fav_idx = np.where(tmod1.classes_ == dataset_orig_train_total.favorable_label)[0][0]
        y_train_pred_prob_p = tmod1.predict_proba(X_train_p)[:, fav_idx]

        X_valid_fullfeature = scale_orig.transform(dataset_orig_valid.features)
        X_valid_p = X_valid_fullfeature[:, featuresubset_p]
        y_valid_pred_prob_p = tmod1.predict_proba(X_valid_p)[:, fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob_p.reshape(-1, 1)
        dataset_orig_valid_pred_p.scores = y_valid_pred_prob_p.reshape(-1, 1)

        y_valid_pred_p = np.zeros_like(dataset_orig_valid_pred_p.labels)
        y_valid_pred_p[y_valid_pred_prob_p >= class_thresh] = dataset_orig_valid_pred_p.favorable_label
        y_valid_pred_p[~(y_valid_pred_prob_p >= class_thresh)] = dataset_orig_valid_pred_p.unfavorable_label
        dataset_orig_valid_pred_p.labels = y_valid_pred_p

        cm_transf_valid_p = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred_p,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

        featureset_index_f, featureset_f = HEFRRanking(datasetname, dataset_orig_train, protectedattribute, 0.1)

        featuresubset_f = []
        for feature in featureset_f[:3]:
            featuresubset_f.append(originalfeatureset.index(feature))
        scale_orig = StandardScaler()
        featuresubset_f = list(set(featuresubset_f))
        X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        X_train_f = X_train_fullfeature[:, featuresubset_f]

        tmod2 = tree.DecisionTreeClassifier(max_depth=10)
        tmod2.fit(X_train_f, y_train)
        fav_idx = np.where(tmod2.classes_ == dataset_orig_train_total.favorable_label)[0][0]
        y_train_pred_prob_f = tmod2.predict_proba(X_train_f)[:, fav_idx]

        X_valid_fullfeature = scale_orig.transform(dataset_orig_valid.features)
        X_valid_f = X_valid_fullfeature[:, featuresubset_f]
        y_valid_pred_prob_f = tmod2.predict_proba(X_valid_f)[:, fav_idx]

        class_thresh = 0.5
        dataset_orig_train_pred.scores = y_train_pred_prob_f.reshape(-1, 1)
        dataset_orig_valid_pred_f.scores = y_valid_pred_prob_f.reshape(-1, 1)

        y_valid_pred_f = np.zeros_like(dataset_orig_valid_pred_f.labels)
        y_valid_pred_f[y_valid_pred_prob_f >= class_thresh] = dataset_orig_valid_pred_f.favorable_label
        y_valid_pred_f[~(y_valid_pred_prob_f >= class_thresh)] = dataset_orig_valid_pred_f.unfavorable_label
        dataset_orig_valid_pred_f.labels = y_valid_pred_f

        cm_transf_valid_f = ClassificationMetric(dataset_orig_valid, dataset_orig_valid_pred_f,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

        score_p = 0
        score_f = 0
        if cm_transf_valid_f.accuracy() >= cm_transf_valid_p.accuracy():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.recall() >= cm_transf_valid_p.recall():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.precision() >= cm_transf_valid_p.precision():
            score_f += 1
        else:
            score_p += 1
        f1_score_p = (2 * cm_transf_valid_p.recall() * cm_transf_valid_p.precision()) / (
                cm_transf_valid_p.precision() + cm_transf_valid_p.recall())
        f1_score_f = (2 * cm_transf_valid_f.recall() * cm_transf_valid_f.precision()) / (
                cm_transf_valid_f.precision() + cm_transf_valid_f.recall())
        if f1_score_f >= f1_score_p:
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.false_positive_rate() <= cm_transf_valid_p.false_positive_rate():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.equal_opportunity_difference() <= cm_transf_valid_p.equal_opportunity_difference():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.statistical_parity_difference() <= cm_transf_valid_p.statistical_parity_difference():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.average_abs_odds_difference() <= cm_transf_valid_p.average_abs_odds_difference():
            score_f += 1
        else:
            score_p += 1
        if cm_transf_valid_f.disparate_impact() <= cm_transf_valid_p.disparate_impact():
            score_f += 1
        else:
            score_p += 1
        if score_p > score_f:
            featureset = featureset_p
            featureset_index = featureset_index_p
        else:
            featureset = featureset_f
            featureset_index = featureset_index_f

        np.savetxt(datasetname + '_' + protectedattribute + '/feature_list/run_' + str(turn) + '.csv', featureset_index)



def runall():
    datasetnamelist = [['adult', 'race'], ['adult', 'sex'], ['bank', 'age'], ['german', 'age'], ['german', 'sex'], ['compas', 'sex'], ['compas', 'race'], ['meps', 'RACE'], ['default', 'sex'], ['home', 'sex']]
    for i in datasetnamelist:
        collectdata(i[0], i[1])
    # from statsmodels.stats.multitest import multipletests
    # When the Wilcoxon rank sum test p-value of the same indicator in ten scenarios is obtained, we use multipletests(using the method "fdr_by") to make adjustments to the p-value


if __name__ == '__main__':
    runall()
