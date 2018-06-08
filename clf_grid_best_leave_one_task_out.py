#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from keyframe_dataset import *
import numpy as np
import argparse
import csv

clf_params = {'svc': [{'kernel': ['linear'],
                       'C': np.arange(1,11,.5),
                       'class_weight': [None, 'balanced']},

                      {'kernel': ['rbf'],
                       'C': np.arange(1,11,.5),
                       'gamma': ['auto', .0001, .001, .01, .1, 1, 10, 100],
                       'class_weight': [None, 'balanced']},

                      {'kernel': ['poly'],
                       'C': np.arange(1,11,.5),
                       'degree': [2, 3, 4],
                       'gamma': ['auto', .0001, .001, .01, .1, 1, 10, 100],
                       'coef0': [0, .1, 1, 10, -.1, -1, 10],
                       'class_weight': [None, 'balanced']}],

              'rfc': [{'n_estimators': [50,70],
                       'criterion': ['entropy'],
                       'max_depth': [4,5],
                       'max_features': [None],
                       'class_weight': [None,'balanced']}]}

clf_fieldnames = {'svc': ['kernel', 'C', 'gamma', 'class_weight', 'degree', 'coef0',
                          'train_acc', 'test_acc', 'true_pos', 'true_neg', 'false_pos',
                          'false_neg', 'precision', 'recall'],

                  'rfc': ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                          'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
                          'max_leaf_nodes', 'min_impurity_descrease', 'min_impurity_split',
                          'bootstrap', 'oob_score', 'class_weight']}

def svc_params_row(params):
    row = {'kernel': params['kernel'],
           'C': params['C']}

    if 'gamma' in params:
        row['gamma'] = params['gamma']
    else:
        row['gamma'] = '-'

    if not params['class_weight'] is None:
        row['class_weight'] = params['class_weight']
    else:
        row['class_weight'] = 'none'

    if 'degree' in params:
        row['degree'] = params['degree']
    else:
        row['degree'] = '-'

    if 'coef0' in params:
        row['coef0'] = params['coef0']
    else:
        row['coef0'] = '-'

    return row

def rfc_params_row(params):
    row = {k: params[k] for k in ['n_estimators', 'criterion']}

    for k in ['max_features', 'max_depth', 'class_weight']:
        if not params[k] is None:
            row[k] = params[k]
        else:
            row[k] = 'none'

    return row

params_row = {'svc': svc_params_row, 'rfc': rfc_params_row}

def main():
    parser = argparse.ArgumentParser(description='Grid search over classifier parameters')
    parser.add_argument('--data', metavar='PKL', required=True, help='Keyframe dataset file')

    args = parser.parse_args()
    data_file = args.data
    task = ['drawer', 'lamp', 'pitcher', 'bowl']
    clf_type = 'rfc'
    split = 'task'

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()

    fieldnames = clf_fieldnames[clf_type] + ['train_acc', 'test_acc', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'precision', 'recall']

    max_test_acc = -float('inf')
    n = len(ParameterGrid(clf_params[clf_type]))
    for run, params in enumerate(ParameterGrid(clf_params[clf_type])):
        print '----------------------------------------'
        print 'Training and testing classifier ' + str(run+1) + ' of ' + str(n) + '...'

        if clf_type == 'svc':
            clf = SVC(**params)
        else:
            clf = RFC(**params)

        train_accs = []
        test_accs = []

        true_pos = 0.
        true_neg = 0.
        false_pos = 0.
        false_neg = 0.

        train_test_iter = dataset.iter_train_test(num_pids-1) if split == 'user' else dataset.task_iter_train_test(len(task)-1)

        for i, train_test in enumerate(train_test_iter):
            train, test = train_test

            train_pid, train = train
            train_X, train_y = train

            test_pid, test = test
            test_X, test_y = test

            print 'Train:', train_pid
            print 'Test:', test_pid

            scaler = StandardScaler()
            scaler.fit(train_X)

            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)

            weights = compute_sample_weight('balanced', train_y)

            clf.fit(train_X, train_y, sample_weight=weights)

            train_accs.append(clf.score(train_X, train_y))
            test_accs.append(clf.score(test_X, test_y))

            pos_idxs = np.where(test_y == 1)[0]
            neg_idxs = np.where(test_y == 0)[0]

            pred_labels = clf.predict(test_X)
     
            tp = 1.*np.sum(pred_labels[pos_idxs] == 1)
            tn = 1.*np.sum(pred_labels[neg_idxs] == 0)

            true_pos += tp
            true_neg += tn

            fp = 1.*np.sum(pred_labels[neg_idxs] == 1)
            fn = 1.*np.sum(pred_labels[pos_idxs] == 0)

            false_pos += fp
            false_neg += fn

            print 'Train Acc:', train_accs[-1]
            print 'Test Acc:', test_accs[-1]
            print 'TP:', int(tp)
            print 'TN:', int(tn)
            print 'FP:', int(fp)
            print 'FN:', int(fn)
            print 'Precision:', tp/(tp + fp) if tp + fp > 0 else 'inf'
            print 'Recall:', tp/(tp + fn) if tp + fn > 0 else 'inf'
            print '----------------------------------------'

        train_mean = np.mean(train_accs)
        test_mean = np.mean(test_accs)
        precision = true_pos/(true_pos + false_pos) if true_pos + false_pos > 0 else 'inf'
        recall = true_pos/(true_pos + false_neg) if true_pos + false_neg > 0 else 'inf'
        true_pos = int(true_pos)
        true_neg = int(true_neg)
        false_pos = int(false_pos)
        false_neg = int(false_neg)
 
        max_test_acc = max([max_test_acc, test_mean])

        print 'Train Acc Mean:', train_mean
        print 'Test Acc Mean:', test_mean
        print 'TP:', true_pos
        print 'TN:', true_neg
        print 'FP:', false_pos
        print 'FN:', false_neg
        print 'Precision:', precision
        print 'Recall:', recall
        print '----------------------------------------'

    print '========================================'
    print 'Best Test Acc:', max_test_acc
    print '========================================'
 
if __name__ == '__main__':
    main()

