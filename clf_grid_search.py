#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from copy import deepcopy
from keyframe_dataset import *
import numpy as np
import argparse
import pickle
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

              'rfc': [{'n_estimators': np.arange(10,110,10),
                       'criterion': ['gini', 'entropy'],
                       'max_depth': [2, 3, 4, 5, 10, 20, 50, 100, None],
                       'max_features': ['sqrt', 'log2', None],
                       'class_weight': [None, 'balanced']}]}

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
    parser.add_argument('--task', metavar='TSK', nargs='+', default=[], required=False, help='Task to train classifier')
    parser.add_argument('--clf', metavar='CLF', choices=['svc', 'rfc'], default='svc', required=False, help='Type of classifier')
    parser.add_argument('--split', metavar='SLT', choices=['user', 'task'], default='user', required=False, help='Type of train/test split')
    parser.add_argument('--results', metavar='CSV', required=True, help='Csv file to save results')
    parser.add_argument('--models', metavar='PKL', required=True, help='Pkl file to save best models')

    args = parser.parse_args()
    data_file = args.data
    task = args.task
    clf_type = args.clf
    split = args.split
    csv_file = args.results
    model_file = args.models

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()

    fieldnames = clf_fieldnames[clf_type] + ['train_acc', 'test_acc', 'true_pos', 'true_neg', 'false_pos', 'false_neg', 'precision', 'recall']

    best_models = {}
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

        curr_models = {}

        train_test_iter = dataset.iter_train_test(num_pids-1) if split == 'user' else dataset.task_iter_train_test(len(task)-1)

        for i, train_test in enumerate(train_test_iter):
            train, test = train_test

            train_pid, train = train
            train_X, train_y = train

            test_pid, test = test
            test_X, test_y = test

            scaler = StandardScaler()
            scaler.fit(train_X)

            train_X = scaler.transform(train_X)
            test_X = scaler.transform(test_X)

            weights = compute_sample_weight('balanced', train_y)

            clf.fit(train_X, train_y, sample_weight=weights)

            train_acc = clf.score(train_X, train_y)
            test_acc = clf.score(test_X, test_y)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            pos_idxs = np.where(test_y == 1)[0]
            neg_idxs = np.where(test_y == 0)[0]

            pred_labels = clf.predict(test_X)
 
            true_pos += np.sum(pred_labels[pos_idxs] == 1)
            true_neg += np.sum(pred_labels[neg_idxs] == 0)

            false_pos += np.sum(pred_labels[neg_idxs] == 1)
            false_neg += np.sum(pred_labels[pos_idxs] == 0)
            
            curr_models[test_pid[0]] = {'scaler': scaler, 'clf': clf, 'train_acc': train_acc, 'test_acc': test_acc}

        train_mean = np.mean(train_accs)
        test_mean = np.mean(test_accs)
        precision = true_pos/(true_pos + false_pos) if true_pos + false_pos > 0 else 'inf'
        recall = true_pos/(true_pos + false_neg) if true_pos + false_neg > 0 else 'inf'
        true_pos = int(true_pos)
        true_neg = int(true_neg)
        false_pos = int(false_pos)
        false_neg = int(false_neg)

        row = params_row[clf_type](params)
        row['train_acc'] = train_mean
        row['test_acc'] = test_mean
        row['true_pos'] = true_pos
        row['true_neg'] = true_neg
        row['false_pos'] = false_pos
        row['false_neg'] = false_neg
        row['precision'] = precision
        row['recall'] = recall
 
        if run == 0:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row) 
        else:
            with open(csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row) 
        
        if test_mean > max_test_acc:
            best_models = deepcopy(curr_models)
            best_models['train_mean'] = train_mean
            best_models['test_mean'] = test_mean
            pickle.dump(best_models, open(model_file, 'w'))

        max_test_acc = max([max_test_acc, test_mean])

        print 'Train Acc Mean:', train_mean
        print 'Test Acc Mean:', test_mean
        print 'Precision:', precision
        print 'Recall:', recall
        print '----------------------------------------'

    print '========================================'
    print 'Best Test Acc:', max_test_acc
    print '========================================'
 
if __name__ == '__main__':
    main()

