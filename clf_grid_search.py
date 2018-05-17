#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keyframe_dataset import *
import numpy as np
import itertools
import argparse
import csv

clfs = {'svc': SVC(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier()}

clf_params = {'svc': {'kernel': ['linear', 'rbf', 'poly'],
                      'C': [],
                      'degree': [],
                      'gamma': [],
                      'coef0': [],
                      'class_weight': [None, 'balanced']}}

def grid_enumerate(params):
    for p in itertools.product(*[params[k] for k in params])
        yield {k: p[i] for i, k in enumerate(params)}

def main():
    parser = argparse.ArgumentParser(description='Grid search over classifier parameters')
    parser.add_argument('--data', metavar='PKL', required=True, help='Keyframe dataset file')
    parser.add_argument('--task', metavar='TSK', choices=['drawer', 'lamp', 'pitcher', 'bowl', 'all'], default='all', required=False, help='Task to train classifier')
    parser.add_argument('--clf', metavar='CLF', choices=['svc'], deafult='svc', required=False, help='Type of classifier')
    parser.add_argument('--output', metavar='CSV', required=True, help='Csv file to save results')

    args = parser.parse_args()
    data_file = args.data
    task = args.task
    csv_file = args.output

    clf = clfs[args.clf]
    params = clf_params[args.clf]

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()

    for p in grid_enumerate(params):
        pid_rm = []
        train_accs = []
        test_accs = []

        true_pos = 0.
        true_neg = 0.
        false_pos = 0.
        false_neg = 0.

        for i, train_test in enumerate(dataset.iter_train_test(num_pids-1)):
            train, test = train_test

            train_pid, train = train
            train_X, train_y = train

            test_pid, test = test
            test_X, test_y = test

            pid_rm.extend(test_pid)

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
     
            true_pos += np.sum(pred_labels[pos_idxs] == 1)
            true_neg += np.sum(pred_labels[neg_idxs] == 0)

            false_pos += np.sum(pred_labels[neg_idxs] == 1)
            false_neg += np.sum(pred_labels[pos_idxs] == 0)

        print 'PID RM:', pid_rm
        print 'Train Accs:', train_accs
        print 'Test Accs:', test_accs
        print 'Train Acc Avg:', np.mean(train_accs)
        print 'Test Acc Avg:', np.mean(test_accs)
        print 'TP:', int(true_pos)
        print 'TN:', int(true_neg)
        print 'FP:', int(false_pos)
        print 'FN:', int(false_neg)
        print 'Precision:', true_pos/(true_pos + false_pos)
        print 'Recall:', true_pos/(true_pos + false_neg)

if __name__ == '__main__':
    main()

