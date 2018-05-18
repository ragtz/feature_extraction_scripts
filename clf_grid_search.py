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

'''
{'kernel': ['poly'],
'C': np.arange(1,11,.5),
'degree': [2, 3, 4],
'gamma': ['auto', .0001, .001, .01, .1, 1, 10, 100],
'coef0': [0, .1, 1, 10, -.1, -1, 10],
'class_weight': [None, 'balanced']},
'''

clf_params = {'svc': [{'kernel': ['linear'],
                       'C': np.arange(1,11,.5),
                       'class_weight': [None, 'balanced']},

                      {'kernel': ['rbf'],
                       'C': np.arange(1,11,.5),
                       'gamma': ['auto', .0001, .001, .01, .1, 1, 10, 100],
                       'class_weight': [None, 'balanced']}],

              'dtc': [{},
                      {}],

              'rfc': [{},
                      {}]}

def main():
    parser = argparse.ArgumentParser(description='Grid search over classifier parameters')
    parser.add_argument('--data', metavar='PKL', required=True, help='Keyframe dataset file')
    parser.add_argument('--task', metavar='TSK', choices=['drawer', 'lamp', 'pitcher', 'bowl', 'all'], default='all', required=False, help='Task to train classifier')
    parser.add_argument('--clf', metavar='CLF', choices=['svc', 'dtc', 'rfc'], deafult='svc', required=False, help='Type of classifier')
    parser.add_argument('--output', metavar='CSV', required=True, help='Csv file to save results')

    args = parser.parse_args()
    data_file = args.data
    task = args.task
    clf_type = args.clf
    csv_file = args.output

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()

    for run, params in enumerate(ParameterGrid(clf_params[clf_type])):
        if clf_type == 'svc':
            clf = SVC(**params)
        elif clf_type == 'dtc':
            clf = DTC(**params)
        else:
            clf = RFC(**params)

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

        if run == 0:
            with open(csv_file, 'w') as csvfile:
                fieldnames = ['kernel', 'C', 'gamma', 'class_weight',
                              'train_acc', 'test_acc', 'true_pos',
                              'true_neg', 'false_pos', 'false_neg',
                              'precision', 'recall']

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                row = {'kernel': params['kernel'],
                       'C': params['C'],
                       'class_weight': params['class_weight']}
                if 'gamma' in params:
                    row['gamma'] = params['gamma']
                else:
                    row['gamma'] = '-'

                row['train_acc'] = np.mean(train_accs)
                row['test_acc'] = np.mean(test_accs)
                row['true_pos'] = int(true_pos)
                row['true_neg'] = int(true_neg)
                row['false_pos'] = int(false_pos)
                row['false_neg'] = int(false_neg)
                row['precision'] = true_pos/(true_pos + false_pos)
                row['recall'] = true_pos/(true_pos + false_neg)

                writer.writeheader()
                writer.writerow(row) 
        else:
            with open(csv_file, 'a') as csvfile:
                writer = csv.DictWriter(csvfile)

                row = {'kernel': params['kernel'],
                       'C': params['C'],
                       'class_weight': params['class_weight']}
                if 'gamma' in params:
                    row['gamma'] = params['gamma']
                else:
                    row['gamma'] = '-'

                row['train_acc'] = np.mean(train_accs)
                row['test_acc'] = np.mean(test_accs)
                row['true_pos'] = int(true_pos)
                row['true_neg'] = int(true_neg)
                row['false_pos'] = int(false_pos)
                row['false_neg'] = int(false_neg)
                row['precision'] = true_pos/(true_pos + false_pos)
                row['recall'] = true_pos/(true_pos + false_neg)
                writer.writerow(row) 

if __name__ == '__main__':
    main()

