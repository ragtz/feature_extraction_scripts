#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier as RFC
from keyframe_dataset import *
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Saved classifier model checker')
    parser.add_argument('--data', metavar='PKL', required=True, help='Keyframe dataset file')
    parser.add_argument('--models', metavar='PKL', required=True, help='Saved models file')
    parser.add_argument('--task', metavar='TSK', nargs='+', default=[], required=False, help='Tasks classifiers were trained on')
    parser.add_argument('--split', metavar='SLT', choices=['user', 'task'], default='user', required=False, help='Type of train/test split')

    args = parser.parse_args()
    data_file = args.data
    model_file = args.models
    task = args.task
    split = args.split

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()
    num_tasks = len(task)

    models = pickle.load(open(model_file))

    train_test_iter = dataset.iter_train_test(num_pids-1) if split == 'user' else dataset.task_iter_train_test(num_tasks-1)

    for i, train_test in enumerate(train_test_iter):
        train, test = train_test

        train_set, train = train
        train_X, train_y = train

        test_set, test = test
        test_X, test_y = test

        model = models[test_set[0]]
        scaler = model['scaler']
        clf = model['clf']

        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        train_acc = clf.score(train_X, train_y)
        test_acc = clf.score(test_X, test_y)

        print '---------- Model ' + str(i) + '----------'
        if model['train_acc'] == train_acc and model['test_acc'] == test_acc:
            print 'Model properly saved.'
        else:
            print 'Model incorrectly saved.'
            print '    - Saved Train Accuracy:', model['train_acc']
            print '    - Train Accuracy:', train_acc
            print
            print '    - Saved Test Accuracy:', model['test_acc']
            print '    - Test Accuracy:', test_acc
 
if __name__ == '__main__':
    main()

