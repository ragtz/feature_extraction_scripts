#!/usr/bin/env python

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier as RFC
from copy import deepcopy
from keyframe_dataset import *
import numpy as np
import argparse
import pickle
import csv

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Grid search over classifier parameters')
    parser.add_argument('--data', metavar='PKL', required=True, help='Keyframe dataset file')
    parser.add_argument('--models', metavar='PKL', required=True, help='Saved models pkl file')
    parser.add_argument('--task', metavar='TSK', nargs='+', default=[], required=False, help='Task to train classifier')
    parser.add_argument('--test', metavar='TST', nargs='+', default=[], required=False, help='User or task left out for testing')
    parser.add_argument('--split', metavar='SLT', choices=['user', 'task'], default='user', required=False, help='Type of train/test split')

    args = parser.parse_args()
    data_file = args.data
    model_file = args.models
    task = args.task
    test = args.test
    split = args.split

    test_dataset = KeyframeDataset()
    test_dataset.load(data_file)

    if split == 'user':
        test_dataset = test_dataset.get_keyframe_dataset(pid=test, task=task)
        print 'Test Set:', test_dataset._get_pids(), test_dataset._get_tasks()
    else:
        test_dataset = test_dataset.get_keyframe_dataset(task=test) 
        print 'Test Set:', test_dataset._get_pids(), test_dataset._get_tasks()

    models = pickle.load(open(model_file))
    scaler = models['scaler_final']
    clf = models['clf_final']

    test_X, test_y = test_dataset._get_feature_label()

    test_X = scaler.transform(test_X)
    pred_y = clf.predict(test_X)

    precision, recall, _, _ = prfs(test_y, pred_y, average='binary')

    print 30*'-'
    print test[0], 'Model'
    print 30*'-'
    print 'Precision:', precision
    print 'Recall:', recall
 
if __name__ == '__main__':
    main()

