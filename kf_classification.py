#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keyframe_dataset import *
import numpy as np
import pickle
import sys

clfs = {'svc': SVC(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier()}

feature_names = {'drawer': ['table_drawer_dist', 'gripper_drawer_dist', 'gripper_state',
                            'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                            'drawer_hist_0', 'drawer_hist_1', 'drawer_hist_2', 'drawer_color_0',
                            'drawer_color_1', 'drawer_color_2', 'drawer_volume'],
                 'lamp': ['table_lamp_dist', 'gripper_lamp_dist', 'gripper_state', 'force_x',
                          'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                          'lamp_hist_0', 'lamp_hist_1', 'lamp_hist_2', 'lamp_color_0',
                          'lamp_color_1', 'lamp_color_2', 'lamp_volume'],
                 'pitcher': ['bowl_pitcher_dist', 'table_bowl_dist', 'table_pitcher_dist',
                             'gripper_bowl_dist', 'gripper_pitcher_dist', 'gripper_state',
                             'force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                             'pitcher_hist_0', 'pitcher_hist_1', 'pitcher_hist_2', 'bowl_hist_0',
                             'bowl_hist_1', 'bowl_hist_2', 'bowl_color_0', 'bowl_color_1',
                             'bowl_color_2', 'pitcher_color_0', 'pitcher_color_1',
                             'pitcher_color_2', 'bowl_volume', 'pitcher_volume'],
                 'bowl': ['large_bowl_spoon_dist', 'large_bowl_small_bowl_dist',
                          'spoon_small_bowl_dist', 'table_large_bowl_dist',
                          'table_spoon_dist', 'table_small_bowl_dist',
                          'gripper_large_bowl_dist', 'gripper_spoon_dist',
                          'gripper_small_bowl_dist', 'gripper_state', 'force_x', 'force_y',
                          'force_z', 'torque_x', 'torque_y', 'torque_z', 'large_bowl_hist_0',
                          'large_bowl_hist_1', 'large_bowl_hist_2', 'small_bowl_hist_0',
                          'small_bowl_hist_1', 'small_bowl_hist_2', 'spoon_hist_0',
                          'spoon_hist_1', 'spoon_hist_2', 'small_bowl_color_0',
                          'small_bowl_color_1', 'small_bowl_color_2', 'large_bowl_color_0',
                          'large_bowl_color_1', 'large_bowl_color_2', 'spoon_color_0',
                          'spoon_color_1', 'spoon_color_2', 'large_bowl_volume',
                          'small_bowl_volume', 'spoon_volume']}

fv_subset = {'drawer': range(len(feature_names['drawer']))[:9],
             'lamp': range(len(feature_names['lamp']))[:9],
             'pitcher': range(len(feature_names['pitcher']))[:12],
             'bowl': range(len(feature_names['bowl']))[:16]}

'''
fv_subset = {'drawer': range(len(feature_names['drawer'])),
             'lamp': range(len(feature_names['lamp'])),
             'pitcher': list(set(range(len(feature_names['pitcher']))) - set([0])),
             'bowl': list(set(range(len(feature_names['bowl']))) - set([0,1,2]))}
'''

def feature_vector_subset(fvs, task):
    return fvs[:,fv_subset[task]] 

def main():
    data_file = sys.argv[1]
    task = sys.argv[2]
    clf = clfs[sys.argv[3]]

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)
    num_pids = dataset.get_num_pids()

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

        #train_X = feature_vector_subset(train_X, task)
        #test_X = feature_vector_subset(test_X, task)

        scaler = StandardScaler()
        scaler.fit(train_X)

        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        #weights = compute_sample_weight('balanced', train_y)

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

        '''
        if type(clf) is DecisionTreeClassifier:
            #export_graphviz(clf, out_file=data_file[:-4]+'_'+task+'_'+str(i)+'.dot',
            #                feature_names=feature_names[task],
            #                class_names=['KF', 'Step'])
            export_graphviz(clf, out_file=data_file[:-4]+'_'+task+'_'+str(i)+'.dot',
                            feature_names=np.array(feature_names[task])[fv_subset[task]],
                            class_names=['KF', 'Step'])
        elif type(clf) is RandomForestClassifier:
            pass
        '''

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

