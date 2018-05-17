#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from keyframe_dataset import *
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

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

def main():
    data_file = sys.argv[1]
    task = sys.argv[2]

    clf = SVC()

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)

    X = dataset.data['kf']
    y = dataset.data['label']

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    fprs = []
    tprs = []
    roc_aucs = []
    for i in range(X.shape[1]):
        x = X[:,i]
        x = np.reshape(x, (len(x), 1))

        clf.fit(x, y)
        p = clf.decision_function(x)

        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)

        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    cols = 5
    rows = len(fprs)/cols
    if len(fprs)%cols != 0:
        rows += 1

    fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')
    fig.suptitle(task+' feature ROCs')
    fig.text(0.5, 0.05, 'False Positive Rate', ha='center')
    fig.text(0.08, 0.5, 'True Positive Rate', va='center', rotation='vertical')
    for i in range(len(fprs)):
        r = i/5
        c = i%5
       
        fpr = fprs[i]
        tpr = tprs[i]
        roc_auc = roc_aucs[i]

        ax[r,c].plot(fpr, tpr, color='darkorange', lw=2, label='area = %0.2f' % roc_auc)
        ax[r,c].plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        ax[r,c].set_xlim([0,1])
        ax[r,c].set_ylim([0,1.05])
        #ax[r,c].set_title(feature_names[task][i])
        ax[r,c].legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()

