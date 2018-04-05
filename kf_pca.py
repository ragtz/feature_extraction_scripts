#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keyframe_dataset import *
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
    
    pca = PCA()

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)

    kfs = dataset.get_keyframes()
    labels = dataset.get_labels()

    scaler = StandardScaler()
    scaler.fit(kfs)

    kfs = scaler.transform(kfs)

    pca.fit(kfs)
    
    print np.cumsum(pca.explained_variance_ratio_)

if __name__ == '__main__':
    main()

