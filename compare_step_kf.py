#!/usr/bin/env python

from extract_kf_images import *
from demo_file_utils import *
from fetch_files import *
import numpy as np
import argparse
import pickle

def is_gripper_feature(name):
    return 'gripper' in name

def is_position_feature(name):
    return '_position' in name

def is_color_feature(name):
    return '_color' in name

def is_volume_feature(name):
    return '_volume' in name

def is_object_feature(name, obj=None):
    if obj is None:
        return not is_gripper_feature(name)
    else:
        return obj in name

def is_object_position_feature(name, obj=None):
    return is_object_feature(name, obj) and is_position_feature(name)

def is_object_color_feature(name, obj=None):
    return is_object_feature(name, obj) and is_color_feature(name)

def is_object_volume_feature(name, obj=None):
    return is_object_feature(name, obj) and is_volume_feature(name)

def is_gripper_position_feature(name):
    return is_gripper_feature(name) and is_position_feature(name)

def is_gripper_state_feature(name):
    return is_gripper_feature(name) and not is_position_feature(name)

def get_object_names(names):
    obj_features = [name for name in names if is_object_feature(name)]
    return list(set([f.split('_')[0] for f in obj_features]))

def compute_step_kf_distance(x, t, steps, kf):
    steps = steps[1:-1]
    step_fvs = get_x_at_t(x, t, steps)
 
    kf_fvs = get_x_at_t(x, t, kf)
    kf_fvs = get_x_at_t(kf_fvs, kf, steps)

    return np.linalg.norm(np.array(step_fvs) - kf_fvs, axis=1)

def compute_max_distance_from_steps(x, steps):
    steps = steps[1:-1]
    max_dists = []

    for step in steps:
        dists = [np.linalg.norm(x[i] - step) for i in range(len(x))]
        max_dists.append(max(dists))

    return np.array(max_dists)

def compute_feature_step_kf_distance(names, f, x, t, steps, kf):
    idxs = [i for i, name in enumerate(names) if f(name)]
    return compute_step_kf_distance(x[:,idxs], t, steps, kf)

def compute_gripper_position_step_kf_distance(names, x, t, steps, kf):
    return compute_feature_step_kf_distance(names, is_gripper_position_feature, x, t, steps, kf)
    
def compute_gripper_state_step_kf_distance(names, x, t, steps, kf):
    return compute_feature_step_kf_distance(names, is_gripper_state_feature, x, t, steps, kf)
    
def compute_object_position_step_kf_distance(names, obj, x, t, steps, kf):
    def f(name):
        return is_object_position_feature(name, obj)

    return compute_feature_step_kf_distance(names, f, x, t, steps, kf)

def compute_object_color_step_kf_distance(names, obj, x, t, steps, kf):
    def f(name):
        return is_object_color_feature(name, obj)

    return compute_feature_step_kf_distance(names, f, x, t, steps, kf)

def compute_object_volume_step_kf_distance(names, obj, x, t, steps, kf):
    def f(name):
        return is_object_volume_feature(name, obj)

    return compute_feature_step_kf_distance(names, f, x, t, steps, kf)

def compute_step_kf_similarities(x, t, steps, kf):
    d = compute_step_kf_distance(x, t, steps, kf)
    m = compute_max_distance_from_steps(x, steps)
    return d/m

def print_results(f_dists, sims):
    for task in sims:
        print '--------------------'
        print task
        print '--------------------'
        for f_name in f_dists[task]:
            print f_name + ': ' + '{0:.2f}'.format(np.mean(f_dists[task][f_name]))
        print 'Percent Change:', '{0:.2f}'.format(100*np.mean(sims[task])) + '%'

def main():
    parser = argparse.ArgumentParser(description='Compare step and nearest keyframe feature vectors')
    parser.add_argument('--demos', metavar='PKL', required=True, help='Demos pkl file')
    parser.add_argument('--task', metavar='TSK', nargs='+', default=['drawer', 'lamp', 'pitcher', 'bowl'], required=False, help='Tasks to compare step and kf')

    args = parser.parse_args()
    demos = pickle.load(open(args.demos))
    tasks = args.task

    f_dists = {}
    sims = {}
    for task in tasks:
        f_dists[task] = {}
        sims[task] = []

        for pid in demos:
            if task in demos[pid]:
                for demo in demos[pid][task].values():
                    names = demo['state_names']
                    x = demo['state']
                    t = demo['t']
                    steps = demo['steps']
                    kf = demo['kf']

                    # add gripper position distances
                    if not 'gripper_position' in f_dists[task]:
                        f_dists[task]['gripper_position'] = []

                    dists = compute_gripper_position_step_kf_distance(names, x, t, steps, kf)
                    f_dists[task]['gripper_position'].extend(dists)

                    # add gripper state distances
                    if not 'gripper_state' in f_dists[task]:
                        f_dists[task]['gripper_state'] = []

                    dists = compute_gripper_state_step_kf_distance(names, x, t, steps, kf)
                    f_dists[task]['gripper_state'].extend(dists)

                    # add object feature distances
                    objs = get_object_names(names)
                    for obj in objs:
                        # add object position distances
                        if not obj+'_position' in f_dists[task]:
                            f_dists[task][obj+'_position'] = []

                        dists = compute_object_position_step_kf_distance(names, obj, x, t, steps, kf)
                        f_dists[task][obj+'_position'].extend(dists)

                        # add object color distances
                        if task == 'lamp':
                            if not obj+'_color' in f_dists[task]:
                                f_dists[task][obj+'_color'] = []

                            dists = compute_object_color_step_kf_distance(names, obj, x, t, steps, kf)
                            f_dists[task][obj+'_color'].extend(dists)

                        # add object volume distances
                        elif task == 'drawer':
                            if not obj+'_volume' in f_dists[task]:
                                f_dists[task][obj+'_volume'] = []

                            dists = compute_object_volume_step_kf_distance(names, obj, x, t, steps, kf)
                            f_dists[task][obj+'_volume'].extend(dists)

                    # add similarity scores
                    s = compute_step_kf_similarities(x, t, steps, kf) 
                    sims[task].extend(s)

    print_results(f_dists, sims)
 
if __name__ == '__main__':
    main()

