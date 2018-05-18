#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import itertools
import argparse
import pickle

objs = ['drawer',
        'lamp',
        'pitcher',
        'bowl',
        'spoon',
        'large_bowl',
        'small_bowl']
eef = 'gripper'
table = 'table'

def get_state(demo, names):
    idxs = [i for i, n in enumerate(demo['state_names']) if n in names]
    s = demo['state'][:,idxs]

    if len(s.shape) > 1:
        return s
    else:
        return s[:,np.newaxis]

def get_action(demo, names):
    idxs = [i for i, n in enumerate(demo['action_names']) if n in names]
    a = demo['action'][:,idxs]

    if len(a.shape) > 1:
        return a
    else:
        return a[:,np.newaxis]

def get_dists(demo, objs, ref=None):
    names = []

    if ref is None:
        for o1, o2 in itertools.product(objs, objs):
            names.append(o1+'_'+o2+'_dist')
    else:
        for obj in objs:
            names.append(ref+'_'+obj+'_dist')
    
    return get_state(demo, names)

def get_colors(demo, objs):
    colors = []
    for obj in objs:
        names = [obj+'_color_0', obj+'_color_1', obj+'_color_2']
        c = get_state(demo, names)
        if c.shape[1] > 0:
            colors.append(get_state(demo, names))
    return np.array(colors)

def get_volumes(demo, objs):
    names = []
    for obj in objs:
        names.append(obj+'_volume')
    return get_state(demo, names)

def get_avg_obj_obj_dist(demo):
    dists = get_dists(demo, objs)
    if dists.shape[1] > 0:
        return (['avg_obj_obj_dist'], np.mean(dists, axis=1)[:,np.newaxis])
    else:
        #return ([], np.array([]))
        return (['avg_obj_obj_dist'], np.zeros((len(demo['t']),1)))

def get_avg_table_obj_dist(demo):
    dists = get_dists(demo, objs, ref=table)
    return (['avg_table_obj_dist'], np.mean(dists, axis=1)[:,np.newaxis])

def get_avg_eef_obj_dist(demo):
    dists = get_dists(demo, objs, ref=eef)
    return (['avg_eef_obj_dist'], np.mean(dists, axis=1)[:,np.newaxis])

def get_avg_obj_color(demo):
    colors = get_colors(demo, objs)
    return (['avg_obj_color_0', 'avg_obj_color_1', 'avg_obj_color_2'], np.mean(colors, axis=0))

def get_avg_obj_volume(demo):
    volumes = get_volumes(demo, objs)
    return (['avg_obj_volume'], np.mean(volumes, axis=1)[:,np.newaxis])

def get_eef_direction(demo):
    names = ['gripper_x_vel', 'gripper_y_vel', 'gripper_z_vel']
    dx_norm = get_action(demo, names)
    dx_norm = dx_norm / np.repeat([np.linalg.norm(dx_norm, axis=1)], 3, axis=0).T
    return (['gripper_dx_norm', 'gripper_dy_norm', 'gripper_dz_norm'], dx_norm)

def get_eef_state(demo):
    names = ['gripper_state']
    return (names, get_state(demo, names))

def get_eef_force(demo):
    names = ['force_x', 'force_y', 'force_z']
    return (names, get_state(demo, names))

def get_eef_torque(demo):
    names = ['torque_x', 'torque_y', 'torque_z']
    return (names, get_state(demo, names))

def merge(*args):
    feature_names = []
    feature_vector = None

    for arg in args:
        n, v = arg

        if n:
            feature_names.extend(list(n))
            if feature_vector is None:
                feature_vector = v
            else:
                feature_vector = np.hstack((feature_vector, v))

    return (np.array(feature_names), np.array(feature_vector))

def get_task_independent_features(demo):
    demo_ti = {}
    demo_ti['kf'] = deepcopy(demo['kf'])
    demo_ti['steps'] = deepcopy(demo['steps'])
    demo_ti['t'] = deepcopy(demo['t'])

    d = merge(get_avg_obj_obj_dist(demo),
              get_avg_table_obj_dist(demo),
              get_avg_eef_obj_dist(demo),
              get_avg_obj_volume(demo),
              get_avg_obj_color(demo),
              get_eef_direction(demo),
              get_eef_state(demo),
              get_eef_force(demo),
              get_eef_torque(demo))
    n, v = d
    
    demo_ti['state_names'] = n
    demo_ti['state'] = v

    return demo_ti

def main():
    parser = argparse.ArgumentParser(description='Extract task independent features')
    parser.add_argument('--demos', metavar='PKL', required=True, help='Demos pkl file')
    parser.add_argument('--filename', metavar='PKL', required=True, help='Output filename')

    args = parser.parse_args()
    demos = pickle.load(open(args.demos))
    filename = args.filename

    demos_ti = {}

    for pid in demos:
        for task in demos[pid]:
            for demo_id in demos[pid][task]:
                demo = demos[pid][task][demo_id]
                
                if not pid in demos_ti:
                    demos_ti[pid] = {}

                if not task in demos_ti[pid]:
                    demos_ti[pid][task] = {}

                demos_ti[pid][task][demo_id] = get_task_independent_features(demo)

    pickle.dump(demos_ti, open(filename, 'w'))

if __name__ == '__main__':
    main()

