#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import argparse
import pickle

def get_pid_num(pid):
    return pid[1:]

def get_did_num(did):
    return did.split('_')[0][1:]

def sort_keys(keys, f):
    return np.array(keys)[np.argsort([f(k) for k in keys])]

def sort_pids(pids):
    return sort_keys(pids, get_pid_num)

def sort_dids(dids):
    return sort_keys(dids, get_did_num)

def handle_pids(data, pids):
    if not pids:
        pids = list(set(data.keys()))
    return pids

def handle_tasks(data, pids, tasks):
    if not tasks:
        tasks = set([])
        for pid in pids:
            tasks = tasks.union(data[pid].keys())
        tasks = list(tasks)
    return tasks    

def data_subset(data, pids, tasks):
    pids = handle_pids(data, pids)
    tasks = handle_tasks(data, pids, tasks)

    d = {}

    for pid in data:
        if pid in pids:
            for task in data[pid]:
                if task in tasks:
                    for demo in data[pid][task]:
                        if not pid in d:
                            d[pid] = {}

                        if not task in d[pid]:
                            d[pid][task] = {}

                        d[pid][task][demo] = deepcopy(data[pid][task][demo])

    return d 

def get_segment(d, t, t_start, t_end):
    idx_start = np.argmin(np.abs(t - t_start))
    idx_end = np.argmin(np.abs(t - t_end))
    return d[idx_start:idx_end]

def change(pre, post, threshold):
    joint = np.concatenate((pre,post))
    var_pre = np.var(pre, axis=0)
    var_post = np.var(post, axis=0)
    var_joint = np.var(joint, axis=0)

    var_pre_joint_ratio = (var_pre/var_joint)[np.newaxis,:]
    var_post_joint_ratio = (var_post/var_joint)[np.newaxis,:]
    
    return np.max(np.concatenate((var_pre_joint_ratio, var_post_joint_ratio)), axis=0) > threshold

def segment(demo, threshold, k):
    t = demo['t']
    keyframes = demo['kf']
    state = demo['state']

    seg_points = []
    for idx, kf in enumerate(keyframes[1:-1]):
        prev_kf = keyframes[idx]
        next_kf = keyframes[idx+2]

        pre = get_segment(state, t, prev_kf, kf)
        post = get_segment(state, t, kf, next_kf)
        #pre = get_segment(state, t, kf-2, kf)
        #post = get_segment(state, t, kf, kf+2)

        c = change(pre, post, threshold)
        if c.sum() > k:
            seg_points.append(idx+1)

    seg_points = [0] + seg_points + [len(keyframes)]
    return seg_points

def print_segmentation_results(segs):
    for task in segs:
        print '==========', task, '=========='
        for pid in sort_pids(segs[task].keys()):
            print '----------', pid, '----------'
            for demo in sort_dids(segs[task][pid].keys()):
                print demo + ': ' + str(segs[task][pid][demo])

def main():
    parser = argparse.ArgumentParser(description='Segment keyframe demonstrations')
    parser.add_argument('--data', metavar='PKL', required=True, help='Dataset pkl file')
    parser.add_argument('--threshold', metavar='FLOAT', type=float, required=True, help='Variance ratio threshold to detect segmentation point')
    parser.add_argument('--k', metavar='INT', type=int, default=1, required=False, help='Number of features needed to change to detect segmentation point')
    parser.add_argument('--pids', metavar='PID', nargs='+', default=[], required=False, help='Pids from dataset to process')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Tasks from dataset to process')

    args = parser.parse_args()
    data = pickle.load(open(args.data))
    thres = args.threshold
    k = args.k
    pids = args.pids
    tasks = args.tasks

    data = data_subset(data, pids, tasks)

    segs = {}
    for pid in data:
        for task in data[pid]:
            for demo in data[pid][task]:
                if not task in segs:
                    segs[task] = {}

                if not pid in segs[task]:
                    segs[task][pid] = {}

                segs[task][pid][demo] = segment(data[pid][task][demo], thres, k)

    print_segmentation_results(segs)
 
if __name__ == '__main__':
    main()

