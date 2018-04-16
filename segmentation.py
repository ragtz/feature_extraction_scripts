#!/usr/bin/env python

from copy import deepcopy
import numpy as np
import argparse

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

    for pid in demos:
        if pid in pids:
            for task in demos[pid]:
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
    var_pre = np.var(pre, axis=1)
    var_post = np.var(post, axis=1)
    var_joint = np.var(joint, axis=1_
    return np.max(np.concatenate((var_pre/var_joint, var_post/var_joint)), axis=1) > threshold

def segment(demo, threshold, k):
    t = demo['t']
    keyframes = demo['kf']
    state = demo['state']

    seg_points = []
    for idx, kf in enumerate(keyframes[1:-1]):
        prev_kf = keyframes[idx]
        next_kf = keyframes[idx+2]

        pre = get_segment(demo, t, prev_kf, kf)
        post = get_segment(demo, t, kf, next_kf)

        c = change(pre, post, threshold)
        if c.sum() > k:
            seg_points.append(idx+1)

    seg_points = [0] + seg_points + [len(keyframes)]
    return seg_points

def print_segmentation_results(segs):
    for task in segs:
        print '==========', task, '=========='
        for pid in segs[task]:
            print '----------', pid, '----------'
            for demo in segs[task][pid]:
                print demo + ': ' + str(segs[task][pid][demo])

def main():
    parser = argparse.ArgumentParser(description='Segment keyframe demonstrations')
    parser.add_argument('--data', metavar='PKL', required=True, help='Dataset pkl file')
    parser.add_argument('--threshold', metavar='FLOAT', type=float, required=True, help='Variance ratio threshold to detect segmentation point')
    parser.add_argument('--k', metavar='INT', type=int, default=1, required=False, help='Number of features needed to change to detect segmentation point')
    parser.add_argument('--pids', metavar='PID', nargs='+', default=[], required=False, help='Pids from dataset to process')
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Tasks from dataset to process')

    args = parser.parse_args()
    data = args.data
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

