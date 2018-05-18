#!/usr/bin/env python

from collections import Counter
import numpy as np
import argparse
import pickle

OR = lambda z: reduce(lambda x, y: x or y, z)

def make_np(d):
    for k in d:
        d[k] = np.array(d[k])

def count_demos(demos):
    n = 0

    for pid in demos:
        for task in demos[pid]:
            for demo_id in demos[pid][task]:
                n += 1

    return n

def get_did_num(did):
    return int(did.split('_')[0][1:])

def get_segment(d, t, t_start, t_end):
    idx_start = np.argmin(np.abs(t - t_start))
    idx_end = np.argmin(np.abs(t - t_end))
    return d[idx_start:idx_end]

def get_keyframe_feature_vector(kf_time, time, state, delta_t):
    pre = get_segment(state, time, kf_time-delta_t, kf_time)
    post = get_segment(state, time, kf_time, kf_time+delta_t)

    if len(pre) > 4 and len(post) > 4:
        pre_median = np.median(pre, axis=0)
        post_median = np.median(post, axis=0)
        median_diff = post_median - pre_median

        f = np.concatenate((pre_median, post_median, median_diff))
        return f.flatten()
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='Compile labeled keyframe dataset')
    parser.add_argument('--data', metavar='PKL', required=True, help='Dataset pkl file')
    parser.add_argument('--labels', metavar='PKL', required=True, help='keyframe labels pkl file')
    parser.add_argument('--delta_t', metavar='SEC', type=float, required=True, help='Time window before and after keyframe')
    parser.add_argument('--kf', metavar='PKL', required=True, help='Name of pkl keyframe dataset file')

    args = parser.parse_args()
    data = pickle.load(open(args.data))
    labels = pickle.load(open(args.labels))
    delta_t = args.delta_t
    kf_file = args.kf

    dataset = {'pid': [],
               'task': [],
               'demo_id': [],
               'kf_idx': [],
               'kf': [],
               'label': []}

    n = count_demos(data)
    i = 1

    for pid in data:
        for task in data[pid]:
            for demo_id in data[pid][task]:
                print 'Processing ' + pid + ', ' + task + ', ' + demo_id + ' (' + str(i) + '/' + str(n) + ')...'

                kf_features = []
                for kf_time in data[pid][task][demo_id]['kf']:
                    kf_feature = get_keyframe_feature_vector(kf_time, 
                                                             data[pid][task][demo_id]['t'],
                                                             data[pid][task][demo_id]['state'],
                                                             delta_t)
                    kf_features.append(kf_feature)

                for idx, kf_feature in enumerate(kf_features):
                    if not kf_feature is None:
                        dataset['pid'].append(pid)
                        dataset['task'].append(task)
                        dataset['demo_id'].append(demo_id)
                        dataset['kf_idx'].append(idx)
                        dataset['kf'].append(kf_feature)
                        dataset['label'].append(labels[pid][task][str(get_did_num(demo_id))][idx])

                i += 1

    make_np(dataset)

    pickle.dump(dataset, open(kf_file, 'w'))

if __name__ == '__main__':
    main()

