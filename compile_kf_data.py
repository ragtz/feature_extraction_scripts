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

def get_keyframe_feature_vector(kf_time, time, state):
    idx = np.argmin(np.abs(time - kf_time))
    return state[idx]

def get_step_keyframe_index(step_time, kf_times):
    return np.argmin(np.abs(kf_times - step_time))

def main():
    parser = argparse.ArgumentParser(description='Compile labeled keyframe dataset')
    parser.add_argument('--demos', metavar='PKL', required=True, help='Name of pkl demo dataset file')
    parser.add_argument('--kf', metavar='PKL', required=True, help='Name of pkl keyframe dataset file')

    args = parser.parse_args()
    demos = args.demos
    kf_file = args.kf

    dataset = {'pid': [],
               'task': [],
               'demo_id': [],
               'kf_idx': [],
               'kf': [],
               'labels': []}

    demos = pickle.load(open(demos))

    n = count_demos(demos)
    i = 1

    for pid in demos:
        for task in demos[pid]:
            for demo_id in demos[pid][task]:
                print 'Processing ' + pid + ', ' + task + ', ' + demo_id + ' (' + str(i) + '/' + str(n) + ')...'

                kf_features = []
                for kf_time in demos[pid][task][demo_id]['kf']:
                    kf_feature = get_keyframe_feature_vector(kf_time, 
                                                             demos[pid][task][demo_id]['t'],
                                                             demos[pid][task][demo_id]['state'])
                    kf_features.append(kf_feature)

                step_kf_idxs = []
                for step_time in demos[pid][task][demo_id]['steps']:
                    step_kf_idx = get_step_keyframe_index(step_time,
                                                          demos[pid][task][demo_id]['kf'])
                    step_kf_idxs.append(step_kf_idx)

                if OR(map(lambda x: x > 1, Counter(step_kf_idxs).values())):
                    print '----- Keyframe repeated! -----'
                    print demos[pid][task][demo_id]['kf']
                    print demos[pid][task][demo_id]['steps']
                    print len(kf_features), step_kf_idxs

                for idx, kf_feature in enumerate(kf_features):
                    dataset['pid'].append(pid)
                    dataset['task'].append(task)
                    dataset['demo_id'].append(demo_id)
                    dataset['kf_idx'].append(idx)
                    dataset['kf'].append(kf_feature)
                    if idx in step_kf_idxs:
                        dataset['labels'].append(1)
                    else:
                        dataset['labels'].append(0)

                i += 1

    make_np(dataset)

    pickle.dump(dataset, open(kf_file, 'w'))

if __name__ == '__main__':
    main()

