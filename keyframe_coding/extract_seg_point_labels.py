#!/usr/bin/env python

from keyframe_labels_utils import *
import numpy as np
import pickle
import argparse

def get_seg_label_sequence(p_labels):
    p_labels = np.array(p_labels)
    s_labels = [1 if s else 0 for s in p_labels[:-1] != p_labels[1:]] + [0]
    return s_labels

def extract_seg_labels(p_labels):
    s_labels = {}
    
    for pid in p_labels:
        for task in p_labels[pid]:
            for demo in p_labels[pid][task]:
                if not pid in s_labels:
                    s_labels[pid] = {}

                if not task in s_labels[pid]:
                    s_labels[pid][task] = {}

                s_labels[pid][task][demo] = get_seg_label_sequence(p_labels[pid][task][demo])

    return s_labels

def main():
    parser = argparse.ArgumentParser(description='Extract segmentation labels from primitive labels')
    parser.add_argument('--p_labels', metavar='PKL', required=True, help='Primitive lables')
    parser.add_argument('--s_labels', metavar='PKL', required=True, help='Segmentation labels')    
    parser.add_argument('--pregrasp', action='store_true', help='Include pre/post grasp labels')

    args = parser.parse_args()
    p_labels = pickle.load(open(args.p_labels))
    s_labels_file = args.s_labels
    pregrasp = args.pregrasp

    if not pregrasp:
        for pid in p_labels:
            for task in p_labels[pid]:
                for demo in p_labels[pid][task]:
                    labels = p_labels[pid][task][demo]
                    p_labels[pid][task][demo] = relabel(labels, task)
    
    s_labels = extract_seg_labels(p_labels)

    pickle.dump(s_labels, open(s_labels_file, 'w'))

if __name__ == '__main__':
    main()

