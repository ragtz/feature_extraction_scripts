#!/usr/bin/env python

from sklearn.metrics import cohen_kappa_score
from keyframe_labels_utils import *
import pickle
import argparse

def get_labels(c1, c2, pregrasp, tasks):
    check_combatability(c1, c2)

    c1_labels = []
    c2_labels = []

    for pid in c1:
        for task in c1[pid]:
            if task in tasks:
                for demo in c1[pid][task]:
                    if not pregrasp:
                        c1[pid][task][demo] = relabel(c1[pid][task][demo], task)
                        c2[pid][task][demo] = relabel(c2[pid][task][demo], task)

                    c1_labels.extend([task+'-'+str(label) for label in c1[pid][task][demo]])
                    c2_labels.extend([task+'-'+str(label) for label in c2[pid][task][demo]])
    
    return c1_labels, c2_labels

def main():
    parser = argparse.ArgumentParser(description='Compute Cohen\'s kappa score for keyframe labels')
    parser.add_argument('--c1', metavar='PKL', required=True, help='Lables assigned by first coder')
    parser.add_argument('--c2', metavar='PKL', required=True, help='Labels assigned by second coder')    
    parser.add_argument('--tasks', metavar='TASK', nargs='+', default=[], required=False, help='Tasks to compute agreement')
    parser.add_argument('--pregrasp', action='store_true', help='Include pre/post grasp labels')

    args = parser.parse_args()
    c1 = pickle.load(open(args.c1))
    c2 = pickle.load(open(args.c2))
    tasks = args.tasks if args.tasks else ['drawer', 'lamp', 'pitcher', 'bowl']
    pregrasp = args.pregrasp
    
    c1_labels, c2_labels = get_labels(c1, c2, pregrasp, tasks)

    print 'Cohen\'s kappa:', cohen_kappa_score(c1_labels, c2_labels)

if __name__ == '__main__':
    main()

