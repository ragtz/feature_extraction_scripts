#!/usr/bin/env python

from os.path import isdir, isfile, join, basename
from os import listdir
import numpy as np
import argparse
import pickle
import csv
import re

pid_pattern = 'p[0-9]+'
label_csv_pattern = pid_pattern + '_labels.csv'

is_pid = lambda f: isdir(f) and re.match(pid_pattern, basename(f))
is_label_csv = lambda f: isfile(f) and re.match(label_csv_pattern, basename(f))

def get_label_dict(f):
    labels = {}

    with open(f) as csvfile:
        reader = csv.DictReader(csvfile)
        curr_task = ''
        curr_demo = ''
        for row in reader:
            task = row['task']
            demo = row['demo']
            label = row['label']

            if task and not task in labels:
                labels[task] = {}

            if demo and not demo in labels[task]:
                labels[task][demo] = []

            if task and task != curr_task:
                curr_task = task

            if demo and demo != curr_demo:
                curr_demo = demo

            if label:
                labels[curr_task][curr_demo].append(ord(label) - ord('a'))

    return labels

def main():
    parser = argparse.ArgumentParser(description='Merge keyframe label csv files and save as pkl')
    parser.add_argument('--path', metavar='DIR', required=True, help='Path to directory containing keyframe label csv files')
    parser.add_argument('--pkl', metavar='PKL', required=True, help='Name of merged labels pkl file')    

    args = parser.parse_args()
    path = args.path
    pkl = join(path, args.pkl)

    labels = {}

    for pid in listdir(path):
        pid_path = join(path, pid)
        if is_pid(pid_path):
            for f in listdir(pid_path):
                label_csv = join(pid_path, f)
                if is_label_csv(label_csv):
                    labels[pid] = get_label_dict(label_csv)

    pickle.dump(labels, open(pkl, 'w'))

if __name__ == '__main__':
    main()

