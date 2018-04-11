#!/usr/bin/env python

from os.path import isdir, isfile, join, basename
from os import listdir
import numpy as np
import argparse
import csv
import re

pid_pattern = 'p[0-9]+'
label_csv_pattern = pid_pattern + '_labels.csv'

is_pid = lambda f: isdir(f) and re.match(pid_pattern, basename(f))
is_label_csv = lambda f: isfile(f) and re.match(label_csv_pattern, basename(f))

AND = lambda z: reduce(lambda x, y: x and y, z)
OR = lambda z: reduce(lambda x, y: x or y, z)

class ErrorTypes:
    NOT_COMPLETE_ERROR = 0
    FINAL_STATE_ERROR = 1
    INCORRECT_LABEL_ERROR = 2
    LABEL_PROGRESS_ERROR = 3
    REQUIRED_KEYFRAME_ERROR = 4

    _error_strings = {NOT_COMPLETE_ERROR: 'Not all keyframes are labeled.',
                      FINAL_STATE_ERROR: 'Does not end in return arm to start.',
                      INCORRECT_LABEL_ERROR: 'Keyframe labels not in correct label range.',
                      LABEL_PROGRESS_ERROR: 'Keyframe labels to not reflect forward progression of the task.',
                      REQUIRED_KEYFRAME_ERROR: 'Required keyframes are missing.'}
    
    _labels = {'drawer': range(ord('a'), ord('g')+1),
               'lamp': range(ord('a'), ord('h')+1),
               'pitcher': range(ord('a'), ord('r')+1),
               'bowl': range(ord('a'), ord('l')+1)}

    _required_kfs = {'drawer': ['b', 'c', 'd', 'e', 'g'],
                     'lamp': ['b', 'c', 'd', 'f', 'h'],
                     'pitcher': ['b', 'c', 'f', 'g', 'j', 'k', 'm', 'n', 'o', 'p', 'r'],
                     'bowl': ['b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'l']}

    def _is_complete(self, label_seq):
        return not '' in label_seq

    def _ends_in_final_state(self, task, label_seq):
        return ord(label_seq[-1]) == self._labels[task][-1]

    def _labels_in_range(self, task, label_seq):
        for label in label_seq:
            if not ord(label) in self._labels[task]:
                return False
        return True

    def _forward_progress(self, label_seq):
        label_seq = np.array([ord(l) for l in label_seq])
        label_diffs = label_seq[1:] - label_seq[:-1]
        forward = label_diffs >= 0
        return forward.all()

    def _has_required_keyframes(self, task, label_seq):
        required_kfs = self._required_kfs[task]
        for kf in required_kfs:
            if not kf in label_seq:
                return False
        return True
        
    def get_errors(self, task, label_seq):
        errors = []

        if not self._is_complete(label_seq):
            errors.append(self.NOT_COMPLETE_ERROR)
        else:
            if not self._ends_in_final_state(task, label_seq):
                errors.append(self.FINAL_STATE_ERROR)
            
            if not self._labels_in_range(task, label_seq):
                errors.append(self.INCORRECT_LABEL_ERROR)
            elif not self._has_required_keyframes(task, label_seq):
                errors.append(self.REQUIRED_KEYFRAME_ERROR)
            elif not self._forward_progress(label_seq):
                errors.append(self.LABEL_PROGRESS_ERROR)

        return errors

    def to_string(self, error):
        return self._error_strings[error]

error_types = ErrorTypes()

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

            labels[curr_task][curr_demo].append(label)

    return labels

def compile_label_errors(f):
    label_dict = get_label_dict(f)
    errors = {task: {demo: [] for demo in label_dict[task]} for task in label_dict}

    for task in label_dict:
        for demo in label_dict[task]:
            label_seq = label_dict[task][demo]
            errors[task][demo] = error_types.get_errors(task, label_seq)
    
    return errors

def print_errors(errors):
    pids = np.array(errors.keys())
    pids = pids[np.argsort([int(pid[1:]) for pid in pids])]

    for pid in pids:

        print '=============================='
        print 'PID:', pid
        print '=============================='
        
        # if at least one demo among all tasks in pid has errors
        if OR([errors[pid][task][demo] for task in errors[pid] for demo in errors[pid][task]]):
            for task in errors[pid]:
                # if at least one demo in task has errors
                if OR([errors[pid][task][demo] for demo in errors[pid][task]]):

                    print 'Task:', task

                    demos = np.array(errors[pid][task].keys())
                    demos = demos[np.argsort([int(demo) for demo in demos])]

                    for demo in demos:
                        # if at demo has at least one error
                        if errors[pid][task][demo]:
                            print '    - Demo:', demo

                            for error in errors[pid][task][demo]:
                                print '        + ', error_types.to_string(error)
        else:
            print 'Done.'

def main():
    parser = argparse.ArgumentParser(description='Check keyframe label csv files')
    parser.add_argument('--path', metavar='DIR', required=True, help='Path to directory containing keyframe label csv files')
    
    args = parser.parse_args()
    path = args.path

    errors = {}

    for pid in listdir(path):
        pid_path = join(path, pid)
        if is_pid(pid_path):
            for f in listdir(pid_path):
                label_csv = join(pid_path, f)
                if is_label_csv(label_csv):
                    errors[pid] = compile_label_errors(label_csv)

    print_errors(errors)

if __name__ == '__main__':
    main()

