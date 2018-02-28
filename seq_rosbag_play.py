#!/usr/bin/env python

from os.path import isfile, join, splitext, basename
from os import listdir, walk
from datetime import datetime
from rosbag_record import *
import numpy as np
import subprocess
import argparse
import signal
import rospy
import time
import sys
import re

OR = lambda z: reduce(lambda x, y: x or y, z, False)
topics_list = ['/j2s7s300_driver/out/joint_state',
               '/j2s7s300_driver/out/joint_torques',
               '/j2s7s300_driver/out/tool_wrench',
               '/joint_states',
               '/kinect/qhd/camera_info',
               '/kinect/qhd/image_color_rect/compressed',
               #'/kinect/qhd/image_depth_rect/compressed',
               #'/kinect/sd/camera_info',
               #'/kinect/sd/image_color_rect/compressed',
               #'/kinect/sd/image_depth_rect/compressed',
               '/vector/right_gripper/stat',
               '/kf_tracker/state',
               '/eef_pose',
               '/audio',
               '/beliefs/features']
topics = reduce(lambda x, y: x + ' ' + y, topics_list)

def get_files(path, type=None):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    if not type is None:
        files = filter(lambda f: f.endswith('.'+type), files)

    return files

def get_files_recursive(path, dirs_ignore=[], type=None, full=True):
    in_dir = lambda path, directory: path.endswith(directory)
    in_dirs = lambda path, directories: OR([in_dir(path, directory) for directory in directories])
    prepend_path = lambda path, files: [join(path, f) for f in files]   

    if full: 
        files = reduce(lambda x, y: x + y, [prepend_path(r, fs) for r, _, fs in walk(path) if not in_dirs(r, dirs_ignore)])
    else:
        files = reduce(lambda x, y: x + y, [f for r, _, fs in walk(path) if not in_dirs(r, dirs_ignore)])

    if not type is None:
        files = filter(lambda f: f.endswith('.'+type), files)

    return files

def sort_by_timestamp(files, delim='_', format='%Y-%m-%dT%H%M%S'):
    basenames = [splitext(basename(f))[0] for f in files]
    timestamps = [datetime.strptime(re.split('_', base)[-1], format) for base in basenames]
    sorted_idxs = np.argsort(timestamps)
    return list(np.array(files)[sorted_idxs])

def get_task_name(bag_file):
    return bag_file.split('/')[-3]

def get_skill_name(bag_file):
    return bag_file.split('/')[-2]

def get_filename(bag_file):
    return bag_file.split('/')[-1]

def main():
    rospy.init_node('seq_rosbag_play_node')

    parser = argparse.ArgumentParser(description='Playback bag files in a directory in timestamped order.')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files to play')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for recorded bag files')
    parser.add_argument('--ignore', metavar='DIR', nargs='+', default=[], required=False, help='Directories to ignore')
    
    args = parser.parse_args()
    src = args.src
    tar = args.target
    dirs_ignore = args.ignore

    bag_files = get_files_recursive(src, dirs_ignore, 'bag')
    sorted_bag_files = sort_by_timestamp(bag_files)
    n = len(sorted_bag_files)

    for i, bag_file in enumerate(sorted_bag_files):
        print '=============================='
        print 'Playing bag file', i+1, 'of', n
        print '=============================='
        
        task = get_task_name(bag_file)
        skill = get_skill_name(bag_file)
        filename = get_filename(bag_file)

        # start recording
        rosbag_proc = start_record(join(tar, task, skill), filename, topics)
        time.sleep(2.0)

        # playback bag file
        subprocess.call(['rosbag', 'play', '--clock', bag_file, '/joint_states:=/joint_states_pre_filtered'])

        # stop recording
        stop_record(rosbag_proc)
        time.sleep(2.0)

    rospy.loginfo('Done.')

if __name__ == '__main__':
    main()

