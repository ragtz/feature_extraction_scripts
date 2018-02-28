#!/usr/bin/env python

from os.path import join, expanduser
from copy import deepcopy
from fetch_files import *
import argparse
import rosbag
import rospy

def is_built_in_type(x):
    return isinstance(x, bool) or isinstance(x, int) or isinstance(x, long) or isinstance(x, float) or isinstance(x, str) or isinstance(x, rospy.Time) or isinstance(x, rospy.Duration)

def rewrite_header(msg, t):
    if not is_built_in_type(msg):
        if isinstance(msg, list) or isinstance(msg, tuple):
            for elm in msg:
                rewrite_header(elm, t)
        else:
            if msg._has_header:
                msg.header.stamp = t
            for fieldname in msg.__slots__:
                field = getattr(msg, fieldname)
                rewrite_header(field, t)
 
def main():
    parser = argparse.ArgumentParser(description='Rewrite header to pusblish time')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved bag files')
    parser.add_argument('--topics', metavar='TOPIC', nargs='+', required=False, help='Topics to rewrite header')

    args = parser.parse_args()
    src = args.src
    tar = args.target
    topics = args.topics
    
    bag_files = get_files_recursive(src, type='bag')
    n = len(bag_files)
    
    for i, bag_file_in in enumerate(bag_files):
        print 'Processing bag file ' + str(i+1) + ' of ' + str(n) + '...'
        
        task = get_task_name(bag_file_in)
        skill = get_skill_name(bag_file_in)
        filename = get_filename(bag_file_in)
        bag_file_out = join(expanduser('~'), tar, task, skill, filename)
        ensure_dir(bag_file_out)

        with rosbag.Bag(bag_file_out, 'w') as out_bag:
            for topic, msg, t in rosbag.Bag(bag_file_in).read_messages():
                if topics is None or topic in topics:
                    rewrite_header(msg, t)
                out_bag.write(topic, msg, t)

if __name__ == '__main__':
    main()

