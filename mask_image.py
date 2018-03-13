#!/usr/bin/env python

from os.path import join, expanduser
from fetch_files import *
from cv_bridge import CvBridge
import numpy as np
import argparse
import rosbag
import cv2
import sys

def mask_image_msg(msg, mask, bridge):
    header = msg.header
    img = bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
    img = mask*img
    msg = bridge.cv2_to_compressed_imgmsg(img)
    msg.header = header
    return msg

def main():
    parser = argparse.ArgumentParser(description='Mask images in image topic')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.add_argument('--target', metavar='DIR', required=True, help='Path to directory for saved bag files')
    
    args = parser.parse_args()
    src = args.src
    tar = args.target
    topics = ['/kinect/qhd/image_color_rect/compressed']
    
    bag_files = get_files_recursive(src, type='bag')
    n = len(bag_files)

    bridge = CvBridge()
    mask = np.ones((540,960,3), dtype=np.uint8)
    mask[:150,:,:] = 0
    mask[:,:150,:] = 0

    for i, bag_file_in in enumerate(bag_files):
        print 'Processing bag file ' + str(i) + ' of ' + str(n) + '...'

        task = get_task_name(bag_file_in)
        skill = get_skill_name(bag_file_in)
        filename = get_filename(bag_file_in)
        bag_file_out = join(expanduser('~'), tar, task, skill, filename)
        ensure_dir(bag_file_out)

        with rosbag.Bag(bag_file_out, 'w') as out_bag:
            for topic, msg, t in rosbag.Bag(bag_file_in).read_messages():
                if topic in topics:
                    msg = mask_image_msg(msg, mask, bridge)
                out_bag.write(topic, msg, t)

if __name__ == '__main__':
    main()

