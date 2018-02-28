#!/usr/bin/env python

from fetch_files import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import rosbag

def main():
    parser = argparse.ArgumentParser(description='Show topic plot for all bag files in directory')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')

    args = parser.parse_args()
    src = args.src

    bag_files = get_files_recursive(src, type='bag')
    n = len(bag_files)
    
    tilt = []
    topic = '/joint_states'

    for i, bag_file in enumerate(bag_files):
        print 'Processing bag file ' + str(i+1) + ' of ' + str(n) + '...'
        with rosbag.Bag(bag_file) as bag:
            for _, msg, _, in bag.read_messages(topics=[topic]):
                if msg.position[15] > 0.6:
                    print bag_file
                    break
                tilt.append(msg.position[15])

    plt.hist(tilt, bins=500)
    plt.xlabel('Tilt Value')
    plt.ylabel('Counts')

    plt.show()    

if __name__ == '__main__':
    main()

