#!/usr/bin/env python

from fetch_files import *
import matplotlib.pyplot as plt
import argparse
import rosbag
import sys

def main():
    parser = argparse.ArgumentParser(description='Show hue/volume plot for task')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.aad_argument('--task', required='TASK', required=True, help='Task to plot')

    args = parser.parse_args()
    src = args.src
    task = args.task

    hue = []
    vol = []
    
    bag_files = get_files_recursive(src, task, 'bag')
    for bag_file in bag_files:
        with bag as rosbag.Bag(bag_file):
            for _, msg, _, in bag.read_messages(topic=['/beliefs/features']):
                for obj in msg.objects:
                    hue.append(obj.basicInfo.hue)
                    vol.append(obj.obb.bb_dims.x*obj.obb.bb_dims.y*obj.obb.bb_dims.z)

    plt.plot(hue, vol)
    plt.show()    

if __name__ == '__main__':
    main()

