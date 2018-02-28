#!/usr/bin/env python

from fetch_files import *
from colorsys import rgb_to_hsv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import rosbag
import sys

def main():
    parser = argparse.ArgumentParser(description='Show hue/volume plot for task')
    parser.add_argument('--src', metavar='DIR', required=True, help='Path to directory containing bag files')
    parser.add_argument('--task', metavar='TASK', required=True, help='Task to plot')

    args = parser.parse_args()
    src = args.src
    task = args.task

    hue = []
    sat = []
    val = []
    vol = []
    
    bag_files = get_files_recursive(src, [task], 'bag')
    n = len(bag_files)
    
    for i, bag_file in enumerate(bag_files):
        print 'Processing bag file ' + str(i+1) + ' of ' + str(n) + '...'
        with rosbag.Bag(bag_file) as bag:
            for _, msg, _, in bag.read_messages(topics=['/beliefs/features']):
                for obj in msg.objects:
                    h, s, v = rgb_to_hsv(obj.basicInfo.rgba_color.r,
                                         obj.basicInfo.rgba_color.g,
                                         obj.basicInfo.rgba_color.b)
                    hue.append(h)
                    sat.append(s)
                    val.append(v)
                    vol.append(obj.obb.bb_dims.x*obj.obb.bb_dims.y*obj.obb.bb_dims.z)

    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

    ax[0,0].plot(hue, vol, 'r.')
    ax[0,0].set_ylabel('Volume')

    ax[1,0].hist(hue, bins=25, color='r')
    ax[1,0].set_xlabel('Hue')
    ax[1,0].set_ylabel('Counts')

    ax[0,1].plot(sat, vol, 'g.')

    ax[1,1].hist(sat, bins=25, color='g')
    ax[1,1].set_xlabel('Saturation')
    
    ax[0,2].plot(val, vol, 'b.')

    ax[1,2].hist(hue, bins=25, color='b')
    ax[1,2].set_xlabel('Value')

    plt.show()    

if __name__ == '__main__':
    main()

