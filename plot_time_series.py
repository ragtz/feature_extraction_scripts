#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Plot time series data in dataset')
    parser.add_argument('--filename', metavar='PKL', required=True, help='Dataset pickle file')
    parser.add_argument('--state_spaces', metavar='PKL', nargs='+', required=False, default=['state', 'action'], help='Dataset pickle file')

    args = parser.parse_args()
    filename = args.filename
    state_spaces = args.state_spaces

    data = pickle.load(open(filename))

    for pid in data:
        for task in data[pid]:
            for demo_num in data[pid][task]:
                demo = data[pid][task][demo_num]
                t = demo['t']
                fig, ax = plt.subplots(len(state_spaces), 1, sharex=True)
                fig.suptitle(pid+'_'+task+'_'+demo_num.split('_')[0])
                for i, state_space in enumerate(state_spaces):
                    x = demo[state_space] / (np.max(demo[state_space], axis=0) - np.min(demo[state_space], axis=0))
                    n = demo[state_space+'_names']
                    ax.plot(t, x)
                    ax.legend(n)
                    ax.set_title(state_space)
                    #ax[i].plot(t, x)
                    #ax[i].legend(n)
                    #ax[i].set_title(state_space)
                ax.set_xlabel('Time (s)')
                #ax[-1].set_xlabel('Time (s)')
    plt.show()
                    
if __name__ == '__main__':
    main()

