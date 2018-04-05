#!/usr/bin/env python

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from keyframe_dataset import *
import numpy as np
import pickle
import sys

def count_cluster_membership(cluster_labels, kf_labels):
    clusters = {k: [] for k in set(cluster_labels)}
    for idx, k in enumerate(cluster_labels):
        clusters[k].append(idx)

    kf_counts_per_cluster = {k: {'kf': 0, 'step': 0} for k in set(cluster_labels)}
    for k in clusters:
        for idx in clusters[k]:
            kf_label = ['kf', 'step'][kf_labels[idx]]
            kf_counts_per_cluster[k][kf_label] += 1

    return kf_counts_per_cluster

def relabel(counts, labels):
    new_labels = []
    for i, label in enumerate(labels):
        if label == 0:
            k = [k for k in counts if i in counts[k]['kf']][0]
            if is_mixed_cluster(counts[k]):
                new_labels.append(1)
            else:
                new_labels.append(0)
        else:
            new_labels.append(1)

def main():
    data_file = sys.argv[1]
    task = sys.argv[2]
    k = int(sys.argv[3])
    
    cluster = KMeans(n_clusters=k)

    dataset = KeyframeDataset()
    dataset.load(data_file)
    dataset = dataset.get_keyframe_dataset(task=task)

    kfs = dataset.get_keyframes()
    labels = dataset.get_labels()

    scaler = StandardScaler()
    scaler.fit(kfs)

    kfs = scaler.transform(kfs)

    cluster.fit(kfs)
    clusters = cluster.predict(kfs)


if __name__ == '__main__':
    main()

