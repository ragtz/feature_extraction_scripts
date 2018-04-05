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

def count_step_clusters(counts):
    n = 0
    c = [0,0]
    for k in counts:
        if counts[k]['kf'] == 0 and counts[k]['step'] > 0:
            n += 1
            c[1] += counts[k]['step']
    return n, c

def count_keyframe_clusters(counts):
    n = 0
    c = [0,0]
    for k in counts:
        if counts[k]['kf'] > 0 and counts[k]['step'] == 0:
            n += 1
            c[0] += counts[k]['kf']
    return n, c

def count_mixed_clusters(counts):
    n = 0
    c = [0,0]
    for k in counts:
        if counts[k]['kf'] > 0 and counts[k]['step'] > 0:
            n += 1
            c[0] += counts[k]['kf']
            c[1] += counts[k]['step']
    return n, c

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

    '''
    gmms = []
    bics = []
    for k in range(10,101):
        print 'Fitting gmm with k=' + str(k) + '...'
        cluster = GMM(n_components=k, n_init=10)
        cluster.fit(kfs)

        gmms.append(cluster)
        bics.append(cluster.bic(kfs))

    best_k = range(10,101)[np.argmin(bics)]
    cluster = gmms[np.argmin(bics)]

    clusters = cluster.predict(kfs)
    '''

    counts = count_cluster_membership(clusters, labels)
    step_clusters = count_step_clusters(counts)
    kf_clusters = count_keyframe_clusters(counts)
    mixed_clusters = count_mixed_clusters(counts)

    #print 'Best k:', best_k
    for k in counts:
        print '---------- Cluster', k, '----------'
        print 'KF:', counts[k]['kf']
        print 'Step:', counts[k]['step']

    print '===================='
    print 'N:', len(kfs)
    print 'Step Clusters:', step_clusters
    print 'Keyframe Clusters:', kf_clusters
    print 'Mixed Clustes:', mixed_clusters
    print '===================='

if __name__ == '__main__':
    main()

