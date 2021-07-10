"""
This is a dummy file for HW5 of CSE353 Machine Learning, Fall 2020
You need to provide implementation for this file

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 26-Oct-2020
Last modified: 26-Oct-2020
"""

import random
from numpy import ones,vstack
from numpy.linalg import lstsq
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev

class TrackletClustering(object):
    """
    You need to implement the methods of this class.
    Do not change the signatures of the methods
    """

    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.tracklets = []
        self.centroids = []
        self.feature_set = []
        self.feature_out = []
        self.label_outlier_dist = []

    def calc_distance(self, X1, X2):
        return(sum((np.array(X1) - np.array(X2))**2))**0.5

    def assignment(self, X, centroids):
        
        labels = []
        for i in X:
            distance=[]
            for j in centroids:
                distance.append(self.calc_distance(i, j))
            labels.append(np.argmin(distance))
            
        return labels

    def k_means(self, X, k):
        
        X = np.array(X)
        random_index = np.random.choice(X.shape[0], k, replace=False) 
        centroids = X[random_index]    
        
        optimal_centroids = centroids
        for i in range(20):
            labels = self.assignment(X, optimal_centroids)
            
            optimal_centroids = []
            new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(labels, columns=['labels'])],
                              axis=1)
            for c in set(new_df['labels']):
                current_cluster = new_df[new_df['labels'] == c][new_df.columns[:-1]]
                cluster_mean = current_cluster.mean(axis=0)
                optimal_centroids.append(cluster_mean)
                
        return optimal_centroids

    ### Using B-spline interpolation to build features
    def get_features(self):

        m = len(self.tracklets)
        self.feature_set = []

        for i in range(m):

            tracklet = self.tracklets[i]
            tracklet_centers = []
            boxes = tracklet["tracks"]

            for box in boxes:

                x_center = (box[1] + box[3])/2.0
                y_center = (box[2] + box[4])/2.0
                tracklet_centers.append([x_center, y_center])

            tracklet_centers = np.array(tracklet_centers)
            ind = np.lexsort((tracklet_centers[:,1],tracklet_centers[:,0])) 
            tracklet_centers = tracklet_centers[ind]

            if(len(tracklet_centers) < 4):
                continue
            
            tck, u = splprep(tracklet_centers.T, u=None, s=0.0, k=3) 
            u_new = np.linspace(u.min(), u.max(), 100)
            x_new, y_new = splev(u_new, tck, der=0)

            self.feature_set.append(np.concatenate((x_new, y_new), axis=0))


    def get_outliers(self):

        X = np.array(self.feature_set)
        labels = self.assignment(X, self.centroids)
        
        vehicles = 1 + np.arange(len(self.tracklets))
        max_from_centroids = [] 

        L = pd.concat([pd.DataFrame(X), pd.DataFrame(labels, columns=['labels'])],
                          axis=1)
        P = pd.concat([pd.DataFrame(L), pd.DataFrame(vehicles, columns=['vehicle_id'])],
                          axis=1)

        for r in np.unique(labels):
            current_cluster = P[P['labels'] == r]
            distance = []
            for i in range(len(current_cluster)):

                distance.append(self.calc_distance(np.array(current_cluster.iloc[i][:-2]), self.centroids[r]))

            distance = np.array(distance)
            max_dist = distance.argsort()[-2:][::-1]
            self.label_outlier_dist.append(distance[max_dist[-1]])


    def add_tracklet(self, tracklet):
        "Add a new tracklet into the database"
        self.tracklets.append(tracklet)
    

    def build_clustering_model(self):
        "Perform clustering algorithm"
        self.get_features()
        X = np.array(self.feature_set)
        
        self.centroids = np.array(self.k_means(X, self.num_cluster))
        ## set outlier thresholds for each of the centroids
        self.get_outliers()


    def get_cluster_id(self, tracklet):
        """
        Assign the cluster ID for a tracklet. This funciton must return a non-negative integer <= num_cluster
        It is possible to return value 0, but it is reserved for special category of abnormal behavior (for Question 2.3)
        """

        tracklet_centers = []
        boxes = tracklet["tracks"]

        for box in boxes:

            x_center = (box[1] + box[3])/2.0
            y_center = (box[2] + box[4])/2.0
            tracklet_centers.append([x_center, y_center])

        tracklet_centers = np.array(tracklet_centers)
        ind = np.lexsort((tracklet_centers[:,1],tracklet_centers[:,0])) 
        tracklet_centers = tracklet_centers[ind]

        if(len(tracklet_centers) < 4):
                return 0

        tck, u = splprep(tracklet_centers.T, u=None, s=0.0, k=3) 
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck, der=0)


        ic = self.centroids
        distance=[]
        for j in ic:
            distance.append(self.calc_distance(np.concatenate((x_new, y_new), axis=0), j))
        label = np.argmin(distance) + 1

        ## check for outlier detection here
        if self.label_outlier_dist[label-1] <= np.min(distance):
            return 0

        ## output feature set for plotting/debugging
        feature = [np.concatenate((x_new, y_new), axis=0), label]
        self.feature_out.append(feature)
        
        return (int)(label)
