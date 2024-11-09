'''kmeans.py
Performs K-Means clustering
Francis O'Hara
CS 251/2: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return ((pt_1 - pt_2) ** 2).sum() ** 0.5

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return ((pt - centroids) ** 2).sum(axis=1) ** 0.5
        

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k
        self.centroids = self.get_data()[np.random.randint(low=0, high=self.get_data().shape[0], size=k)]
        return self.centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')
        
        self.initialize(k)
        n_iter = 0
        while True:
            self.update_labels(self.centroids)
            new_centroids, centroid_diff = self.update_centroids(k, self.data_centroid_labels, self.centroids)
            n_iter += 1
            
            if (n_iter >= 1000) or (abs(centroid_diff).sum() <= tol):
                break
        
        self.compute_inertia()
        return self.inertia, n_iter
    


    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_inertia = self.cluster(k)[0]
        best_centroids = self.centroids
        best_data_centroid_labels = self.data_centroid_labels
        if verbose:
            print(f"inertia={best_inertia}")
            print(f"centroids={best_centroids}")
            print(f"data_centroid_labels={best_data_centroid_labels}")

        for i in range(n_iter-1):
            inertia = self.cluster(k)[0]
            if inertia <= best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids
                best_data_centroid_labels = self.data_centroid_labels
            else:
                self.centroids = best_centroids
                self.data_centroid_labels = best_data_centroid_labels
            
            if verbose:
                print(f"inertia={best_inertia}")
                print(f"centroids={best_centroids}")
                print(f"data_centroid_labels={best_data_centroid_labels}")

        self.inertia = best_inertia
        

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = np.zeros(shape=(self.data.shape[0],), dtype=int)

        for i in range(self.data.shape[0]):
            dists = self.dist_pt_to_centroids(self.data[i, :], centroids)
            labels[i] =  np.argmin(dists)

        self.data_centroid_labels = labels
        return labels
            

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        new_centroids = np.zeros(shape=prev_centroids.shape)
        
        for i in range(new_centroids.shape[0]):
            points = self.data[data_centroid_labels == i]
            if points.size > 0:
                new_centroids[i] = points.sum(axis=0) / points.shape[0]
            else:
                new_centroids[i] = self.data[np.random.randint(0, self.data.shape[0])]
            
        self.centroids = new_centroids
        centroids_diff = new_centroids - prev_centroids

        return new_centroids, centroids_diff
        

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sum_squard_dists = 0
        for i in range(self.data.shape[0]):
            sum_squard_dists += self.dist_pt_to_pt(self.data[i], self.centroids[self.data_centroid_labels[i]]) ** 2
        self.inertia = sum_squard_dists/self.data.shape[0]
        return self.inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
        for i in range(self.k):
            cluster = self.data[self.data_centroid_labels == i]
            plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f"Cluster {i}")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c="black", marker="x", label="Centroid")
        plt.title("Clustered Datapoints")
        # plt.legend()
        plt.show()


    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iters: int. The number of iterations to run kmeans for every given k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        ks = list(range(1, max_k + 1))
        inertias = []
        for k in ks:
            self.cluster_batch(k, n_iter)
            inertias.append(self.inertia)
        plt.plot(ks, inertias, marker="x", mec="black")
        plt.xlabel("k clusters")
        plt.ylabel("inertia")
        plt.title("Elbow Plot of Inertia for Different Ks")
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for i in range(self.data.shape[0]):
            self.data[i] = self.centroids[self.data_centroid_labels[i]]
        
