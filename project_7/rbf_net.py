'''rbf_net.py
Radial Basis Function Neural Network
Francis O'Hara
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import kmeans
from scipy.linalg import lstsq


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        """RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit

        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        prototypes: Hidden unit prototypes (i.e. center)
            shape=(num_hidden_units, num_features)

        sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        are similar to the unit's prototype (i.e. center).
          shape=(num_hidden_units,)
          Larger sigma -> hidden unit becomes active to dissimilar inputs
          Smaller sigma -> hidden unit only becomes active to similar inputs

        wts: Weights connecting hidden and output layer neurons.
          shape=(num_hidden_units+1, num_classes)
          The reason for the +1 is to account for the bias (a hidden unit whose activation is always
          set to 1).
        """
        self.k = num_hidden_units
        self.num_classes = num_classes
        self.prototypes = None
        self.sigmas = None
        self.wts = None

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        sigmas = np.ndarray(shape=(centroids.shape[0]))

        for i in range(centroids.shape[0]):
            cluster_points = data[cluster_assignments == i]
            sigmas[i] = (((cluster_points - centroids[i]) ** 2).sum(axis=1) ** 0.5).sum() / cluster_points.shape[0]
        
        return sigmas


    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmeans_ = kmeans.KMeans(data)
        kmeans_.cluster_batch(self.k, 10)
        self.prototypes = kmeans_.centroids
        self.sigmas = self.avg_cluster_dist(data, kmeans_.centroids, kmeans_.data_centroid_labels, kmeans_)


    def linear_regression(self, A, y):
        '''Performs linear regression. Adapt your SciPy lstsq code from the linear regression project.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept column.
        '''
        A_hat = np.hstack((A, np.ones(shape=(A.shape[0], 1))))
        c, _, _, _ = lstsq(A_hat, y)
        return c



    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        h = np.ndarray(shape=(data.shape[0], self.k))
        for i in range(data.shape[0]):
            for j in range(self.k):
                h[i, j] = np.exp(-1 * ((((self.prototypes[j] - data[i]) ** 2).sum() ** 0.5) ** 2)/((2 * (self.sigmas[j] ** 2)) + (10 ** -8)))
        return h


    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        h_hat = np.hstack((hidden_acts, np.ones(shape=(hidden_acts.shape[0],1))))
        return h_hat @ self.wts


    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)
        h = self.hidden_act(data)

        ideal_z = np.zeros(shape=(len(y), self.num_classes))
        for i in range(ideal_z.shape[0]):
            ideal_z[i, y[i]] = 1

        self.wts = np.ndarray(shape=(self.k + 1, self.num_classes))

        for i in range(self.num_classes):
            self.wts[:, i] = self.linear_regression(h, ideal_z[:, i])




    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        h = self.hidden_act(data)
        output_acts = self.output_act(h)
        return np.argmax(output_acts, axis=1)



    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        return (y == y_pred).sum()/len(y)
