'''knn.py
K-Nearest Neighbors algorithm for classification
Francis O'Hara
CS 251: Data Analysis and Visualization
Fall 2024
'''
import cupy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

from classifier import Classifier

class KNN(Classifier):
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Call superclass constructor
        '''
        super().__init__(num_classes)

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        sample_size = data.shape[0]
        y_pred = np.ndarray(shape=(sample_size))
        
        
        for i in range(sample_size):
            if i == 0:
                start_time = time()
                
            distances = (((self.exemplars - data[i]) ** 2).sum(axis=1)) ** 0.5
            k_closest_classes = self.classes[np.argsort(distances)][:k]
            classes, counts = np.unique(k_closest_classes, return_counts=True)
            predicted_class = classes[np.argsort(counts)][-1]
            y_pred[i] = predicted_class
            
            if i == 0:
                time_1_iteration = time() - start_time
                estimated_total_time = time_1_iteration * sample_size
                print(f"It will take approximately {estimated_total_time} seconds for the KNN model to predict the classes for all {sample_size} observations in the dataset.")
                print("Progress:\n")
            
            print(f"{i}/{sample_size}")
            
        return y_pred
        


    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

        x = np.linspace(start= - 40, stop= 40, num=n_sample_pts)
        y = np.linspace(start= - 40, stop= 40, num=n_sample_pts)

        x_samples, y_samples = np.meshgrid(x, y)
        x_flattened = x_samples.flatten()
        y_flattened = y_samples.flatten()
        pred_input = np.hstack((x_flattened.reshape(x_flattened.shape[0], 1), y_flattened.reshape(y_flattened.shape[0], 1)))
        y_pred = self.predict(pred_input, k).reshape((n_sample_pts, n_sample_pts))
        plt.figure(figsize=(12, 10))
        plt.pcolormesh(x_samples, y_samples, y_pred, cmap=ListedColormap(colors))
        plt.colorbar()
        plt.title("KNN Prediction Landscape")
        plt.show()
        


