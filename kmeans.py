import numpy as np
from numpy.linalg import norm
from PIL import Image


# KMeans class
class KMeans:
    def __init__(self, n_clusters: int, max_iter=100):
        """
        KMeans object constructor
        :param int n_clusters: number of clusters, and thus the number of centroids
        :param int max_iter: maximum number of iterations inside the "fit" method
        :return Kmeans
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = np.array([])
        self.labels = np.array([])
        self.labels = np.array([])

    def initialize_centroids(self, centroids: np.array):
        """
        Initialize centroids from given ones
        :param np.array centroids: given centroids
        """

        self.centroids = centroids

    def compute_centroids(self, labels: np.array, data: np.array) -> np.array:
        """
        Compute centroids from data and labels
        :param np.array labels: pixels' memberships to different centroids
        :param np.array data: pixels and their RGB values
        :return np.array
        """

        centroids = np.zeros((self.n_clusters, data.shape[1]))
        for k in range(self.n_clusters):
            if data[labels == k, :].shape[0] != 0:  # In case no pixels belong to a centroid
                centroids[k, :] = np.mean(data[labels == k, :], axis=0)  # Pixels' colours mean belonging to k-centroid
        return centroids

    def compute_distance(self, data: np.array, centroids: np.array) -> np.array:
        """
        Compute distance between data's pixels and each centroids
        :param np.array data: pixels and their RGB values
        :param np.array centroids: centroids
        :return np.array
        """

        distance = np.zeros((data.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(data - centroids[k, :], axis=1)  # Euclidian distance between pixels and k-centroid
            distance[:, k] = np.square(row_norm)
        return distance

    def compute_labels(self, data: np.array, centroids: np.array):
        """
        Compute labels with given centroids
        :param np.array data: pixels and their RGB values
        :param np.array centroids: centroids
        """

        distance = self.compute_distance(data, centroids)
        self.labels = np.argmin(distance, axis=1)  # Closest distance between every pixels and each centroids

    def fit(self, data: np.array):
        """
        K-means method : computing new centroids depending of
        the previous iteration's labels and centroids
        :param np.array data: pixels and their RGB values
        """

        for i in range(self.max_iter):
            old_centroids = self.centroids
            self.compute_labels(data, old_centroids)
            self.centroids = self.compute_centroids(self.labels, data)
            if np.all(old_centroids == self.centroids):  # If the algorithm converges
                break


# Static functions
def compute_local_counter(labels: np.array) -> np.array:
    """
    Compute local counter
    :param np.array labels: pixels' memberships to different centroids
    :return: np.array
    """

    local_counter = np.array([0, 0])
    for k in range(labels.shape[0]):
        slot = labels[k]
        local_counter[slot] += 1
    return local_counter


def compute_local_accumulator(labels: np.array, data: np.array) -> np.array:
    """
    Compute local accumulator
    :param np.array labels: pixels' memberships to different centroids
    :param np.array data: pixels and their RGB values
    :return: np.array
    """

    local_accumulator = np.array([[0, 0, 0], [0, 0, 0]])
    for k in range(labels.shape[0]):
        slot = labels[k]
        local_accumulator[slot, :] += data[k, :]
    return local_accumulator


def compute_global_centroids(counters: np.array, accumulators: np.array) -> np.array:
    """
    Compute global centroids
    :param np.array counters: computed global counters
    :param np.array accumulators: computed global accumulators
    :return: np.array
    """

    global_centroids = np.array([[0, 0, 0], [0, 0, 0]])
    for k in range(2):
        global_centroids[k, :] = np.divide(accumulators[k, :], counters[k])
    return global_centroids


def binarization(image_array: np.array) -> np.array:
    """
    Binarize image from given file path
    :param np.array image_array: given image array
    :return: np.array
    """

    '''
    Instead of finding "centroids" - explained below - in the image itself - which could bias
    results depending on the image itself - we consider a 3-dimensional space where each axes
    are the Red, Green, Blue colour values of each pixel. Thus, the process is only dependant
    from the image's colours.


    Centroid : Virtual point from which data can belong to. With more than one centroid, data
    is divided into "clusters", which can help processing it.
      - In our case : we want to binarize an image, so we will use two centroids (either local or global)
      to regroup pixels into.
    '''

    # Initialization
    data = image_array.reshape(image_array.shape[0] * image_array.shape[1], image_array.shape[2])
    km = KMeans(n_clusters=2)

    # K-means method
    km.initialize_centroids(np.array([[0, 0, 0], [255, 255, 255]]))
    km.fit(data)
    km.compute_labels(data, km.centroids)

    # Reinitializing centroids while keeping the same labels to have black and white result
    km.initialize_centroids(np.array([[0, 0, 0], [255, 255, 255]]))
    binary_array = km.centroids[km.labels]  # Getting black and white array
    binary_array = np.clip(binary_array.astype("uint8"), 0, 255)

    # Reformatting array to image data then greyscaling it
    binary_array = binary_array.reshape(image_array.shape[0], image_array.shape[1], image_array.shape[2])
    binary_array = np.array(Image.fromarray(binary_array).convert("L"))

    return binary_array
