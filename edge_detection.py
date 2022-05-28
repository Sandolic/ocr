import numpy as np


def edge_detection(binary_array: np.array) -> np.array:
    """
    Compute edge image array of given binary image array
    :param np.array binary_array: given binary array
    :return np.array
    """

    edge_matrix = np.copy(binary_array)
    for x in range(binary_array.shape[0]):
        for y in range(binary_array.shape[1]):
            # We only care about the left, upper, and upper right pixels of the one we are looking at
            left = 0
            upper = 0
            upper_right = 0

            # Computing difference of intensity between current pixel and left, upper and upper right pixels
            if x > 0:
                left = abs(int(binary_array[x, y]) - int(binary_array[x - 1, y]))
            if y > 0:
                upper = abs(int(binary_array[x, y]) - int(binary_array[x, y - 1]))
            if y > 0 and x < binary_array.shape[0] - 1:
                upper_right = abs(int(binary_array[x, y]) - int(binary_array[x + 1, y - 1]))

            # Current pixel receives maximum of the 3 computed differences
            edge_matrix[x, y] = max(left, upper, upper_right)
    return np.array(edge_matrix)
