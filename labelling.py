import numpy as np
import union_tree
import edge_detection


def connected_components(binary_array: np.array) -> []:
    """
    Compute padded boxes of connected-components of given image binary array using labelling
    :param np.array binary_array: given binary array
    :return []
    """

    # Edge of binary array
    edge_array = edge_detection.edge_detection(binary_array)

    # Padded array for labelling
    height, width = edge_array.shape
    padded_array = np.zeros((height + 2, width + 2))
    padded_array[1:height + 1, 1:width + 1] = edge_array
    padded_array = padded_array.astype("uint32")  # If we have a high number label (>255), uint8 won't be enough

    # Initialize union tree for label-equivalences
    label = 1
    tree = union_tree.UnionTree()

    # Forward mask
    for x in range(1, height + 2):
        for y in range(1, width + 2):
            # Only look labelling white pixels (edges)
            if padded_array[x, y] == 255:
                neighbours = np.array([padded_array[x, y - 1], padded_array[x - 1, y - 1],
                                       padded_array[x - 1, y], padded_array[x - 1, y + 1]])

                # No labelled neighbour
                if np.count_nonzero(neighbours) == 0:
                    padded_array[x, y] = label
                    tree.new_parent(label)
                    label = label + 1
                # One labelled neighbour
                elif np.count_nonzero(neighbours) == 1:
                    padded_array[x, y] = neighbours[np.nonzero(neighbours)]
                # More than one labelled neighbour
                else:
                    indexes = np.where(neighbours != 0)[0]

                    # Always give lowest label to pixel
                    minimum_label = np.min(neighbours[indexes])
                    padded_array[x, y] = minimum_label

                    # All different labels in the mask are equivalent
                    for index in indexes:
                        temp_label = neighbours[index]
                        if temp_label != minimum_label:
                            tree.new_parent(temp_label)
                            tree.union(minimum_label, temp_label)

    # Give lowest possible equivalence :
    # 12 <=> 13, 17 <=> 13 : 17 <=> 12
    tree.flatten()
    tree.new_parent(0)

    # Remove padding, apply lowest labelling using equivalences
    padded_array = padded_array[1:height + 1, 1: width + 1]
    for x in range(height):
        for y in range(width):
            padded_array[x, y] = tree.parents[padded_array[x, y]]

    # Store connected components
    components = {}
    for x in range(height):
        for y in range(width):
            if padded_array[x, y] != 0:
                if padded_array[x, y] in components.keys():
                    components[padded_array[x, y]][0].append(x)
                    components[padded_array[x, y]][1].append(y)
                else:
                    components[padded_array[x, y]] = [[x], [y]]

    return component_boxes(binary_array, components)


def component_boxes(binary_array: np.array, components: {}) -> []:
    """
    Compute padded boxes of computed connected-components
    :param np.array binary_array: original binary array
    :param {} components: computed connected-components
    :return []
    """

    boxes = []

    for component in components.values():
        # Remove potential noise connected-components
        if len(component[0]) > 50:
            x_max = max(component[0])
            x_min = min(component[0])
            y_max = max(component[1])
            y_min = min(component[1])

            x_difference = x_max - x_min
            y_difference = y_max - y_min

            # Remove long lines or other geometrically elongated connected-components
            if abs(x_difference - y_difference) < 50:
                # Padded box containing connected-component from original binary array
                box = np.zeros((x_difference + 20, y_difference + 20))
                box[10:x_difference + 10, 10:y_difference + 10] = binary_array[x_min:x_max, y_min:y_max]
                boxes.append(box)

    return boxes
