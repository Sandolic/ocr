import kmeans
import labelling
import numpy as np
from PIL import Image


image_array = np.array(Image.open("fire_poster.png"))  # Array of given image
binary_array = kmeans.binarization(image_array)  # Array of binary image
components = labelling.connected_components(binary_array)  # Array of padded boxes of connected-components in the image

# Saving padded boxes for visualization purposes
count = 0
for component in components:
    count += 1
    Image.fromarray(component).convert("L").save("results/result_" + str(count) + ".png")
