from PIL import Image
import numpy as np

img = Image.open(fr"Cherry-trees\images\Training\bag0histogram_0.png")
histogram = np.array(img.getdata())
print(len(histogram))
histogram = histogram.reshape((32, 16))
for i in range(len(histogram)):
    print(histogram[i])