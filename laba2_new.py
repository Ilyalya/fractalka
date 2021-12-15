from scipy import ndimage, misc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


fname = 'silver/large.jpg'
image = Image.open(fname).convert("L")
fig = plt.figure()
arr = np.asarray(image)
ascent = plt.imshow(arr, cmap='gray', vmin=0, vmax=255)

# plt.gray()  # show the filtered result in grayscale
# ax2 = fig.add_subplot()  # right side
# ascent = misc.ascent()
result = ndimage.maximum_filter(ascent, size=20)
imar.imshow(result)
plt.show()

# plt.show()