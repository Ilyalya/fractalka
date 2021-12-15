import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

im = Image.open('silver/large.jpg')


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1/3, 1/3, 1/3])


imar = rgb2gray(np.array(im))
sz = imar.shape[0]

def fractal_volume(imar, d_=10) -> list:
    u = imar.copy()
    b = imar.copy()

    footprint = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    ds = range(4, d_)
    vols = []

    for d in ds:
        fst_u = u + 1
        fst_b = b - 1

        scnd_u = maximum_filter(u, mode='constant', footprint=footprint, cval=0)
        scnd_b = minimum_filter(b, mode='constant', footprint=footprint, cval=255)

        u = np.maximum(fst_u, scnd_u)
        b = np.minimum(fst_b, scnd_b)

        vols.append(np.sum(u - b))

    return vols


def __fractal_signature(imar, d_=10) -> float:
    vols = fractal_volume(imar, d_)
    return (vols[-1] - vols[-2]) / 2


def fractal_signature(imar, epsilons=range(4, 30, 2), d=10) -> list:
    local_signatures = []

    for eps in epsilons:
        ads = 0
        for start1, end1 in zip(range(0, imar.shape[0] - eps, eps), range(eps, imar.shape[0], eps)):
            for start2, end2 in zip(range(0, imar.shape[1] - eps, eps), range(eps, imar.shape[1], eps)):
                ads += __fractal_signature(imar[start1:end1, start2:end2], d)

        local_signatures.append(round((ads/100000), 2))
    return local_signatures


h = fractal_signature(imar)
# z = sorted(h)
print(h)
plt.plot(h)
plt.show()


